"""CROWN user encoder (Flax NNX) — candidate-aware, GNN selectable.

Mirror of the PyTorch CROWN user encoder. Implements the candidate-aware
scaled-dot attention from ``seongeunryu/crown-www25/userEncoders.py``:
``Q = candidate``, ``K/V = GNN-pooled history``, output is a
per-candidate user vector ``(B, C, D)``.

GNN selectable via ``config.gnn_type``:
    * ``"gat"``       — matches the paper TEXT (§4.2.(3-b): "We use GAT").
                        Multi-head bipartite GAT with self-loops.
    * ``"graphsage"`` — matches the reference CODE
                        (``GraphSAGE(num_layers=1)`` in PyG).
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from flax import nnx

from src.core.models.configs import CROWNConfig

from .news_encoder import NewsEncoder


class BipartiteSAGELayer(nnx.Module):
    """1-layer bipartite GraphSAGE (mean aggregator) — reference-faithful.

    With a single shared neighbour (the user-proxy node attached to all
    history slots), the SAGE mean aggregator collapses to::

        user_new[b]   = ReLU(W_self · user[b]   + W_neigh · mean_h(news[b,h]))
        news_new[b,h] = ReLU(W_self · news[b,h] + W_neigh · user[b])

    PyG's ``SAGEConv`` defaults to ``normalize=False`` and PyG's
    ``GraphSAGE`` wrapper applies dropout only **between** layers — so
    with ``num_layers=1`` (the reference setting) the SAGE output is
    neither L2-normalised nor dropped. An earlier version did both,
    which capped ``K = self.K(gcn_news)`` magnitude and flattened the
    candidate-attention softmax. Removed to match the reference.
    """

    def __init__(self, dim: int, dropout_rate: float, *, rngs: nnx.Rngs):
        self.W_self = nnx.Linear(dim, dim, rngs=rngs)
        self.W_neigh = nnx.Linear(dim, dim, rngs=rngs)

    def __call__(
        self,
        user: jax.Array,
        news: jax.Array,
        news_mask: jax.Array,
        *,
        deterministic: bool = True,
    ) -> tuple[jax.Array, jax.Array]:
        m = news_mask[:, :, None].astype(news.dtype)  # (B, H, 1)
        news_count = jnp.maximum(jnp.sum(m, axis=1), 1.0)  # (B, 1)
        news_mean = jnp.sum(news * m, axis=1) / news_count  # (B, D)

        user_new = jax.nn.relu(self.W_self(user) + self.W_neigh(news_mean))
        news_new = jax.nn.relu(
            self.W_self(news)
            + self.W_neigh(jnp.broadcast_to(user[:, None, :], news.shape))
        )
        return user_new, news_new


class BipartiteGATLayer(nnx.Module):
    """1-layer multi-head bipartite GAT with self-loops.

    Matches the paper TEXT (§4.2.(3-b): "We use GAT"). Mirrors the
    PyTorch :class:`BipartiteGATLayer`: per-head coefficients
    ``e_ij = LeakyReLU(a_src · Wh_i + a_dst · Wh_j)``; user attends over
    ``{self, all valid news}``; each news attends over ``{self, user}``.
    With ``concat=True`` head outputs are concatenated so the layer is
    dim-preserving (``num_heads * head_dim == dim``).
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        dropout_rate: float,
        alpha: float = 0.2,
        concat: bool = True,
        *,
        rngs: nnx.Rngs,
    ):
        if dim % num_heads != 0:
            raise ValueError(
                f"BipartiteGATLayer: dim ({dim}) must be divisible by "
                f"num_heads ({num_heads})."
            )
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dim = dim
        self.alpha = alpha
        self.concat = concat

        self.W = nnx.Linear(dim, dim, use_bias=False, rngs=rngs)
        glorot = nnx.initializers.glorot_uniform()
        self.a_src = nnx.Param(glorot(rngs.params(), (num_heads, self.head_dim)))
        self.a_dst = nnx.Param(glorot(rngs.params(), (num_heads, self.head_dim)))
        self.dropout = nnx.Dropout(rate=dropout_rate, rngs=rngs)

    def __call__(
        self,
        user: jax.Array,
        news: jax.Array,
        news_mask: jax.Array,
        *,
        deterministic: bool = True,
    ) -> tuple[jax.Array, jax.Array]:
        B, H, D = news.shape
        NH, HD = self.num_heads, self.head_dim
        neg_inf = jnp.finfo(news.dtype).min

        Wu = self.W(user).reshape(B, NH, HD)
        Wn = self.W(news).reshape(B, H, NH, HD)

        src_u = jnp.sum(Wu * self.a_src.value, axis=-1)
        dst_u = jnp.sum(Wu * self.a_dst.value, axis=-1)
        src_n = jnp.sum(Wn * self.a_src.value, axis=-1)
        dst_n = jnp.sum(Wn * self.a_dst.value, axis=-1)

        # User update: attend over {self, all news}.
        e_un = jax.nn.leaky_relu(
            src_u[:, None, :] + dst_n, negative_slope=self.alpha
        )  # (B, H, NH)
        e_uu = jax.nn.leaky_relu(src_u + dst_u, negative_slope=self.alpha)  # (B, NH)
        e_un = jnp.where(news_mask[:, :, None], e_un, neg_inf)
        all_u = jnp.concatenate([e_un, e_uu[:, None, :]], axis=1)  # (B, H+1, NH)
        attn_u = self.dropout(
            jax.nn.softmax(all_u, axis=1), deterministic=deterministic
        )
        u_from_n = jnp.sum(attn_u[:, :H, :, None] * Wn, axis=1)  # (B, NH, HD)
        u_self = attn_u[:, H, :, None] * Wu
        user_new = jax.nn.elu(u_from_n + u_self)  # (B, NH, HD)

        # News update: each news attends over {self, user}.
        e_nu = jax.nn.leaky_relu(
            src_n + dst_u[:, None, :], negative_slope=self.alpha
        )  # (B, H, NH)
        e_nn = jax.nn.leaky_relu(src_n + dst_n, negative_slope=self.alpha)
        stacked = jnp.stack([e_nu, e_nn], axis=-2)  # (B, H, 2, NH)
        attn_n = self.dropout(
            jax.nn.softmax(stacked, axis=-2), deterministic=deterministic
        )
        n_from_u = attn_n[:, :, 0, :, None] * Wu[:, None, :, :]  # (B, H, NH, HD)
        n_self = attn_n[:, :, 1, :, None] * Wn
        news_new = jax.nn.elu(n_from_u + n_self)  # (B, H, NH, HD)

        if self.concat:
            user_new = user_new.reshape(B, NH * HD)
            news_new = news_new.reshape(B, H, NH * HD)
        else:
            user_new = jnp.mean(user_new, axis=1)
            news_new = jnp.mean(news_new, axis=2)

        return user_new, news_new


class UserEncoder(nnx.Module):
    """CROWN user encoder (reference-faithful, candidate-aware)."""

    def __init__(
        self,
        config: CROWNConfig,
        news_encoder: NewsEncoder,
        *,
        rngs: nnx.Rngs,
    ):
        self.config = config
        self.news_encoder = news_encoder
        D = news_encoder.news_embedding_dim
        A = config.user_attention_dim
        self.D = D
        self.attention_dim = A
        self.scale = math.sqrt(float(A))

        # Zero-init user proxy (reference: ``zeros([batch_size, D])``).
        self.user_proxy = nnx.Param(jnp.zeros((D,)))

        gnn_type = (config.gnn_type or "graphsage").lower()
        if gnn_type == "gat":
            self.gnn: nnx.Module = BipartiteGATLayer(
                dim=D,
                num_heads=config.gat_num_heads,
                dropout_rate=config.dropout_rate,
                alpha=config.gat_alpha,
                concat=config.gat_concat_heads,
                rngs=rngs,
            )
        elif gnn_type == "graphsage":
            self.gnn = BipartiteSAGELayer(
                dim=D, dropout_rate=config.dropout_rate, rngs=rngs
            )
        else:
            raise ValueError(
                f"CROWN gnn_type must be 'gat' or 'graphsage'; got {config.gnn_type!r}."
            )

        # Candidate-aware scaled-dot attention. Reference has K w/o bias,
        # Q w/ bias and xavier init.
        self.K = nnx.Linear(D, A, use_bias=False, rngs=rngs)
        self.Q = nnx.Linear(D, A, use_bias=True, rngs=rngs)
        glorot = nnx.initializers.glorot_uniform()
        self.K.kernel.value = glorot(rngs.params(), (D, A))
        self.Q.kernel.value = glorot(rngs.params(), (D, A))
        self.Q.bias.value = jnp.zeros((A,))

    # ------------------------------------------------------------------
    # Primitives reused by training and the custom eval.
    # ------------------------------------------------------------------
    def run_gnn(
        self,
        news: jax.Array,
        history_mask: jax.Array,
        *,
        training: bool = False,
    ) -> jax.Array:
        """1-layer bipartite GNN (GAT or SAGE) over ``(user_proxy, news)``.

        Returns ``(B, H, D)`` GNN-updated history reps.
        """
        det = not training
        B = news.shape[0]
        user = jnp.broadcast_to(self.user_proxy.value[None, :], (B, self.D))
        _, gcn_news = self.gnn(user, news, history_mask, deterministic=det)
        return gcn_news

    def candidate_attention(
        self,
        gcn_news: jax.Array,
        candidate_repr: jax.Array,
        history_mask: jax.Array,
    ) -> jax.Array:
        """Candidate-aware scaled dot-product attention -> ``(B, C, D)``."""
        K = self.K(gcn_news)  # (B, H, A)
        Q = self.Q(candidate_repr)  # (B, C, A)
        # scores[b, c, h] = K[b, h, :] · Q[b, c, :] / sqrt(A)
        scores = jnp.einsum("bha,bca->bch", K, Q) / self.scale  # (B, C, H)
        # No history-mask on the softmax — matches reference
        # ``crown-www25/userEncoders.py:107`` (``F.softmax(a, dim=1)``).
        # ``history_mask`` is intentionally unused here.
        del history_mask
        alpha = jax.nn.softmax(scores, axis=-1)  # (B, C, H)
        return jnp.einsum("bch,bhd->bcd", alpha, gcn_news)  # (B, C, D)

    # ------------------------------------------------------------------
    # Training entry.
    # ------------------------------------------------------------------
    def forward_with_candidates(
        self,
        history_packed: jax.Array,
        history_mask: jax.Array,
        candidate_repr: jax.Array,
        *,
        training: bool = False,
    ) -> jax.Array:
        """Training: ``(B, H, 3)`` packed history + ``(B, C, D)`` cands -> ``(B, C, D)``."""
        news_history = self.news_encoder(
            history_packed, compute_aux_loss=False, training=training
        )
        gcn_news = self.run_gnn(news_history, history_mask, training=training)
        return self.candidate_attention(gcn_news, candidate_repr, history_mask)

    # ------------------------------------------------------------------
    # Eval entry — returns GNN-updated history reps; the custom evaluator
    # runs ``candidate_attention`` once candidates are known.
    # ------------------------------------------------------------------
    def __call__(
        self, packed_features: jax.Array, *, training: bool = False
    ) -> jax.Array:
        history_mask = self.news_encoder.valid_mask(packed_features)
        news_history = self.news_encoder(
            packed_features, compute_aux_loss=False, training=training
        )
        return self.run_gnn(news_history, history_mask, training=training)
