"""CROWN news encoder (PyTorch) — encoder-agnostic, two text views.

Pipeline (paper §3.1):
    1. text_encoder(news_idx) -> tokens, mask  (title and abstract have
       their own TextEncoder instance)
    2. Optional Linear(text_dim -> embedding_size) per view
    3. Positional encoding + dropout
    4. ``nn.TransformerEncoder`` (paper: 1 layer, 10 heads)
    5. Mask-weighted mean pool over sequence
    6. Category-aware concatenation with [cat | subcat] -> Linear
    7. k-intent disentanglement (k separate FC -> ReLU)
    8. Additive attention over k intents
    9. Title-body cosine similarity scaling
    10. Concatenate ``[title_intent, sim * body_intent, cat_emb, subcat_emb]``

Also computes the auxiliary category-prediction loss used by
:meth:`CROWN.get_auxiliary_loss`.

The encoder accepts a packed integer feature tensor with columns
``[news_idx | category | subcategory]``, identical for training and
eval call sites. No ``if encoder_type`` branches — the TextEncoder
factory picks GloVe vs PLM at construction time.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.core.models.configs import CROWNConfig

from ...layers import TextEncoder


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (standard transformer)."""

    def __init__(self, d_model: int, dropout_rate: float = 0.1, max_len: int = 512):
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, : x.size(1)])


class AdditiveAttention(nn.Module):
    """Additive (Bahdanau-style) attention: ``tanh(Wx+b) @ q`` -> weighted sum."""

    def __init__(self, feature_dim: int, attention_dim: int):
        super().__init__()
        self.affine = nn.Linear(feature_dim, attention_dim)
        self.query = nn.Linear(attention_dim, 1, bias=False)

    def forward(
        self, features: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        scores = self.query(torch.tanh(self.affine(features))).squeeze(-1)  # (B, L)
        if mask is not None:
            scores = scores.masked_fill(~mask.bool(), -1e9)
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)  # (B, L, 1)
        return (features * weights).sum(dim=1)


class NewsEncoder(nn.Module):
    """CROWN news encoder. Outputs ``(*leading, news_embedding_dim)``.

    Args:
        config: CROWN hyperparameters.
        title_text_encoder: TextEncoder for the title view.
        abstract_text_encoder: TextEncoder for the abstract view.
        category_embedding: ``nn.Embedding`` for category ids.
        subcategory_embedding: ``nn.Embedding`` for subcategory ids.
    """

    def __init__(
        self,
        config: CROWNConfig,
        title_text_encoder: TextEncoder,
        abstract_text_encoder: TextEncoder,
        category_embedding: nn.Embedding,
        subcategory_embedding: nn.Embedding,
    ):
        super().__init__()
        self.config = config
        self.title_text_encoder = title_text_encoder
        self.abstract_text_encoder = abstract_text_encoder
        self.category_embedding = category_embedding
        self.subcategory_embedding = subcategory_embedding

        # Option-2 PLM: the Transformer encoder + intent disentanglement
        # run at the text encoder's native dim (300 for GloVe, 768 for
        # BERT-base). The only post-pool projection is implicit in the
        # intent_layers (text_dim+cat_dim -> intent_dim), which collapse
        # to news_dim-shaped output. Preserves the BERT signal through
        # the per-news Transformer + mean pool.
        t_dim = title_text_encoder.output_dim
        a_dim = abstract_text_encoder.output_dim
        # CROWN runs a single Transformer block shared across views, so
        # title and abstract must have the same per-token dim. With the
        # standard TextEncoder factory both views come from the same
        # backbone (GloVe or BERT-base) so this holds.
        if t_dim != a_dim:
            raise ValueError(
                f"CROWN expects title_text_encoder.output_dim ({t_dim}) "
                f"to match abstract_text_encoder.output_dim ({a_dim})."
            )
        if t_dim % config.num_heads != 0:
            raise ValueError(
                f"CROWN Transformer: text_dim={t_dim} not divisible by "
                f"num_heads={config.num_heads}. Override "
                "spec.model.architecture.news_encoder.num_heads in the "
                f"experiment yaml to a divisor of {t_dim} (e.g. 12 for "
                "BERT-base 768d -> head_dim 64)."
            )
        self.text_dim = int(t_dim)

        self.news_embedding_dim = (
            config.intent_embedding_dim * 2
            + config.category_embedding_dim
            + config.subcategory_embedding_dim
        )

        self.dropout = nn.Dropout(config.dropout_rate)

        self.title_pos = PositionalEncoding(
            t_dim, config.dropout_rate, config.max_title_length
        )
        self.body_pos = PositionalEncoding(
            t_dim, config.dropout_rate, config.max_abstract_length
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=t_dim,
            nhead=config.num_heads,
            dim_feedforward=config.feedforward_dim,
            dropout=config.dropout_rate,
            batch_first=True,
        )
        self.title_transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=config.num_layers
        )
        self.body_transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=config.num_layers
        )

        cat_concat_dim = (
            config.category_embedding_dim + config.subcategory_embedding_dim
        )
        self.category_affine = nn.Linear(cat_concat_dim, config.category_embedding_dim)

        # Intent layers map text_dim+cat_dim -> intent_dim. This is the
        # only post-pool projection for CROWN — collapses BERT to intent_dim.
        intent_input_dim = t_dim + config.category_embedding_dim
        self.intent_layers = nn.ModuleList(
            [
                nn.Linear(intent_input_dim, config.intent_embedding_dim)
                for _ in range(config.intent_num)
            ]
        )

        self.title_intent_attn = AdditiveAttention(
            config.intent_embedding_dim, config.attention_dim
        )
        self.body_intent_attn = AdditiveAttention(
            config.intent_embedding_dim, config.attention_dim
        )

        # Placeholder — ``CROWN.__init__`` overrides this once
        # ``num_categories`` is known.
        self.category_predictor = nn.Linear(config.intent_embedding_dim, 1)

        # Accumulated during forward, reset each call.
        self.auxiliary_loss = torch.tensor(0.0)

    @staticmethod
    def valid_mask(packed_features: torch.Tensor) -> torch.Tensor:
        """A news slot is valid when its parsed news_idx (column 0) is non-zero."""
        if packed_features.dim() >= 2 and packed_features.shape[-1] > 1:
            return packed_features[..., 0] != 0
        return packed_features != 0

    def _split_columns(
        self, packed: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(news_idx, category_id, subcategory_id)`` from packed
        ``[news_idx | category | subcategory]``."""
        news_idx = packed[..., 0]
        category_id = packed[..., 1]
        subcategory_id = packed[..., 2]
        return news_idx, category_id, subcategory_id

    def _k_intent_disentangle(self, x: torch.Tensor) -> torch.Tensor:
        """``(N, input_dim) -> (N, k, intent_dim)``"""
        intents = [F.relu(layer(x)).unsqueeze(1) for layer in self.intent_layers]
        return torch.cat(intents, dim=1)

    def forward(
        self,
        packed_features: torch.Tensor,
        *,
        compute_aux_loss: bool = False,
        training: bool = True,
    ) -> torch.Tensor:
        """Encode a batch of news articles.

        Args:
            packed_features: ``(*leading, 3)`` int with column order
                ``[news_idx | category_id | subcategory_id]``.
            compute_aux_loss: When True, populates
                ``self.auxiliary_loss`` from title-intent category
                prediction.

        Returns:
            ``(*leading, news_embedding_dim)`` news representations.
        """
        leading_shape = packed_features.shape[:-1]
        flat = packed_features.reshape(-1, packed_features.shape[-1])
        news_idx, category_ids, subcategory_ids = self._split_columns(flat)

        title_tokens, title_mask = self.title_text_encoder(news_idx)
        body_tokens, body_mask = self.abstract_text_encoder(news_idx)
        title_emb = self.dropout(title_tokens)  # (N, T, text_dim)
        body_emb = self.dropout(body_tokens)
        title_mask = title_mask.bool()
        body_mask = body_mask.bool()

        title_emb = self.title_pos(title_emb)
        body_emb = self.body_pos(body_emb)

        title_enc = self.title_transformer(title_emb)  # (N, T, E)
        body_enc = self.body_transformer(body_emb)  # (N, A, E)

        # Unmasked mean pool — matches reference
        # ``crown-www25/newsEncoders.py:164,168`` which uses plain
        # ``.mean(dim=1)`` over the transformer output. For GloVe with
        # ``pad_id=0`` this is a (T_valid / T) scaling vs the masked
        # mean; for PLM/BERT pad tokens have non-zero outputs so the
        # divergence is larger. The masked variant flattened the intent
        # signal in our ablations; reverted to plain mean.
        title_pool = title_enc.mean(dim=1)
        body_pool = body_enc.mean(dim=1)

        cat_emb = self.category_embedding(category_ids.long())
        subcat_emb = self.subcategory_embedding(subcategory_ids.long())
        cat_repr = self.category_affine(torch.cat([cat_emb, subcat_emb], dim=-1))

        cat_aware_title = torch.cat([title_pool, cat_repr], dim=-1)
        cat_aware_body = torch.cat([body_pool, cat_repr], dim=-1)

        title_k = self._k_intent_disentangle(cat_aware_title)
        body_k = self._k_intent_disentangle(cat_aware_body)

        title_intent = self.title_intent_attn(title_k)
        body_intent = self.body_intent_attn(body_k)

        if compute_aux_loss:
            logits = self.category_predictor(title_intent)
            self.auxiliary_loss = F.cross_entropy(logits, category_ids.long())
        else:
            self.auxiliary_loss = torch.tensor(0.0, device=news_idx.device)

        sim = F.cosine_similarity(title_intent, body_intent, dim=-1)
        sim = (sim + 1.0) / 2.0

        news_repr = torch.cat(
            [
                title_intent,
                sim.unsqueeze(-1) * body_intent,
                self.dropout(cat_emb),
                self.dropout(subcat_emb),
            ],
            dim=-1,
        )

        # Paper §4.3: only the title intent is used for click prediction.
        self._last_title_intent = title_intent

        return news_repr.reshape(*leading_shape, self.news_embedding_dim)
