"""Custom PyTorch layers for news recommendation models."""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Pure-function masking utilities (isomorphic with Keras/JAX)
# ---------------------------------------------------------------------------


def compute_mask(inputs: torch.Tensor) -> torch.Tensor:
    """Compute a boolean mask where non-zero positions are True."""
    return (inputs != 0).float()


def overwrite_mask(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Zero out positions indicated by a mask."""
    return values * mask.unsqueeze(-1)


class AdditiveAttention(nn.Module):
    """Soft-alignment attention layer (additive / Bahdanau-style).

    Computes a weighted sum of the input sequence where the weights are learned.

    Args:
        input_dim: Last dimension of the input tensor (feature size).
        query_vec_dim: Hidden dimension of the attention mechanism.
    """

    def __init__(self, input_dim: int, query_vec_dim: int = 200):
        super().__init__()
        self.W = nn.Parameter(torch.empty(input_dim, query_vec_dim))
        self.b = nn.Parameter(torch.zeros(query_vec_dim))
        self.q = nn.Parameter(torch.empty(query_vec_dim, 1))

        # Glorot / Xavier uniform initialisation
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.q)

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """Forward pass.

        Args:
            inputs: (batch, seq_len, features)
            mask:   (batch, seq_len) bool tensor. ``True`` keeps, ``False`` masks.

        Returns:
            (batch, features) weighted sum.
        """
        # 1. Dense + tanh
        attention_hidden = torch.tanh(torch.matmul(inputs, self.W) + self.b)

        # 2. Project to scalar per timestep
        attention_scores = torch.matmul(attention_hidden, self.q).squeeze(
            -1
        )  # (batch, seq_len)

        # 3. exp + masked normalization (matches official Microsoft implementation)
        attention = torch.exp(attention_scores)
        if mask is not None:
            attention = attention * mask.float()
        attention_weights = attention / (
            attention.sum(dim=-1, keepdim=True) + 1e-7
        )

        # 4. Weighted sum
        attention_weights_expanded = attention_weights.unsqueeze(
            -1
        )  # (batch, seq_len, 1)
        weighted_input = inputs * attention_weights_expanded
        return weighted_input.sum(dim=1)  # (batch, features)


class ComputeMasking(nn.Module):
    """Produce a boolean mask where ``inputs != 0``."""

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return a float mask (1.0 / 0.0) with the same dtype as inputs."""
        return (inputs != 0).float()


class OverwriteMasking(nn.Module):
    """Zero out positions according to a mask.

    Args:
        values: (batch, seq_len, features)
        mask:   (batch, seq_len)  -- 1 keeps, 0 masks.

    Returns:
        values * mask (broadcast over last dim).
    """

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return values * mask.unsqueeze(-1)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer-style models.

    Adds deterministic sinusoidal position information to input embeddings
    so that attention mechanisms can distinguish token positions.

    Args:
        dropout_rate: Dropout applied after adding positional encoding.
        max_len: Maximum sequence length to generate encodings for.
    """

    def __init__(self, d_model: int, dropout_rate: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)

        # Build the sinusoidal table once
        position = np.arange(0, max_len, dtype=np.float32).reshape(-1, 1)
        div_term = np.exp(
            np.arange(0, d_model, 2, dtype=np.float32) * (-math.log(10000.0) / d_model)
        )

        pe = np.zeros((1, max_len, d_model), dtype=np.float32)
        pe[0, :, 0::2] = np.sin(position * div_term)
        pe[0, :, 1::2] = np.cos(position * div_term)

        # Register as buffer (not a parameter -- not trained)
        self.register_buffer("pe", torch.from_numpy(pe))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to *x* and apply dropout.

        Args:
            x: (batch, seq_len, d_model)

        Returns:
            (batch, seq_len, d_model) with positional information added.
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class MultiHeadAttentionBlock(nn.Module):
    """Multi-head attention block (MAB) with residual connections and layer norm.

    Implements MAB(Q, K) = LN(H + FF(H)) where H = LN(Q + MHA(Q, K, K)).

    Args:
        dim_out: Output / hidden dimension (must be divisible by *num_heads*).
        num_heads: Number of attention heads.
        use_layer_norm: Whether to apply layer normalisation.
        dropout_rate: Dropout rate for attention weights and feed-forward.
    """

    def __init__(
        self,
        dim_out: int,
        num_heads: int,
        use_layer_norm: bool = True,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.dim_out = dim_out
        self.num_heads = num_heads
        self.use_layer_norm = use_layer_norm

        self.fc_q = nn.Linear(dim_out, dim_out)
        self.fc_k = nn.Linear(dim_out, dim_out)
        self.fc_v = nn.Linear(dim_out, dim_out)
        self.fc_o = nn.Linear(dim_out, dim_out)

        if use_layer_norm:
            self.ln0 = nn.LayerNorm(dim_out)
            self.ln1 = nn.LayerNorm(dim_out)

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass (self-attention: Q = K = inputs).

        Args:
            inputs: (batch, seq_len, dim_out)

        Returns:
            (batch, seq_len, dim_out)
        """
        Q = K = inputs

        Q_proj = self.fc_q(Q)
        K_proj = self.fc_k(K)
        V_proj = self.fc_v(K)

        B, S_q, _ = Q_proj.shape
        S_k = K_proj.size(1)
        head_dim = self.dim_out // self.num_heads

        # Reshape to (B, heads, seq, head_dim)
        Q_proj = Q_proj.view(B, S_q, self.num_heads, head_dim).transpose(1, 2)
        K_proj = K_proj.view(B, S_k, self.num_heads, head_dim).transpose(1, 2)
        V_proj = V_proj.view(B, S_k, self.num_heads, head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q_proj, K_proj.transpose(-2, -1)) / math.sqrt(head_dim)
        attn_weights = self.dropout(torch.softmax(scores, dim=-1))
        attn_out = torch.matmul(attn_weights, V_proj)

        # Concat heads
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S_q, self.dim_out)

        O = self.dropout(self.fc_o(attn_out))
        O = O + Q  # residual
        if self.use_layer_norm:
            O = self.ln0(O)

        # Feed-forward + residual
        O_ff = self.fc_o(F.relu(O))
        O = O + O_ff
        if self.use_layer_norm:
            O = self.ln1(O)

        return O


class GraphSAGELayer(nn.Module):
    """GraphSAGE layer for bipartite user-news graphs (CROWN model).

    Performs mutual message passing between user nodes and news nodes
    on a bipartite adjacency matrix.

    Args:
        user_dim: Input dimension of user features.
        news_dim: Input dimension of news features.
        units: Output dimension for both user and news embeddings.
        aggregator: Aggregation type (``'mean'``, ``'max'``, ``'sum'``).
        dropout_rate: Dropout rate.
        normalize: Whether to L2-normalise the output embeddings.
    """

    def __init__(
        self,
        user_dim: int,
        news_dim: int,
        units: int,
        aggregator: str = "mean",
        dropout_rate: float = 0.0,
        normalize: bool = True,
    ):
        super().__init__()
        self.units = units
        self.aggregator = aggregator
        self.normalize = normalize

        self.W_user_self = nn.Parameter(torch.empty(user_dim, units))
        self.W_user_neigh = nn.Parameter(torch.empty(news_dim, units))
        self.b_user = nn.Parameter(torch.zeros(units))

        self.W_news_self = nn.Parameter(torch.empty(news_dim, units))
        self.W_news_neigh = nn.Parameter(torch.empty(user_dim, units))
        self.b_news = nn.Parameter(torch.zeros(units))

        nn.init.xavier_uniform_(self.W_user_self)
        nn.init.xavier_uniform_(self.W_user_neigh)
        nn.init.xavier_uniform_(self.W_news_self)
        nn.init.xavier_uniform_(self.W_news_neigh)

        self.dropout = nn.Dropout(dropout_rate)

    def _aggregate(
        self, adjacency_weights: torch.Tensor, neighbor_features: torch.Tensor
    ) -> torch.Tensor:
        """Aggregate neighbor features according to the chosen strategy."""
        if self.aggregator == "mean":
            neighbor_sum = torch.matmul(adjacency_weights, neighbor_features)
            degree = adjacency_weights.sum(dim=-1, keepdim=True).clamp(min=1e-7)
            return neighbor_sum / degree
        elif self.aggregator == "sum":
            return torch.matmul(adjacency_weights, neighbor_features)
        elif self.aggregator == "max":
            # Expand for element-wise masking
            expanded_adj = adjacency_weights.unsqueeze(-1)  # (B, N, K, 1)
            expanded_neigh = neighbor_features.unsqueeze(1)  # (B, 1, K, D)
            masked = expanded_neigh * expanded_adj
            mask = expanded_adj > 0
            masked = torch.where(mask, masked, torch.tensor(-1e9, device=masked.device))
            return masked.max(dim=2).values
        else:
            raise ValueError(f"Unknown aggregator: {self.aggregator}")

    def forward(
        self,
        user_features: torch.Tensor,
        news_features: torch.Tensor,
        adjacency_matrix: torch.Tensor,
    ) -> tuple:
        """Forward pass.

        Args:
            user_features: (B, num_users, user_dim)
            news_features: (B, num_news, news_dim)
            adjacency_matrix: (B, num_users+num_news, num_users+num_news)

        Returns:
            Tuple of (updated_users, updated_news), each (B, N, units).
        """
        num_users = user_features.size(1)

        user_to_news = adjacency_matrix[:, :num_users, num_users:]
        news_to_user = adjacency_matrix[:, num_users:, :num_users]

        # User update
        news_for_users = self._aggregate(user_to_news, news_features)
        updated_users = F.relu(
            torch.matmul(user_features, self.W_user_self)
            + torch.matmul(news_for_users, self.W_user_neigh)
            + self.b_user
        )
        updated_users = self.dropout(updated_users)

        # News update
        users_for_news = self._aggregate(news_to_user, user_features)
        updated_news = F.relu(
            torch.matmul(news_features, self.W_news_self)
            + torch.matmul(users_for_news, self.W_news_neigh)
            + self.b_news
        )
        updated_news = self.dropout(updated_news)

        if self.normalize:
            updated_users = F.normalize(updated_users, p=2, dim=-1)
            updated_news = F.normalize(updated_news, p=2, dim=-1)

        return updated_users, updated_news


class GraphAttentionLayer(nn.Module):
    """Graph Attention Network (GAT) layer for bipartite user-news graphs.

    Args:
        user_dim: Input dimension of user features.
        news_dim: Input dimension of news features.
        units: Output dimension.
        num_heads: Number of attention heads.
        dropout_rate: Dropout rate.
        alpha: Negative slope for LeakyReLU in attention.
        concat_heads: If True, concatenate heads (units must be divisible by num_heads);
                       otherwise average them.
    """

    def __init__(
        self,
        user_dim: int,
        news_dim: int,
        units: int,
        num_heads: int = 1,
        dropout_rate: float = 0.0,
        alpha: float = 0.2,
        concat_heads: bool = True,
    ):
        super().__init__()
        self.units = units
        self.num_heads = num_heads
        self.alpha = alpha
        self.concat_heads = concat_heads

        if concat_heads:
            assert units % num_heads == 0, (
                "units must be divisible by num_heads when concat_heads=True"
            )
            self.head_dim = units // num_heads
        else:
            self.head_dim = units

        # Learnable linear projections per head
        self.W_user = nn.Parameter(torch.empty(num_heads, user_dim, self.head_dim))
        self.W_news = nn.Parameter(torch.empty(num_heads, news_dim, self.head_dim))
        self.a_user_news = nn.Parameter(torch.empty(num_heads, self.head_dim * 2, 1))
        self.a_news_user = nn.Parameter(torch.empty(num_heads, self.head_dim * 2, 1))

        out_dim = units if concat_heads else self.head_dim
        self.b_user = nn.Parameter(torch.zeros(out_dim))
        self.b_news = nn.Parameter(torch.zeros(out_dim))

        for p in [self.W_user, self.W_news, self.a_user_news, self.a_news_user]:
            nn.init.xavier_uniform_(p.view(p.size(0), -1))

        self.dropout = nn.Dropout(dropout_rate)

    def _attention_coeffs(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        a_weights: torch.Tensor,
        adj_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute GAT attention coefficients."""
        # queries: (B, H, Nq, D), keys: (B, H, Nk, D)
        Nq = queries.size(2)
        Nk = keys.size(2)

        q_exp = queries.unsqueeze(3).expand(-1, -1, -1, Nk, -1)
        k_exp = keys.unsqueeze(2).expand(-1, -1, Nq, -1, -1)
        concat_qk = torch.cat([q_exp, k_exp], dim=-1)  # (B, H, Nq, Nk, 2D)

        scores = torch.matmul(concat_qk, a_weights.unsqueeze(0)).squeeze(
            -1
        )  # (B, H, Nq, Nk)
        scores = F.leaky_relu(scores, negative_slope=self.alpha)

        adj_exp = adj_mask.unsqueeze(1)  # (B, 1, Nq, Nk)
        scores = scores.masked_fill(adj_exp <= 0, -1e9)
        coeffs = self.dropout(torch.softmax(scores, dim=-1))
        return coeffs

    def forward(
        self,
        user_features: torch.Tensor,
        news_features: torch.Tensor,
        adjacency_matrix: torch.Tensor,
    ) -> tuple:
        """Forward pass.

        Args:
            user_features: (B, num_users, user_dim)
            news_features: (B, num_news, news_dim)
            adjacency_matrix: (B, num_users+num_news, num_users+num_news)

        Returns:
            Tuple of (updated_users, updated_news).
        """
        B = user_features.size(0)
        num_users = user_features.size(1)
        num_news = news_features.size(1)

        user_to_news = adjacency_matrix[:, :num_users, num_users:]
        news_to_user = adjacency_matrix[:, num_users:, :num_users]

        # Project per head: einsum 'bud,hdk->bhuk'
        user_t = torch.einsum("bud,hdk->bhuk", user_features, self.W_user)
        news_t = torch.einsum("bnd,hdk->bhnk", news_features, self.W_news)

        # User ← News attention
        un_attn = self._attention_coeffs(user_t, news_t, self.a_user_news, user_to_news)
        user_updates = torch.einsum("bhun,bhnk->bhuk", un_attn, news_t)

        # News ← User attention
        nu_attn = self._attention_coeffs(news_t, user_t, self.a_news_user, news_to_user)
        news_updates = torch.einsum("bhnu,bhuk->bhnk", nu_attn, user_t)

        if self.concat_heads:
            updated_users = (
                user_updates.permute(0, 2, 1, 3)
                .contiguous()
                .view(B, num_users, self.units)
            )
            updated_news = (
                news_updates.permute(0, 2, 1, 3)
                .contiguous()
                .view(B, num_news, self.units)
            )
        else:
            updated_users = user_updates.mean(dim=1)
            updated_news = news_updates.mean(dim=1)

        updated_users = self.dropout(F.relu(updated_users + self.b_user))
        updated_news = self.dropout(F.relu(updated_news + self.b_news))

        return updated_users, updated_news
