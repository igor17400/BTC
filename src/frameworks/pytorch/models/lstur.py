"""LSTUR (Long- and Short-term User Representations) -- PyTorch."""

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.core.models.configs import LSTURConfig

from ..layers import (
    AdditiveAttention,
    ComputeMasking,
    OverwriteMasking,
    PLMTokenCNNEncoder,
)
from .base import BaseModel

# ------------------------------------------------------------------
# Sub-encoders
# ------------------------------------------------------------------


class CategoryEncoder(nn.Module):
    """Simple embedding for category IDs (LSTUR-style, no projection)."""

    def __init__(self, config: LSTURConfig, num_categories: int):
        super().__init__()
        self.embedding = nn.Embedding(num_categories + 1, config.category_embedding_dim)

    def forward(self, inputs: torch.Tensor, training: bool = True) -> torch.Tensor:
        embedded = self.embedding(inputs)  # (B, 1, dim)
        return embedded.squeeze(1)  # (B, dim)


class SubcategoryEncoder(nn.Module):
    """Simple embedding for subcategory IDs."""

    def __init__(self, config: LSTURConfig, num_subcategories: int):
        super().__init__()
        self.embedding = nn.Embedding(
            num_subcategories + 1, config.subcategory_embedding_dim
        )

    def forward(self, inputs: torch.Tensor, training: bool = True) -> torch.Tensor:
        embedded = self.embedding(inputs)
        return embedded.squeeze(1)


class NewsEncoder(nn.Module):
    """CNN-based news encoder with optional category/subcategory concatenation."""

    def __init__(
        self,
        config: LSTURConfig,
        embedding_layer: nn.Embedding,
        category_encoder: CategoryEncoder | None = None,
        subcategory_encoder: SubcategoryEncoder | None = None,
    ):
        super().__init__()
        self.config = config
        self.embedding_layer = embedding_layer
        self.category_encoder = category_encoder
        self.subcategory_encoder = subcategory_encoder

        self.dropout1 = nn.Dropout(config.dropout_rate)
        self.cnn = nn.Conv1d(
            in_channels=config.embedding_size,
            out_channels=config.cnn_filter_num,
            kernel_size=config.cnn_kernel_size,
            padding="same",
        )
        self.dropout2 = nn.Dropout(config.dropout_rate)

        self.compute_masking = ComputeMasking()
        self.overwrite_masking = OverwriteMasking()

        self.additive_attention = AdditiveAttention(
            input_dim=config.cnn_filter_num,
            query_vec_dim=config.attention_hidden_dim,
        )

    @property
    def output_dim(self) -> int:
        """Dimension of the news encoder output."""
        dim = self.config.cnn_filter_num
        if self.category_encoder is not None:
            dim += self.config.category_embedding_dim
        if self.subcategory_encoder is not None:
            dim += self.config.subcategory_embedding_dim
        return dim

    def _process_title(
        self, title_tokens: torch.Tensor, training: bool
    ) -> torch.Tensor:
        embedded = self.embedding_layer(title_tokens)  # (B, T, E)
        y = self.dropout1(embedded)

        # Conv1d: (B, E, T) -> (B, F, T) -> (B, T, F)
        y = y.transpose(1, 2)
        y = F.relu(self.cnn(y))
        y = y.transpose(1, 2)
        y = self.dropout2(y)

        # Masking
        mask = self.compute_masking(title_tokens)  # (B, T) float
        y = self.overwrite_masking(y, mask)

        padding_mask = title_tokens != 0  # (B, T) bool
        return self.additive_attention(y, mask=padding_mask)

    def forward(self, inputs: torch.Tensor, training: bool = True) -> torch.Tensor:
        """Encode news tokens (optionally concatenated with category/subcategory).

        Args:
            inputs: (batch, title_len) or (batch, title_len + 2) if categories present.

        Returns:
            (batch, output_dim) news representation.
        """
        input_len = inputs.shape[-1]
        has_cat = input_len > self.config.max_title_length

        if has_cat and (
            self.category_encoder is not None or self.subcategory_encoder is not None
        ):
            title_tokens = inputs[:, : self.config.max_title_length]
            category_id = inputs[
                :, self.config.max_title_length : self.config.max_title_length + 1
            ]
            subcategory_id = inputs[:, self.config.max_title_length + 1 :]

            title_vec = self._process_title(title_tokens, training)
            representations = [title_vec]

            if self.category_encoder is not None:
                representations.append(
                    self.category_encoder(category_id, training=training)
                )
            if self.subcategory_encoder is not None:
                representations.append(
                    self.subcategory_encoder(subcategory_id, training=training)
                )

            if len(representations) > 1:
                return torch.cat(representations, dim=-1)
            return title_vec
        else:
            if has_cat:
                title_tokens = inputs[:, : self.config.max_title_length]
            else:
                title_tokens = inputs
            return self._process_title(title_tokens, training)


class PLMNewsEncoder(nn.Module):
    """PLM-mode news encoder for LSTUR.

    Replaces the GloVe ``Embedding + Conv1d + AdditiveAttention`` title
    pipeline with a frozen token-level BERT lookup + the same Conv1d /
    AddAttn pipeline (delegated to :class:`PLMTokenCNNEncoder`).
    Concatenates optional category / subcategory embeddings the same
    way the GloVe :class:`NewsEncoder` does so the GRU input dim is
    unchanged.

    Input layout (packed, last dim):
        - column 0: parsed news id (used by PLM title lookup)
        - column 1: category id          (if ``use_category``)
        - column 2: subcategory id       (if ``use_subcategory``)
    """

    def __init__(
        self,
        config: LSTURConfig,
        title_encoder: PLMTokenCNNEncoder,
        category_encoder: CategoryEncoder | None = None,
        subcategory_encoder: SubcategoryEncoder | None = None,
    ):
        super().__init__()
        self.config = config
        self.title_encoder = title_encoder
        self.category_encoder = category_encoder
        self.subcategory_encoder = subcategory_encoder

    @property
    def output_dim(self) -> int:
        dim = self.config.cnn_filter_num
        if self.category_encoder is not None:
            dim += self.config.category_embedding_dim
        if self.subcategory_encoder is not None:
            dim += self.config.subcategory_embedding_dim
        return dim

    def forward(self, inputs: torch.Tensor, training: bool = True) -> torch.Tensor:
        # inputs: (..., k) where k = 1 + use_category + use_subcategory.
        news_idx = inputs[..., 0]
        title_vec = self.title_encoder(news_idx, training=training)

        reps = [title_vec]
        col = 1
        if self.category_encoder is not None:
            cat = inputs[..., col : col + 1]
            reps.append(self.category_encoder(cat, training=training))
            col += 1
        if self.subcategory_encoder is not None:
            sub = inputs[..., col : col + 1]
            reps.append(self.subcategory_encoder(sub, training=training))

        if len(reps) > 1:
            return torch.cat(reps, dim=-1)
        return title_vec


class UserEncoder(nn.Module):
    """GRU-based user encoder with long-term user embeddings.

    Supports two modes:
    - ``ini``: user embedding as GRU initial hidden state.
    - ``con``: concatenation of GRU output and user embedding, then linear.
    """

    def __init__(self, config: LSTURConfig, news_encoder: NewsEncoder, num_users: int):
        super().__init__()
        self.config = config
        self.news_encoder = news_encoder
        self.num_users = num_users

        # The GRU input size must match the news encoder output dimension
        gru_input_size = news_encoder.output_dim

        self.user_embedding = nn.Embedding(num_users, config.gru_unit, padding_idx=None)
        nn.init.zeros_(self.user_embedding.weight)

        # Bernoulli masking on user embeddings during training (paper §3.2)
        self.user_embedding_dropout = nn.Dropout(p=config.user_embedding_dropout_rate)

        self.gru = nn.GRU(
            input_size=gru_input_size,
            hidden_size=config.gru_unit,
            batch_first=True,
        )

        if config.type == "con":
            self.concat_dense = nn.Linear(config.gru_unit * 2, config.gru_unit)

    def forward(
        self,
        inputs: torch.Tensor | list[torch.Tensor],
        training: bool = True,
    ) -> torch.Tensor:
        """Encode user from history and user id.

        Args:
            inputs: list of [history_tokens, user_indices]
                - history_tokens: (batch, history_len, title_len)
                - user_indices: (batch,) or (batch, 1)

        Returns:
            (batch, gru_unit) user representation.
        """
        history_tokens, user_indices = inputs

        # Handle shape
        if user_indices.dim() == 1:
            user_indices = user_indices.unsqueeze(-1)

        long_u_emb = self.user_embedding(user_indices).squeeze(1)  # (B, gru_unit)
        # Bernoulli masking of long-term user repr during training (paper §3.2)
        long_u_emb = self.user_embedding_dropout(long_u_emb)

        # TimeDistributed news encoder
        B, H, T = history_tokens.shape
        flat = history_tokens.reshape(B * H, T)
        news_vecs = self.news_encoder(flat, training=training)
        click_title_presents = news_vecs.reshape(B, H, -1)  # (B, H, news_dim)

        if self.config.type == "ini":
            # Use user embedding as initial hidden state
            h0 = long_u_emb.unsqueeze(0)  # (1, B, gru_unit)
            output, _ = self.gru(click_title_presents, h0)
            user_present = output[:, -1, :]  # last hidden state
        elif self.config.type == "con":
            output, _ = self.gru(click_title_presents)
            short_uemb = output[:, -1, :]
            concat_emb = torch.cat([short_uemb, long_u_emb], dim=-1)
            user_present = self.concat_dense(concat_emb)
        else:
            raise ValueError(f"Invalid user encoder type: {self.config.type}")

        return user_present


# ------------------------------------------------------------------
# Full model
# ------------------------------------------------------------------


class LSTUR(BaseModel):
    """LSTUR: Long- and Short-term User Representations for news recommendation."""

    def __init__(
        self,
        processed_news: dict[str, Any],
        num_users: int,
        config: LSTURConfig | None = None,
        **config_overrides,
    ):
        super().__init__()
        if config is None:
            config = LSTURConfig(**config_overrides)
        self.config = config
        self.processed_news = processed_news
        self.num_users = num_users
        self.process_user_id = config.process_user_id

        encoder_type = getattr(getattr(config, "encoder", None), "type", "glove")
        self.encoder_type = encoder_type

        # Optional category encoders (shared across both encoder modes).
        category_encoder = None
        subcategory_encoder = None
        if config.use_category:
            category_encoder = CategoryEncoder(config, processed_news["num_categories"])
        if config.use_subcategory:
            subcategory_encoder = SubcategoryEncoder(
                config, processed_news["num_subcategories"]
            )

        if encoder_type == "glove":
            # GloVe path — shared word embedding + token-level CNN+attn.
            vocab_size = processed_news["vocab_size"]
            embeddings_matrix = processed_news["embeddings"]
            self.embedding_layer = nn.Embedding(vocab_size, config.embedding_size)
            self.embedding_layer.weight = nn.Parameter(
                torch.tensor(embeddings_matrix, dtype=torch.float32)
            )
            self.news_encoder = NewsEncoder(
                config,
                self.embedding_layer,
                category_encoder=category_encoder,
                subcategory_encoder=subcategory_encoder,
            )
        else:
            # PLM path — frozen token-level cache + Conv1d / AddAttn over
            # BERT tokens (NAML-style), then concat optional cat/subcat.
            if "plm_token_embeddings_by_id" not in processed_news:
                raise KeyError(
                    f"LSTUR with encoder.type='{encoder_type}' requires "
                    "processed_news['plm_token_embeddings_by_id']. Call "
                    "attach_plm_embeddings(..., level='token') in the runner."
                )
            plm_dim = int(processed_news["plm_dim"])
            title_encoder = PLMTokenCNNEncoder(
                processed_news["plm_token_embeddings_by_id"],
                processed_news["plm_attention_mask_by_id"],
                news_dim=config.cnn_filter_num,
                plm_dim=plm_dim,
                cnn_kernel_size=config.cnn_kernel_size,
                dropout_rate=config.dropout_rate,
                word_attention_query_dim=config.attention_hidden_dim,
                activation=config.cnn_activation,
            )
            self.news_encoder = PLMNewsEncoder(
                config,
                title_encoder,
                category_encoder=category_encoder,
                subcategory_encoder=subcategory_encoder,
            )

        self.user_encoder = UserEncoder(config, self.news_encoder, num_users)

    def forward(
        self,
        inputs: dict[str, torch.Tensor],
        training: bool = True,
    ) -> torch.Tensor:
        """Forward pass for training. Returns raw logits.

        Inference uses ``self.news_encoder`` and ``self.user_encoder``
        directly via the shared evaluator (see
        :mod:`src.core.models.evaluation`), not this method.
        """
        return self.score_training_batch(inputs, training)

    # ----- helpers --------------------------------------------------------

    def _maybe_concat_category(
        self, tokens: torch.Tensor, inputs: dict, cat_key: str, subcat_key: str
    ) -> torch.Tensor:
        """Optionally concatenate category/subcategory to token tensor."""
        parts = [tokens]
        if cat_key in inputs and subcat_key in inputs:
            cat = inputs[cat_key]
            subcat = inputs[subcat_key]
            if cat.dim() < tokens.dim():
                cat = cat.unsqueeze(-1)
            if subcat.dim() < tokens.dim():
                subcat = subcat.unsqueeze(-1)
            parts.extend([cat, subcat])
        if len(parts) > 1:
            return torch.cat(parts, dim=-1)
        return tokens

    def _pack_plm_features(
        self,
        news_ids: torch.Tensor,
        inputs: dict[str, torch.Tensor],
        cat_key: str,
        subcat_key: str,
    ) -> torch.Tensor:
        """Pack PLM-mode features [news_idx | cat | subcat] along last dim.

        ``news_ids`` is one integer per news slot (no token axis). The
        returned tensor matches the packed layout :class:`PLMNewsEncoder`
        consumes.
        """
        parts = [news_ids.unsqueeze(-1)]
        if self.config.use_category and cat_key in inputs:
            cat = inputs[cat_key]
            if cat.dim() == news_ids.dim():
                cat = cat.unsqueeze(-1)
            parts.append(cat)
        if self.config.use_subcategory and subcat_key in inputs:
            subcat = inputs[subcat_key]
            if subcat.dim() == news_ids.dim():
                subcat = subcat.unsqueeze(-1)
            parts.append(subcat)
        return torch.cat(parts, dim=-1)

    def score_training_batch(
        self, inputs: dict[str, torch.Tensor], training: bool
    ) -> torch.Tensor:
        if self.encoder_type == "glove":
            history_packed = self._maybe_concat_category(
                inputs["hist_tokens"], inputs, "hist_category", "hist_subcategory"
            )
            candidate_packed = self._maybe_concat_category(
                inputs["cand_tokens"], inputs, "cand_category", "cand_subcategory"
            )
        else:
            history_packed = self._pack_plm_features(
                inputs["hist_features"], inputs, "hist_category", "hist_subcategory"
            )
            candidate_packed = self._pack_plm_features(
                inputs["cand_features"], inputs, "cand_category", "cand_subcategory"
            )

        user_ids = inputs.get("user_ids", inputs.get("user_indices"))
        user_repr = self.user_encoder([history_packed, user_ids], training=training)

        B, C, F = candidate_packed.shape
        flat_cand = candidate_packed.reshape(B * C, F)
        cand_repr = self.news_encoder(flat_cand, training=training).reshape(B, C, -1)

        scores = torch.sum(cand_repr * user_repr.unsqueeze(1), dim=-1)
        return scores  # raw logits; loss applies log-softmax internally
