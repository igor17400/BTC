"""LSTUR (Long- and Short-term User Representations) -- PyTorch."""

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.core.models.configs import LSTURConfig

from ..layers import AdditiveAttention, ComputeMasking, OverwriteMasking
from .base import BaseModel

# ------------------------------------------------------------------
# Sub-encoders
# ------------------------------------------------------------------


class LSTURCategoryEncoder(nn.Module):
    """Simple embedding for category IDs (LSTUR-style, no projection)."""

    def __init__(self, config: LSTURConfig, num_categories: int):
        super().__init__()
        self.embedding = nn.Embedding(num_categories + 1, config.category_embedding_dim)

    def forward(self, inputs: torch.Tensor, training: bool = True) -> torch.Tensor:
        embedded = self.embedding(inputs)  # (B, 1, dim)
        return embedded.squeeze(1)  # (B, dim)


class LSTURSubcategoryEncoder(nn.Module):
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
        category_encoder: LSTURCategoryEncoder | None = None,
        subcategory_encoder: LSTURSubcategoryEncoder | None = None,
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
        **kwargs,
    ):
        super().__init__()
        if config is None:
            config = LSTURConfig(**kwargs)
        self.config = config
        self.processed_news = processed_news
        self.num_users = num_users
        self.process_user_id = config.process_user_id

        # Shared word embedding
        vocab_size = processed_news["vocab_size"]
        embeddings_matrix = processed_news["embeddings"]
        self.embedding_layer = nn.Embedding(vocab_size, config.embedding_size)
        self.embedding_layer.weight = nn.Parameter(
            torch.tensor(embeddings_matrix, dtype=torch.float32)
        )

        # Optional category encoders
        category_encoder = None
        subcategory_encoder = None
        if config.use_category:
            category_encoder = LSTURCategoryEncoder(
                config, processed_news["num_categories"]
            )
        if config.use_subcategory:
            subcategory_encoder = LSTURSubcategoryEncoder(
                config, processed_news["num_subcategories"]
            )

        # Encoders
        self.news_encoder = NewsEncoder(
            config,
            self.embedding_layer,
            category_encoder=category_encoder,
            subcategory_encoder=subcategory_encoder,
        )
        self.user_encoder = UserEncoder(config, self.news_encoder, num_users)

    def forward(
        self,
        inputs: dict[str, torch.Tensor],
        training: bool = True,
    ) -> torch.Tensor:
        if training:
            return self._score_training(inputs, training)
        elif "single_candidate_tokens" in inputs:
            return self._score_single(inputs)
        else:
            return self._score_multi(inputs)

    # ----- helpers --------------------------------------------------------

    def _maybe_concat_cat(
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

    def _score_training(
        self, inputs: dict[str, torch.Tensor], training: bool
    ) -> torch.Tensor:
        history_tokens = self._maybe_concat_cat(
            inputs["hist_tokens"], inputs, "hist_category", "hist_subcategory"
        )
        candidate_tokens = self._maybe_concat_cat(
            inputs["cand_tokens"], inputs, "cand_category", "cand_subcategory"
        )
        user_ids = inputs.get("user_ids", inputs.get("user_indices"))

        user_repr = self.user_encoder([history_tokens, user_ids], training=training)

        B, C, T = candidate_tokens.shape
        flat_cand = candidate_tokens.reshape(B * C, T)
        cand_repr = self.news_encoder(flat_cand, training=training).reshape(B, C, -1)

        scores = torch.sum(cand_repr * user_repr.unsqueeze(1), dim=-1)
        return torch.softmax(scores, dim=-1)

    def _score_single(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        history_tokens = self._maybe_concat_cat(
            inputs["hist_tokens"], inputs, "hist_category", "hist_subcategory"
        )
        candidate_tokens = inputs["single_candidate_tokens"]
        user_ids = inputs.get("user_ids", inputs.get("user_indices"))

        user_repr = self.user_encoder([history_tokens, user_ids], training=False)
        cand_repr = self.news_encoder(candidate_tokens, training=False)

        score = torch.sum(cand_repr * user_repr, dim=-1, keepdim=True)
        return torch.sigmoid(score)

    def _score_multi(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        history_tokens = self._maybe_concat_cat(
            inputs["hist_tokens"], inputs, "hist_category", "hist_subcategory"
        )
        candidate_tokens = self._maybe_concat_cat(
            inputs["cand_tokens"], inputs, "cand_category", "cand_subcategory"
        )
        user_ids = inputs.get("user_ids", inputs.get("user_indices"))

        user_repr = self.user_encoder([history_tokens, user_ids], training=False)

        B, C, T = candidate_tokens.shape
        flat_cand = candidate_tokens.reshape(B * C, T)
        cand_repr = self.news_encoder(flat_cand, training=False).reshape(B, C, -1)

        scores = torch.sum(cand_repr * user_repr.unsqueeze(1), dim=-1)
        return torch.sigmoid(scores)
