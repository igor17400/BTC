"""Base model class for PyTorch news recommendation models.

Ported from src/models/base.py (Keras version).
"""

from typing import Any

import numpy as np
import torch
import torch.nn as nn
from rich.progress import Progress

from src.core.io.saving import save_predictions_to_file_fn


class BaseModel(nn.Module):
    """Base class for PyTorch news recommendation models.

    Provides precompute / fast-evaluate helpers that mirror the Keras
    ``BaseModel`` interface so that the evaluation pipeline can stay
    framework-agnostic.
    """

    def __init__(self):
        super().__init__()

        # Subclasses must set these
        self.news_encoder: nn.Module | None = None
        self.user_encoder: nn.Module | None = None
        self.process_user_id: bool = False
        self.float_dtype: str = "float32"

    # ------------------------------------------------------------------
    # Precompute helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def precompute_news_vectors(
        self,
        news_dataloader,
        progress: Progress,
    ) -> dict[str, np.ndarray]:
        """Precompute vectors for all news articles.

        Args:
            news_dataloader: Iterable yielding ``{"news_id": ..., "news_features": ...}``
            progress: Rich progress bar manager.

        Returns:
            Dict mapping news ID strings to numpy vectors.
        """
        self.eval()
        news_vecs_dict: dict[str, np.ndarray] = {}
        total_news = len(news_dataloader)

        news_progress = progress.add_task(
            "Computing news vectors...", total=total_news, visible=True
        )

        for batch in news_dataloader:
            news_ids = batch["news_id"]
            news_features = batch["news_features"]

            if self.news_encoder is None:
                raise RuntimeError(
                    "News Encoder not initialised. Ensure the model is properly built."
                )

            batch_vecs = self.news_encoder(news_features, training=False)
            batch_vecs_np = batch_vecs.cpu().numpy()

            for i, news_id in enumerate(news_ids):
                if isinstance(news_id, (torch.Tensor, np.generic)):
                    news_id = news_id.item() if hasattr(news_id, "item") else int(news_id)
                news_vecs_dict[news_id] = batch_vecs_np[i]

            progress.update(news_progress, advance=len(news_ids))

        progress.remove_task(news_progress)
        return news_vecs_dict

    @torch.no_grad()
    def precompute_user_vectors(
        self,
        user_dataloader,
        progress: Progress,
    ) -> dict[int, np.ndarray]:
        """Precompute vectors for all users.

        Args:
            user_dataloader: Iterable yielding
                ``(impression_ids, user_ids, features)`` triples.
            progress: Rich progress bar manager.

        Returns:
            Dict mapping impression IDs to numpy user vectors.
        """
        self.eval()
        user_vecs_dict: dict[int, np.ndarray] = {}
        user_progress = progress.add_task(
            "Computing user vectors...", total=len(user_dataloader), visible=True
        )

        if self.user_encoder is None:
            raise RuntimeError(
                "User Encoder not initialised. Ensure the model is properly built."
            )

        for impression_ids, user_ids, features in user_dataloader:
            if self.process_user_id:
                user_vec = self.user_encoder([features, user_ids], training=False)
            else:
                user_vec = self.user_encoder(features, training=False)

            user_vec_np = user_vec.cpu().numpy()

            for i, imp_id in enumerate(impression_ids):
                user_vecs_dict[int(imp_id)] = user_vec_np[i]

            progress.update(user_progress, advance=len(impression_ids))

        progress.remove_task(user_progress)
        return user_vecs_dict

    # ------------------------------------------------------------------
    # Fast evaluation (same logic as Keras base)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def fast_evaluate(
        self,
        user_hist_dataloader,
        news_dataloader,
        impression_iterator,
        metrics_calculator,
        progress: Progress,
        mode: str = "validate",
        save_predictions_path: str | None = None,
        epoch: int | None = None,
        int_to_news_id_map: dict | None = None,
    ) -> dict[str, float]:
        """Fast evaluation using precomputed news & user vectors."""
        self.eval()

        # 1. Precompute news vectors
        news_vecs_dict = self.precompute_news_vectors(news_dataloader, progress)

        # 2. Precompute user vectors
        user_vecs_dict = self.precompute_user_vectors(user_hist_dataloader, progress)

        # 3. Score impressions
        group_labels_list: list[np.ndarray] = []
        group_preds_list: list[np.ndarray] = []
        predictions_to_save: dict = {}

        impression_progress = progress.add_task(
            "Processing impressions...", total=len(impression_iterator), visible=True
        )

        for impression in impression_iterator:
            _, labels, impression_id, cand_ids = impression

            user_vector = user_vecs_dict[impression_id]

            # Resolve candidate IDs to numpy
            if isinstance(cand_ids, torch.Tensor):
                cand_ids_np = cand_ids.cpu().numpy()
            else:
                cand_ids_np = np.asarray(cand_ids)

            news_vectors = []
            for nid in cand_ids_np:
                if int_to_news_id_map and nid in int_to_news_id_map:
                    news_key = int_to_news_id_map[nid]
                else:
                    news_key = f"N{str(nid)}"

                vec = news_vecs_dict.get(news_key)
                if vec is not None:
                    news_vectors.append(vec)

            if not news_vectors:
                scores = np.array([])
            else:
                news_vectors_array = np.stack(news_vectors, axis=0)
                scores = np.dot(news_vectors_array, user_vector)

            if isinstance(labels, torch.Tensor):
                labels_np = labels.cpu().numpy()
            else:
                labels_np = np.asarray(labels)

            group_labels_list.append(labels_np)
            group_preds_list.append(scores)

            if save_predictions_path:
                predictions_to_save[str(cand_ids_np)] = (
                    labels_np.tolist(),
                    scores.tolist(),
                )

            progress.update(impression_progress, advance=1)

        progress.remove_task(impression_progress)

        # 4. Compute metrics
        final_metrics = self._compute_metrics(
            group_labels_list, group_preds_list, metrics_calculator, progress
        )
        final_metrics["num_impressions"] = len(group_labels_list)

        if save_predictions_path:
            save_predictions_to_file_fn(
                predictions_to_save, save_predictions_path, epoch, mode
            )

        return final_metrics

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_metrics(
        group_labels_list: list[np.ndarray],
        group_preds_list: list[np.ndarray],
        metrics_calculator: Any,
        progress: Progress,
    ) -> dict[str, float]:
        """Compute and aggregate metrics from labels and predictions.

        Uses the *core* metrics engine (numpy-based) so that results are
        comparable regardless of framework.
        """
        val_loss_total = 0.0
        num_valid = 0
        metric_values_agg = {k: [] for k in metrics_calculator.METRIC_NAMES}

        for labels_np, scores_np in zip(group_labels_list, group_preds_list):
            if labels_np.size == 0 or scores_np.size == 0:
                continue

            if np.isnan(scores_np).any() or np.isinf(scores_np).any():
                progress.console.print(
                    "[WARNING fast_evaluate] NaN/Inf detected in scores. Skipping impression."
                )
                continue

            # Simple softmax + cross-entropy for validation loss
            logits = scores_np - scores_np.max()
            exp_logits = np.exp(logits)
            probs = exp_logits / (exp_logits.sum() + 1e-7)
            loss = -np.sum(labels_np * np.log(probs + 1e-7))
            val_loss_total += loss
            num_valid += 1

            impression_metrics = metrics_calculator.compute_metrics(
                y_true=labels_np, y_pred_logits=scores_np
            )
            for metric_name, value in impression_metrics.items():
                if metric_name in metric_values_agg:
                    metric_values_agg[metric_name].append(float(value))

        final_metrics: dict[str, float] = {
            "loss": (val_loss_total / num_valid) if num_valid > 0 else 0.0
        }
        for metric_name, values_list in metric_values_agg.items():
            final_metrics[metric_name] = float(np.mean(values_list)) if values_list else 0.0

        return final_metrics
