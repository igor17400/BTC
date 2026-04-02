"""Base model class for Flax NNX news recommendation models."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from rich.progress import Progress


class BaseModel(nnx.Module):
    """Base class for Flax NNX news recommendation models.

    Subclasses must set the following attributes during ``__init__``:
    * ``news_encoder`` -- an ``nnx.Module`` that maps news features to vectors.
    * ``user_encoder`` -- an ``nnx.Module`` that maps user history to vectors.
    * ``process_user_id`` -- whether the user encoder requires explicit user
      IDs (e.g., LSTUR).
    """

    # These will be assigned by concrete subclasses.
    news_encoder: Optional[nnx.Module]
    user_encoder: Optional[nnx.Module]
    process_user_id: bool

    # ------------------------------------------------------------------
    # Pre-computation helpers
    # ------------------------------------------------------------------

    def precompute_news_vectors(
        self,
        news_dataloader,
        progress: Progress,
    ) -> Dict[str, np.ndarray]:
        """Precompute vectors for all news articles.

        Args:
            news_dataloader: Iterable yielding batches of
                ``{"news_id": ..., "news_features": ...}``.
            progress: Rich progress bar manager.

        Returns:
            Dictionary mapping news IDs (strings) to numpy vectors.
        """
        if self.news_encoder is None:
            raise RuntimeError("news_encoder is not initialised.")

        news_vecs: Dict[str, np.ndarray] = {}
        total_news = len(news_dataloader)

        task = progress.add_task(
            "Computing news vectors...", total=total_news, visible=True
        )

        for batch in news_dataloader:
            news_ids = batch["news_id"]
            news_features = batch["news_features"]

            # Ensure features are JAX arrays
            if not isinstance(news_features, jnp.ndarray):
                news_features = jnp.asarray(news_features)

            batch_vecs = self.news_encoder(news_features, training=False)
            batch_vecs_np = np.asarray(batch_vecs)

            if np.isnan(batch_vecs_np).any() or np.isinf(batch_vecs_np).any():
                print(
                    f"WARNING: news batch has NaN/Inf -- "
                    f"min={np.min(batch_vecs_np):.6f} max={np.max(batch_vecs_np):.6f}"
                )

            for i, nid in enumerate(news_ids):
                key = np.asarray(nid).item() if hasattr(nid, "item") else nid
                news_vecs[key] = batch_vecs_np[i]

            progress.update(task, advance=len(news_ids))

        progress.remove_task(task)
        return news_vecs

    def precompute_user_vectors(
        self,
        user_dataloader,
        progress: Progress,
    ) -> Dict[int, np.ndarray]:
        """Precompute user vectors for fast evaluation.

        Args:
            user_dataloader: Iterable yielding
                ``(impression_ids, user_ids_or_None, features)``.
            progress: Rich progress bar manager.

        Returns:
            Dictionary mapping impression IDs to numpy vectors.
        """
        if self.user_encoder is None:
            raise RuntimeError("user_encoder is not initialised.")

        user_vecs: Dict[int, np.ndarray] = {}
        task = progress.add_task(
            "Computing user vectors...", total=len(user_dataloader), visible=True
        )

        for impression_ids, user_ids, features in user_dataloader:
            if not isinstance(features, jnp.ndarray):
                features = jnp.asarray(features)

            if self.process_user_id:
                if not isinstance(user_ids, jnp.ndarray):
                    user_ids = jnp.asarray(user_ids)
                user_vec = self.user_encoder(features, user_ids, training=False)
            else:
                user_vec = self.user_encoder(features, training=False)

            user_vec_np = np.asarray(user_vec)

            if np.isnan(user_vec_np).any() or np.isinf(user_vec_np).any():
                print(
                    f"WARNING: user batch has NaN/Inf -- "
                    f"min={np.min(user_vec_np):.6f} max={np.max(user_vec_np):.6f}"
                )

            for i, imp_id in enumerate(impression_ids):
                user_vecs[int(imp_id)] = user_vec_np[i]

            progress.update(task, advance=len(impression_ids))

        progress.remove_task(task)
        return user_vecs

    # ------------------------------------------------------------------
    # Fast evaluation
    # ------------------------------------------------------------------

    def fast_evaluate(
        self,
        user_hist_dataloader,
        news_dataloader,
        impression_iterator,
        metrics_calculator,
        progress: Progress,
        mode: str = "validate",
        save_predictions_path=None,
        epoch: Optional[int] = None,
        int_to_news_id_map: Optional[Dict[int, str]] = None,
    ) -> Dict[str, float]:
        """Evaluate using precomputed news and user vectors.

        The flow mirrors ``BaseModel.fast_evaluate`` from the Keras version:
        1. Precompute news vectors.
        2. Precompute user vectors.
        3. Score every impression via dot product.
        4. Aggregate metrics.

        Returns:
            Dictionary of metric name to metric value.
        """
        news_vecs = self.precompute_news_vectors(news_dataloader, progress)
        user_vecs = self.precompute_user_vectors(user_hist_dataloader, progress)

        group_labels: List[np.ndarray] = []
        group_preds: List[np.ndarray] = []

        imp_task = progress.add_task(
            "Processing impressions...",
            total=len(impression_iterator),
            visible=True,
        )

        for impression in impression_iterator:
            _, labels, impression_id, cand_ids = impression

            user_vector = user_vecs[impression_id]
            cand_ids_np = np.asarray(cand_ids)

            news_vectors = []
            for nid in cand_ids_np:
                if int_to_news_id_map and nid in int_to_news_id_map:
                    news_key = int_to_news_id_map[nid]
                else:
                    news_key = f"N{str(nid)}"
                vec = news_vecs.get(news_key)
                if vec is not None:
                    news_vectors.append(vec)

            if not news_vectors:
                scores = np.array([])
            else:
                news_mat = np.stack(news_vectors, axis=0)
                scores = np.dot(news_mat, user_vector)

            group_labels.append(np.asarray(labels))
            group_preds.append(scores)
            progress.update(imp_task, advance=1)

        progress.remove_task(imp_task)

        # Compute metrics
        return self._compute_metrics(
            group_labels, group_preds, metrics_calculator, progress
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_metrics(
        group_labels: List[np.ndarray],
        group_preds: List[np.ndarray],
        metrics_calculator: Any,
        progress: Progress,
    ) -> Dict[str, float]:
        """Aggregate per-impression metrics."""
        val_loss_total = 0.0
        num_valid = 0
        metric_agg: Dict[str, list] = {k: [] for k in metrics_calculator.METRIC_NAMES}

        for labels_np, scores_np in zip(group_labels, group_preds):
            if labels_np.size == 0 or scores_np.size == 0:
                continue
            if np.isnan(scores_np).any() or np.isinf(scores_np).any():
                continue

            # Softmax-based loss (mirrors Keras version)
            scores_jnp = jnp.asarray(scores_np[None, :])
            labels_jnp = jnp.asarray(labels_np[None, :])
            probs = jax.nn.softmax(scores_jnp, axis=-1)
            loss = -jnp.sum(labels_jnp * jnp.log(probs + 1e-7), axis=-1)
            val_loss_total += float(loss.mean())
            num_valid += 1

            imp_metrics = metrics_calculator.compute_metrics(
                y_true=labels_np, y_pred_logits=scores_np
            )
            for name, value in imp_metrics.items():
                if name in metric_agg:
                    metric_agg[name].append(float(value))

        final: Dict[str, float] = {
            "loss": val_loss_total / num_valid if num_valid > 0 else 0.0,
        }
        for name, vals in metric_agg.items():
            final[name] = float(np.mean(vals)) if vals else 0.0
        final["num_impressions"] = len(group_labels)
        return final
