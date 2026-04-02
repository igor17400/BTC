from typing import Any
import numpy as np
import keras
from keras import ops
from rich.progress import Progress

from src.core.io.saving import save_predictions_to_file_fn
from src.frameworks.keras.dataloaders import NewsBatchDataloader, UserHistoryBatchDataloader


class BaseModel(keras.Model):
    """Base class for news recommendation models.

    This class provides common functionality for news recommendation models,
    including methods for fast evaluation using precomputed vectors.
    """

    def __init__(self, name: str = "base_news_recommender"):
        super().__init__(name=name)

        # Attributes to be set by subclasses
        self.news_encoder: keras.Model | None = None
        self.user_encoder: keras.Model | None = None
        self.process_user_id: bool = False
        self.float_dtype: str = "float32"  # Will be set by training config

    def _validate_processed_news(self) -> None:
        """Validate processed news data integrity."""
        required_keys = ["vocab_size", "embeddings", "num_categories", "num_subcategories"]
        for key in required_keys:
            if key not in self.processed_news:
                raise ValueError(f"Missing required key '{key}' in processed_news")

        embeddings_matrix = self.processed_news["embeddings"]
        if np.isnan(embeddings_matrix).any():
            raise ValueError("Embeddings matrix contains NaN values")

        if embeddings_matrix.shape[1] != self.config.embedding_size:
            raise ValueError(
                f"Embeddings dimension {embeddings_matrix.shape[1]} doesn't match "
                f"configured embedding_size {self.config.embedding_size}"
            )

    def precompute_news_vectors(
            self,
            news_dataloader: NewsBatchDataloader,
            progress: Progress,
    ) -> dict[str, np.ndarray]:
        """Precompute vectors for all news articles.

        Args:
            news_dataloader: NewsBatchDataloader instance for news articles
            progress: Progress bar manager

        Returns:
            Dictionary mapping news IDs to their vectors
        """
        news_vecs_dict = {}
        total_news = len(news_dataloader)

        news_progress = progress.add_task(
            "Computing news vectors...", total=total_news, visible=True
        )

        for batch in news_dataloader:
            # Convert batch to numpy arrays
            news_ids = batch["news_id"]
            news_features = batch["news_features"]

            if self.news_encoder is None:
                raise RuntimeError("News Encoder not initialized. Ensure the model is properly built.")
            batch_vecs = ops.convert_to_numpy(
                self.news_encoder(news_features, training=False)
            )

            # Debug: Check for NaN/Inf in news vectors
            if np.isnan(batch_vecs).any() or np.isinf(batch_vecs).any():
                print(f"DEBUG: batch_vecs has NaN/Inf: {np.isnan(batch_vecs).any()}/{np.isinf(batch_vecs).any()}")
                print(f"DEBUG: batch_vecs min/max: {np.min(batch_vecs):.6f} / {np.max(batch_vecs):.6f}")

            for i, news_id in enumerate(news_ids):
                news_vecs_dict[ops.convert_to_numpy(news_id).item()] = batch_vecs[i]

            progress.update(news_progress, advance=len(news_ids))

        progress.remove_task(news_progress)
        return news_vecs_dict

    def precompute_user_vectors(
            self, user_dataloader: UserHistoryBatchDataloader, progress: Progress
    ) -> dict[int, np.ndarray]:
        """Pre-compute user vectors for fast evaluation.

        Args:
            user_dataloader: UserHistoryBatchDataloader for user history data
            progress: Optional progress bar

        Returns:
            Dictionary mapping impression IDs based on the users behaviors to their vectors
        """
        user_vecs_dict = {}
        user_progress = progress.add_task(
            "Computing user vectors...", total=len(user_dataloader), visible=True
        )

        if self.user_encoder is None:
            raise RuntimeError("User Encoder not initialized. Ensure the model is properly built.")

        for impression_ids, user_ids, features in user_dataloader:
            # Get user representation from history
            if self.process_user_id:  # Used for LSTUR model
                user_vec = self.user_encoder([features, user_ids], training=False)
            else:
                user_vec = self.user_encoder(features, training=False)

            # Debug: Check for NaN/Inf in user vectors
            user_vec_np = ops.convert_to_numpy(user_vec)
            if np.isnan(user_vec_np).any() or np.isinf(user_vec_np).any():
                print(f"DEBUG: user_vec has NaN/Inf: {np.isnan(user_vec_np).any()}/{np.isinf(user_vec_np).any()}")
                print(f"DEBUG: user_vec min/max: {np.min(user_vec_np):.6f} / {np.max(user_vec_np):.6f}")

            # Store user vector for each impression in the batch
            for i, imp_id in enumerate(impression_ids):
                user_vecs_dict[int(imp_id)] = user_vec_np[i]

            progress.update(user_progress, advance=len(impression_ids))

        progress.remove_task(user_progress)

        return user_vecs_dict

    def fast_evaluate(
            self,
            user_hist_dataloader,
            news_dataloader,
            impression_iterator,
            metrics_calculator,
            progress: Progress,
            mode="validate",
            save_predictions_path=None,
            epoch=None,
            int_to_news_id_map=None,
    ) -> dict[str, float]:
        """Fast evaluation of the model using precomputed vectors and dataloader iterators."""
        # 1. Precompute news vectors
        news_vecs_dict = self.precompute_news_vectors(news_dataloader, progress)

        # 2. Precompute user vectors
        user_vecs_dict = self.precompute_user_vectors(user_hist_dataloader, progress)

        # 3. Score impressions using precomputed user and news vectors
        group_labels_list: list[np.ndarray] = []
        group_preds_list: list[np.ndarray] = []
        predictions_to_save = {}

        # Create progress bar for impressions
        impression_progress = progress.add_task(
            "Processing impressions...", total=len(impression_iterator), visible=True
        )

        for impression in impression_iterator:
            _, labels, impression_id, cand_ids = impression

            # Get user vector for this impression
            user_vector = user_vecs_dict[impression_id]

            # Get news vectors for candidate news
            cand_ids_np = ops.convert_to_numpy(cand_ids)  # Get numpy array of candidate IDs
            news_vectors = []
            for nid in cand_ids_np:
                # Convert integer ID to string news ID
                if int_to_news_id_map and nid in int_to_news_id_map:
                    # Use the mapping if provided (for datasets with custom IDs)
                    news_key = int_to_news_id_map[nid]
                else:
                    # Default behavior for MIND dataset (add "N" prefix)
                    news_key = f"N{str(nid)}"

                vec = news_vecs_dict.get(news_key)
                if vec is not None:
                    news_vectors.append(vec)

            # Calculate scores using dot product
            if not news_vectors:
                scores = np.array([])
            else:
                news_vectors_array = np.stack(news_vectors, axis=0)

                # Debug: Check for NaN/Inf in user_vector and news_vectors
                if np.isnan(user_vector).any() or np.isinf(user_vector).any():
                    print(
                        f"DEBUG: user_vector has NaN/Inf: {np.isnan(user_vector).any()}/{np.isinf(user_vector).any()}")
                    print(f"DEBUG: user_vector min/max: {np.min(user_vector):.6f} / {np.max(user_vector):.6f}")

                if np.isnan(news_vectors_array).any() or np.isinf(news_vectors_array).any():
                    print(
                        f"DEBUG: news_vectors has NaN/Inf: {np.isnan(news_vectors_array).any()}/{np.isinf(news_vectors_array).any()}")
                    print(
                        f"DEBUG: news_vectors min/max: {np.min(news_vectors_array):.6f} / {np.max(news_vectors_array):.6f}")

                scores = np.dot(news_vectors_array, user_vector)

                # Debug: Check final scores
                if np.isnan(scores).any() or np.isinf(scores).any():
                    print(f"DEBUG: Final scores has NaN/Inf: {np.isnan(scores).any()}/{np.isinf(scores).any()}")
                    print(f"DEBUG: Final scores min/max: {np.min(scores):.6f} / {np.max(scores):.6f}")

            group_labels_list.append(ops.convert_to_numpy(labels))
            group_preds_list.append(scores)

            if save_predictions_path:
                predictions_to_save[str(ops.convert_to_numpy(cand_ids))] = (
                    ops.convert_to_numpy(labels).tolist(),
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
            save_predictions_to_file_fn(predictions_to_save, save_predictions_path, epoch, mode)

        return final_metrics

    def _compute_metrics(
            self,
            group_labels_list: list[np.ndarray],
            group_preds_list: list[np.ndarray],
            metrics_calculator: Any,
            progress: Progress,
    ) -> dict[str, float]:
        """Computes and aggregates metrics from lists of labels and predictions.

        This method iterates through each impression's labels and predicted scores.
        For each impression, it calculates the loss and various ranking metrics
        (e.g., AUC, MRR, nDCG) using the provided `metrics_calculator`. It then
        aggregates these scores to compute the final average metrics over all
        impressions.

        Args:
            group_labels_list (List[np.ndarray]): A list of 1D NumPy arrays, where
                each array contains the ground truth labels (0 or 1) for one impression.
            group_preds_list (List[np.ndarray]): A list of 1D NumPy arrays, where
                each array contains the predicted scores (logits) for one impression.
            metrics_calculator (Any): An object equipped with a `compute_metrics`
                method that calculates metrics from labels and logits. It should
                also have a `METRIC_NAMES` attribute.
            progress (Progress): The progress bar instance, used here to log
                warnings if invalid values (NaN/Inf) are found in predictions.

        Returns:
            Dict[str, float]: A dictionary containing the final computed metrics,
                including the average loss and other metrics like AUC, MRR, etc.
        """
        val_loss_total = 0.0
        num_valid_impressions_for_loss = 0
        metric_values_agg = {key: [] for key in metrics_calculator.METRIC_NAMES}

        for labels_np, scores_np in zip(group_labels_list, group_preds_list):
            if labels_np.size == 0 or scores_np.size == 0:
                continue

            # Check for NaN/Inf in numpy arrays before any computation
            if np.isnan(scores_np).any() or np.isinf(scores_np).any():
                progress.console.print(
                    "[WARNING fast_evaluate] NaN/Inf detected in scores. Skipping impression."
                )
                continue

            # For loss calculation, convert to tensors
            labels_tensor = ops.convert_to_tensor([labels_np], dtype=self.float_dtype)
            scores_logits_tensor = ops.convert_to_tensor([scores_np], dtype=self.float_dtype)
            scores_probs_tensor = ops.softmax(scores_logits_tensor, axis=-1)

            try:
                # Use compiled loss function for consistency with training
                if hasattr(self, 'compiled_loss') and self.compiled_loss is not None:
                    loss = self.compiled_loss(labels_tensor, scores_probs_tensor)
                else:
                    # Fallback to categorical crossentropy if no compiled loss
                    loss = ops.categorical_crossentropy(labels_tensor, scores_probs_tensor, from_logits=False)
                    loss = ops.mean(loss)

                if loss is not None:
                    val_loss_total += ops.convert_to_numpy(loss)
                    num_valid_impressions_for_loss += 1
            except Exception as e:
                progress.console.print(f"[WARNING fast_evaluate] Error calculating loss: {e}.")

            impression_metrics = metrics_calculator.compute_metrics(
                y_true=labels_np, y_pred_logits=scores_np
            )
            for metric_name, value in impression_metrics.items():
                if metric_name in metric_values_agg:
                    metric_values_agg[metric_name].append(float(value))

        final_metrics = {
            "loss": (
                (val_loss_total / num_valid_impressions_for_loss)
                if num_valid_impressions_for_loss > 0
                else 0.0
            )
        }
        for metric_name, values_list in metric_values_agg.items():
            final_metrics[metric_name] = np.mean(values_list) if values_list else 0.0

        return final_metrics
