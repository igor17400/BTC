import logging

import numpy as np

# Setup logger
logger = logging.getLogger(__name__)


def auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute AUC using the trapezoidal rule.

    Args:
        y_true: Binary ground truth labels.
        y_pred: Predicted scores.

    Returns:
        AUC score, or 0.5 if only one class is present.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    n_pos = np.sum(y_true)
    n_neg = np.sum(1 - y_true)

    # If only one class present, return 0.5
    if n_pos == 0 or n_neg == 0:
        return 0.5

    # Sort by predictions in descending order
    sorted_indices = np.argsort(-y_pred)
    sorted_y_true = y_true[sorted_indices]

    # Calculate cumulative sums
    tps = np.cumsum(sorted_y_true)
    fps = np.cumsum(1 - sorted_y_true)

    # Add initial point (0, 0)
    tps = np.concatenate([[0], tps])
    fps = np.concatenate([[0], fps])

    # Calculate TPR and FPR
    tpr = tps / n_pos
    fpr = fps / n_neg

    # Calculate AUC using trapezoidal rule
    dx = fpr[1:] - fpr[:-1]
    y_avg = (tpr[1:] + tpr[:-1]) / 2
    return float(np.sum(dx * y_avg))


def mrr(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute Mean Reciprocal Rank.

    MRR = sum(relevance_i / rank_i) / sum(relevance_i)
    This follows the MIND dataset evaluation convention.

    Args:
        y_true: Binary ground truth labels.
        y_score: Predicted scores.

    Returns:
        MRR score.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_score = np.asarray(y_score, dtype=np.float64)

    # Sort indices in descending order of scores
    order = np.argsort(y_score)[::-1]
    y_true_sorted = y_true[order]

    # Calculate reciprocal ranks
    rr_scores = y_true_sorted / (np.arange(len(y_true_sorted)) + 1)

    total_relevant = np.sum(y_true)
    if total_relevant > 0:
        return float(np.sum(rr_scores) / total_relevant)
    return 0.0


def dcg_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    """Compute Discounted Cumulative Gain at k.

    DCG@k = sum_{i=1}^k (2^rel_i - 1) / log2(i + 1)

    Args:
        y_true: Binary ground truth labels.
        y_score: Predicted scores.
        k: Number of top positions to consider.

    Returns:
        DCG@k score.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_score = np.asarray(y_score, dtype=np.float64)

    # Sort by score in descending order
    order = np.argsort(y_score)[::-1]
    y_true_sorted = y_true[order[:k]]

    # Calculate gains: 2^relevance - 1
    gains = np.power(2, y_true_sorted) - 1

    # Calculate discounts: log2(rank + 1) where rank starts from 1
    discounts = np.log2(np.arange(len(y_true_sorted)) + 2)

    return float(np.sum(gains / discounts))


def ndcg_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    """Compute Normalized Discounted Cumulative Gain at k.

    Args:
        y_true: Binary ground truth labels.
        y_score: Predicted scores.
        k: Number of top positions to consider.

    Returns:
        NDCG@k score.
    """
    actual_dcg = dcg_at_k(y_true, y_score, k)
    ideal_dcg = dcg_at_k(y_true, y_true, k)  # Best possible DCG

    if ideal_dcg > 0:
        return actual_dcg / ideal_dcg
    return 0.0


class NewsRecommenderMetrics:
    """NumPy-based metrics for evaluating news recommendation models.

    Computes AUC, MRR, NDCG@5, and NDCG@10 using pure NumPy operations.
    """

    METRIC_NAMES = ["auc", "mrr", "ndcg@5", "ndcg@10"]

    def __init__(self):
        """Initialize the metrics calculator."""
        pass

    def compute_metrics(self, y_true, y_pred_logits, progress=None):
        """Compute metrics for a single impression or batch.

        Args:
            y_true: Array of shape (batch_size, num_candidates) or (num_candidates,).
            y_pred_logits: Array of shape (batch_size, num_candidates) or (num_candidates,).
            progress: Optional progress bar for logging (unused, kept for API compat).

        Returns:
            dict: Dictionary of metric names and their values.
        """
        y_true = np.asarray(y_true)
        y_pred_logits = np.asarray(y_pred_logits)

        if y_true.ndim == 1:
            # Single impression
            return self._compute_metrics_single(y_true, y_pred_logits)

        # Batch processing
        batch_size = y_true.shape[0]
        auc_scores = np.empty(batch_size)
        mrr_scores = np.empty(batch_size)
        ndcg5_scores = np.empty(batch_size)
        ndcg10_scores = np.empty(batch_size)

        for i in range(batch_size):
            auc_scores[i] = auc(y_true[i], y_pred_logits[i])
            mrr_scores[i] = mrr(y_true[i], y_pred_logits[i])
            ndcg5_scores[i] = ndcg_at_k(y_true[i], y_pred_logits[i], k=5)
            ndcg10_scores[i] = ndcg_at_k(y_true[i], y_pred_logits[i], k=10)

        return {
            "auc": float(np.mean(auc_scores)),
            "mrr": float(np.mean(mrr_scores)),
            "ndcg@5": float(np.mean(ndcg5_scores)),
            "ndcg@10": float(np.mean(ndcg10_scores)),
        }

    def compute_metrics_from_scores(
        self, y_true_grouped, y_pred_scores_grouped, progress=None
    ):
        """Compute metrics from grouped scores (for compatibility with existing code).

        Args:
            y_true_grouped: Array of shape (batch_size, num_candidates).
            y_pred_scores_grouped: Array of shape (batch_size, num_candidates).
            progress: Optional progress bar for logging.

        Returns:
            dict: Dictionary of metric names and their values.
        """
        return self.compute_metrics(y_true_grouped, y_pred_scores_grouped, progress)

    def _compute_metrics_single(self, y_true, y_pred_logits):
        """Compute all metrics for a single impression.

        Args:
            y_true: 1D array of binary ground truth labels.
            y_pred_logits: 1D array of predicted scores.

        Returns:
            dict: Dictionary of metric names and their values.
        """
        return {
            "auc": auc(y_true, y_pred_logits),
            "mrr": mrr(y_true, y_pred_logits),
            "ndcg@5": ndcg_at_k(y_true, y_pred_logits, k=5),
            "ndcg@10": ndcg_at_k(y_true, y_pred_logits, k=10),
        }
