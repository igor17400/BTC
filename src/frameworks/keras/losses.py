import keras

from typing import Any


class NewsRecommenderLoss(keras.losses.Loss):
    """Base class for news recommendation losses."""

    def __init__(
            self,
            name: str = "news_recommender_loss",
            reduction: str = "sum_over_batch_size",
            **kwargs: Any,
    ) -> None:
        super().__init__(name=name, reduction=reduction, **kwargs)


class CategoricalCrossEntropyLoss(NewsRecommenderLoss):
    """Categorical cross-entropy loss for training with multiple candidates.

    This loss is used during training when we have multiple candidates and use softmax activation.
    It computes the categorical cross-entropy between the predicted scores and the true labels.

    Args:
        name: Loss name
        from_logits: Whether the predictions are logits or probabilities
        reduction: Type of reduction to apply to loss. One of:
            - 'none': No reduction
            - 'sum': Sum of losses
            - 'sum_over_batch_size': Sum of losses divided by batch size
            - 'mean': Mean of losses
            - 'mean_with_sample_weight': Mean of losses weighted by sample weights
        label_smoothing: Float in [0, 1]. When > 0, label values are smoothed
    """

    def __init__(
            self,
            name: str = "categorical_crossentropy",
            from_logits: bool = False,
            reduction: str = "sum_over_batch_size",
            label_smoothing: float = 0.0,
            **kwargs: Any,
    ) -> None:
        super().__init__(name=name, reduction=reduction, **kwargs)
        self.from_logits = from_logits
        self.label_smoothing = label_smoothing

    def call(self, y_true: keras.KerasTensor, y_pred: keras.KerasTensor) -> keras.KerasTensor:
        """Compute the loss.

        Args:
            y_true: True labels (one-hot encoded)
            y_pred: Predicted scores (probabilities after softmax)

        Returns:
            Loss value
        """
        return keras.losses.categorical_crossentropy(
            y_true, y_pred, from_logits=self.from_logits, label_smoothing=self.label_smoothing
        )


class BinaryCrossEntropyLoss(NewsRecommenderLoss):
    """Binary cross-entropy loss for validation/testing with single candidate.

    This loss is used during validation/testing when we have a single candidate and use sigmoid activation.
    It computes the binary cross-entropy between the predicted score and the true label.

    Args:
        name: Loss name
        from_logits: Whether the predictions are logits or probabilities
        reduction: Type of reduction to apply to loss. One of:
            - 'none': No reduction
            - 'sum': Sum of losses
            - 'sum_over_batch_size': Sum of losses divided by batch size
            - 'mean': Mean of losses
            - 'mean_with_sample_weight': Mean of losses weighted by sample weights
        label_smoothing: Float in [0, 1]. When > 0, label values are smoothed
    """

    def __init__(
            self,
            name: str = "binary_crossentropy",
            from_logits: bool = False,
            reduction: str = "sum_over_batch_size",
            label_smoothing: float = 0.0,
            **kwargs: Any,
    ) -> None:
        super().__init__(name=name, reduction=reduction, **kwargs)
        self.from_logits = from_logits
        self.label_smoothing = label_smoothing

    def call(self, y_true: keras.KerasTensor, y_pred: keras.KerasTensor) -> keras.KerasTensor:
        """Compute the loss.

        Args:
            y_true: True label (0 or 1)
            y_pred: Predicted score (probability after sigmoid)

        Returns:
            Loss value
        """
        return keras.losses.binary_crossentropy(
            y_true, y_pred, from_logits=self.from_logits, label_smoothing=self.label_smoothing
        )


class CROWNCombinedLoss(NewsRecommenderLoss):
    """Combined loss for CROWN model with primary click prediction and auxiliary category prediction.

    This loss implements the total loss from the CROWN paper:
    L = L_P + beta * L_A

    Where:
    - L_P is the primary click prediction loss (categorical crossentropy)
    - L_A is the auxiliary category prediction loss
    - beta (alpha) controls the weight of the auxiliary task

    Args:
        alpha: Weight for the auxiliary category prediction loss
        num_categories: Number of news categories for one-hot encoding
        name: Loss name
        reduction: Type of reduction to apply to loss
    """

    def __init__(
            self,
            alpha: float = 0.3,
            num_categories: int = 18,
            name: str = "crown_combined_loss",
            reduction: str = "sum_over_batch_size",
            **kwargs: Any,
    ) -> None:
        super().__init__(name=name, reduction=reduction, **kwargs)
        self.alpha = alpha
        self.num_categories = num_categories

    def call(self, y_true: keras.KerasTensor, y_pred: Any) -> keras.KerasTensor:
        """Compute the combined CROWN loss.

        Args:
            y_true: True labels for click prediction (one-hot encoded)
            y_pred: Tuple of (click_scores, category_logits, category_ids) where:
                - click_scores: Predicted click scores after softmax (batch_size, num_candidates)
                - category_logits: Category prediction logits (batch_size * num_candidates, num_categories)
                - category_ids: True category IDs (batch_size * num_candidates,)

        Returns:
            Combined loss value
        """
        # Handle tuple output from CROWN model
        if isinstance(y_pred, (list, tuple)):
            click_scores = y_pred[0]
            category_logits = y_pred[1] if len(y_pred) > 1 else None
            category_ids = y_pred[2] if len(y_pred) > 2 else None
        else:
            # Fallback for models without auxiliary output
            click_scores = y_pred
            category_logits = None
            category_ids = None

        # Primary loss: Click prediction (categorical crossentropy)
        primary_loss = keras.losses.categorical_crossentropy(
            y_true, click_scores, from_logits=False
        )

        # Auxiliary loss: Category prediction (if available)
        if category_logits is not None and category_ids is not None:
            # Create one-hot encoding for true categories
            category_one_hot = keras.ops.one_hot(
                keras.ops.cast(category_ids, "int32"),
                self.num_categories
            )

            # Compute category prediction loss
            auxiliary_loss = keras.losses.categorical_crossentropy(
                category_one_hot,
                category_logits,
                from_logits=True
            )

            # Combine losses with weight
            total_loss = primary_loss + self.alpha * keras.ops.mean(auxiliary_loss)
        else:
            # No auxiliary loss available
            total_loss = primary_loss

        return total_loss


def get_loss(loss_name: str, **kwargs: Any) -> NewsRecommenderLoss:
    """Get a loss function by name.

    Args:
        loss_name: Name of the loss function
        **kwargs: Additional arguments for the loss function including:
            - from_logits: bool
            - reduction: str (one of: 'none', 'sum', 'sum_over_batch_size', 'mean', 'mean_with_sample_weight')
            - label_smoothing: float
            - alpha: float (for CROWN combined loss)
            - num_categories: int (for CROWN combined loss)

    Returns:
        Loss function instance

    Raises:
        ValueError: If the loss name is not recognized
    """
    losses = {
        "categorical_crossentropy": CategoricalCrossEntropyLoss,
        "binary_crossentropy": BinaryCrossEntropyLoss,
        "crown_combined": CROWNCombinedLoss,
    }

    if loss_name not in losses:
        raise ValueError(f"Unknown loss: {loss_name}. Available losses: {list(losses.keys())}")

    return losses[loss_name](**kwargs)
