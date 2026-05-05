"""Types for model-specific setup results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class ModelSetupResult:
    """Return type for model-specific setup hooks.

    Attributes:
        features: Training features dict, or None for models with custom
            dataloaders (e.g. GLORY).
        labels: Training labels array, or None.
        train_dataset: Custom dataset object (e.g. GLORYTrainDataset).
            None for models that use standard features/labels.
        eval_context: Mutable dict holding state shared between
            ``make_eval_fn`` and ``rebuild_test_remap``. The eval_fn
            closure reads from this dict, so ``rebuild_test_remap`` can
            update it (e.g. id_remap) and the eval_fn picks up the
            change automatically.
        make_eval_fn: Factory that creates an eval_fn closure.
            Signature: ``(model, adapter, metrics_engine, dataset_provider,
            processed_news, eval_batch_size, output_run_dir,
            build_eval_dataloaders=None) -> eval_fn`` where
            eval_fn: ``(model, mode="val", epoch=None) -> metrics_dict``.
            ``build_eval_dataloaders`` is the framework-native eval-data
            builder (returns dict or 3-tuple of dataloaders) provided by
            the runner; hooks that don't need it (DIGAT, GLORY) ignore
            it via ``**_`` in their signature.
        rebuild_test_remap: Called before test evaluation to rebuild
            ID remaps now that test data is loaded. Mutates
            ``eval_context`` in place. Signature:
            ``(dataset_provider, processed_news) -> None``.
    """

    features: dict | None = None
    labels: Any | None = None
    train_dataset: Any | None = None
    eval_context: dict = field(default_factory=dict)
    make_eval_fn: Callable[..., Callable] | None = None
    rebuild_test_remap: Callable[..., None] | None = None
