"""Model-specific setup hooks.

Most models (NRMS, NAML, LSTUR, MINER, PP-REC, CROWN) use the standard
pipeline: ``_build_train_features`` → standard dataloader → default
evaluator. DIGAT and GLORY need custom data pipelines, graph
construction, and evaluation closures.

Usage in runners::

    from src.core.setup import setup_model

    result = setup_model(spec, dataset_provider, processed_news)
    if result is not None:
        # Use result.features, result.train_dataset, result.make_eval_fn, etc.
    else:
        # Standard path
"""

from .digat import setup_digat
from .glory import setup_glory
from .tccm import setup_tccm
from .types import ModelSetupResult


def setup_model(
    spec, dataset_provider, processed_news, encoder_cfg=None
) -> ModelSetupResult | None:
    """Dispatch model-specific setup by name.

    Returns None for standard models that don't need custom setup.

    ``encoder_cfg`` (``cfg.encoder``) is threaded through so hooks that
    materialise their own training features can branch on
    ``encoder.type`` the same way the runner's ``_build_train_features``
    does (GloVe tokens vs PLM news ids).
    """
    name = spec.model.name.lower()
    if name == "digat":
        return setup_digat(spec, dataset_provider, processed_news, encoder_cfg)
    elif name == "glory":
        return setup_glory(spec, dataset_provider, processed_news)
    elif name == "tccm":
        return setup_tccm(spec, dataset_provider, processed_news, encoder_cfg)
    return None


__all__ = ["ModelSetupResult", "setup_model"]
