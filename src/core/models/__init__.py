"""Core model configurations and spec utilities shared across all frameworks."""

from .configs import NRMSConfig, NAMLConfig, LSTURConfig, CROWNConfig
from .spec import spec_to_config, get_model_class, build_model_from_spec

__all__ = [
    "NRMSConfig", "NAMLConfig", "LSTURConfig", "CROWNConfig",
    "spec_to_config", "get_model_class", "build_model_from_spec",
]
