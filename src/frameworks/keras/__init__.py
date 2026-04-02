"""Keras framework components for NewsReX.

This package contains all Keras-specific code including models, layers,
dataloaders, training orchestration, evaluation, and utilities.
"""

from .models import NRMS, NAML, LSTUR, BaseModel
from .layers import AdditiveAttentionLayer, ComputeMasking, OverwriteMasking
from .losses import get_loss, CategoricalCrossEntropyLoss, BinaryCrossEntropyLoss
from .training import training_loop_orchestrator
from .evaluation import run_evaluation_epoch
from .device import setup_device
from .runner import run
from .utils import (
    initialize_model_and_dataset,
    warmup_jit_compilation,
    create_news_metrics,
    LightweightNewsMetrics,
    test_step_fn,
)
