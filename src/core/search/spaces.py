"""Search space definitions for Optuna hyperparameter search.

Organized by phase:
- Phase 1: Training dynamics (learning rate, dropout, negative sampling)
- Phase 2: Architecture parameters (model-specific, stubs for future expansion)

Each function takes an ``optuna.Trial`` and returns a list of Hydra override
strings that map directly to spec YAML paths.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import optuna


# ---------------------------------------------------------------------------
# Phase 1: Training dynamics (model-agnostic)
# ---------------------------------------------------------------------------


def phase1_overrides(trial: optuna.Trial) -> list[str]:
    """Phase 1: learning rate, dropout, negative sampling."""
    lr = trial.suggest_categorical("learning_rate", [1e-4, 3e-4, 5e-4])
    dropout = trial.suggest_categorical("dropout_rate", [0.2, 0.3])
    neg_k = trial.suggest_categorical("neg_candidates", [4, 8])

    return [
        f"spec.training.learning_rate={lr}",
        f"spec.model.dropout_rate={dropout}",
        f"spec.training.negative_sampling.candidates={neg_k}",
        # max_impressions_length = neg_candidates + 1 (1 positive + k negatives)
        # see src/core/data/processing/sampling.py line 67
        f"spec.inputs.impressions.max_length={neg_k + 1}",
    ]


# ---------------------------------------------------------------------------
# Phase 2: Architecture parameters (model-specific, stubs)
# ---------------------------------------------------------------------------


def phase2_nrms_overrides(trial: optuna.Trial) -> list[str]:
    """Phase 2: NRMS architecture parameters."""
    overrides = phase1_overrides(trial)

    num_heads = trial.suggest_categorical("num_heads", [12, 15, 20, 25])
    head_dim = trial.suggest_categorical("head_dim", [12, 15, 20, 25])
    attn_dim = trial.suggest_int("attention_hidden_dim", 100, 300, step=50)

    overrides += [
        f"spec.model.architecture.news_encoder.num_heads={num_heads}",
        f"spec.model.architecture.news_encoder.head_dim={head_dim}",
        f"spec.model.architecture.news_encoder.attention_hidden_dim={attn_dim}",
        f"spec.model.architecture.user_encoder.num_heads={num_heads}",
        f"spec.model.architecture.user_encoder.head_dim={head_dim}",
        f"spec.model.architecture.user_encoder.attention_hidden_dim={attn_dim}",
    ]
    return overrides


def phase2_naml_overrides(trial: optuna.Trial) -> list[str]:
    """Phase 2: NAML architecture parameters."""
    overrides = phase1_overrides(trial)

    filter_num = trial.suggest_categorical("cnn_filter_num", [200, 300, 400])
    kernel_size = trial.suggest_categorical("cnn_kernel_size", [3, 5])
    word_attn_dim = trial.suggest_int("word_attention_query_dim", 100, 300, step=50)
    view_attn_dim = trial.suggest_int("view_attention_query_dim", 100, 300, step=50)
    user_attn_dim = trial.suggest_int("user_attention_query_dim", 100, 300, step=50)

    overrides += [
        f"spec.model.architecture.news_encoder.views.title.filter_num={filter_num}",
        f"spec.model.architecture.news_encoder.views.title.kernel_size={kernel_size}",
        f"spec.model.architecture.news_encoder.views.abstract.filter_num={filter_num}",
        f"spec.model.architecture.news_encoder.views.abstract.kernel_size={kernel_size}",
        f"spec.model.architecture.news_encoder.views.title.attention_query_dim={word_attn_dim}",
        f"spec.model.architecture.news_encoder.views.abstract.attention_query_dim={word_attn_dim}",
        f"spec.model.architecture.news_encoder.view_attention_query_dim={view_attn_dim}",
        f"spec.model.architecture.user_encoder.attention_query_dim={user_attn_dim}",
    ]
    return overrides


def phase2_lstur_overrides(trial: optuna.Trial) -> list[str]:
    """Phase 2: LSTUR architecture parameters."""
    overrides = phase1_overrides(trial)

    filter_num = trial.suggest_categorical("cnn_filter_num", [200, 300, 400])
    attn_dim = trial.suggest_int("attention_hidden_dim", 100, 300, step=50)
    variant = trial.suggest_categorical("user_encoder_type", ["ini", "con"])

    # GRU unit must match news encoder output:
    # filter_num + category_dim(100) + subcategory_dim(100)
    gru_unit = filter_num + 200

    overrides += [
        f"spec.model.architecture.news_encoder.filter_num={filter_num}",
        f"spec.model.architecture.news_encoder.attention_hidden_dim={attn_dim}",
        f"spec.model.architecture.user_encoder.gru_unit={gru_unit}",
        f"spec.model.architecture.user_encoder.variant={variant}",
    ]
    return overrides


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

SEARCH_SPACES: dict[int, dict[str, callable]] = {
    1: {"all": phase1_overrides},
    2: {
        "nrms": phase2_nrms_overrides,
        "naml": phase2_naml_overrides,
        "lstur": phase2_lstur_overrides,
    },
}


def get_search_space(phase: int, model: str):
    """Get the search space function for a given phase and model.

    Args:
        phase: Search phase (1 = training dynamics, 2 = architecture).
        model: Model name (nrms, naml, lstur).

    Returns:
        Callable that takes an ``optuna.Trial`` and returns Hydra overrides.
    """
    phase_spaces = SEARCH_SPACES.get(phase)
    if phase_spaces is None:
        raise ValueError(f"Unknown phase {phase}. Available: {list(SEARCH_SPACES)}")

    if "all" in phase_spaces:
        return phase_spaces["all"]

    fn = phase_spaces.get(model)
    if fn is None:
        raise ValueError(
            f"No phase {phase} search space for model '{model}'. "
            f"Available: {list(phase_spaces)}"
        )
    return fn
