"""Wall-clock timing for training runs.

Minimal by design — per-epoch train wall time, per-epoch validation
wall time, the final test pass wall time, total run time, plus the
JIT-compile / cuDNN-autotune cost (``time_to_first_step_seconds``)
and a parameter count for cross-framework sanity-checking.

GPU utilization, power draw, energy sampling, and per-eval sub-phase
timings were prototyped earlier and removed to keep the surface small.
Add them back later as a separate concern if needed.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Phase statistics
# ---------------------------------------------------------------------------


@dataclass
class PhaseStats:
    """Per-phase wall time + optional throughput.

    A "phase" is one training epoch, one validation pass, one test
    pass, or a custom sub-phase. Throughput fields stay ``None`` for
    phases where step / sample counts aren't tracked.
    """

    wall_seconds: float = 0.0
    n_steps: int | None = None
    n_samples: int | None = None
    throughput_samples_per_s: float | None = None
    throughput_steps_per_s: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Param count — cheap sanity helper, no CUDA required
# ---------------------------------------------------------------------------


def count_params(model: Any, framework: str) -> dict[str, int]:
    """Total + trainable parameter counts. Used as a cross-framework
    sanity check that JAX and PyTorch built the same architecture."""
    if framework == "pytorch":
        try:
            total = sum(p.numel() for p in model.parameters())
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            return {"total": int(total), "trainable": int(trainable)}
        except Exception:
            return {"total": 0, "trainable": 0}
    if framework == "jax":
        try:
            import jax
            import jax.numpy as jnp
            from flax import nnx

            state = nnx.state(model)
            total = 0
            for leaf in jax.tree_util.tree_leaves(state):
                arr = getattr(leaf, "value", leaf)
                if not hasattr(arr, "shape") or not hasattr(arr, "dtype"):
                    continue
                if jnp.issubdtype(arr.dtype, jax.dtypes.prng_key):
                    continue
                total += int(arr.size)
            return {"total": total, "trainable": total}
        except Exception:
            return {"total": 0, "trainable": 0}
    return {"total": 0, "trainable": 0}


# ---------------------------------------------------------------------------
# Top-level run tracker + JSON dump
# ---------------------------------------------------------------------------


@dataclass
class RunTiming:
    """Top-level container for a single training run's wall-clock times."""

    framework: str
    model_name: str
    dataset_name: str
    seed: int
    n_params_total: int = 0
    n_params_trainable: int = 0
    time_to_first_step_seconds: float | None = None
    train_epochs: list[PhaseStats] = field(default_factory=list)
    val_epochs: list[PhaseStats] = field(default_factory=list)
    test: PhaseStats | None = None
    total_seconds: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def dump_run_timing(timing: RunTiming, path: Path | str) -> None:
    """Write a RunTiming as ``timing.json`` (pretty-printed)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(timing.to_dict(), f, indent=2)


def dict_to_run_timing(d: dict[str, Any]) -> RunTiming:
    """Inverse of ``dump_run_timing`` — used by the aggregator."""
    rt = RunTiming(
        framework=d["framework"],
        model_name=d["model_name"],
        dataset_name=d["dataset_name"],
        seed=int(d["seed"]),
        n_params_total=int(d.get("n_params_total", 0)),
        n_params_trainable=int(d.get("n_params_trainable", 0)),
        time_to_first_step_seconds=d.get("time_to_first_step_seconds"),
        total_seconds=d.get("total_seconds"),
    )
    rt.train_epochs = [PhaseStats(**p) for p in d.get("train_epochs", [])]
    rt.val_epochs = [PhaseStats(**p) for p in d.get("val_epochs", [])]
    if d.get("test"):
        rt.test = PhaseStats(**d["test"])
    return rt
