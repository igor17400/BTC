#!/usr/bin/env python3
"""Upload NewsReX model weights to Hugging Face Hub.

Usage:
    # Upload a specific run:
    uv run python scripts/upload_to_hf.py \
        --run-path outputs/mind_nrms/jax/2026-04-10-07-26-23 \
        --repo-id igor17400/NewsReX-NRMS-MIND-small \
        --model-name NRMS

    # Upload with auto-detected model name:
    uv run python scripts/upload_to_hf.py \
        --run-path outputs/mind_nrms/jax/2026-04-10-07-26-23 \
        --repo-id igor17400/NewsReX-NRMS-MIND-small

    # Dry run (show what would be uploaded):
    uv run python scripts/upload_to_hf.py \
        --run-path outputs/mind_nrms/jax/2026-04-10-07-26-23 \
        --repo-id igor17400/NewsReX-NRMS-MIND-small \
        --dry-run
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path


def _detect_model_name(run_path: Path) -> str:
    """Detect model name from the run path (e.g. mind_nrms -> NRMS)."""
    for parent in [run_path] + list(run_path.parents):
        name = parent.name
        for model in ["nrms", "naml", "lstur", "pprec", "crown", "digat", "glory"]:
            if model in name.lower():
                return model.upper()
    return "unknown"


def _detect_framework(run_path: Path) -> str:
    """Detect framework from the run path."""
    for parent in [run_path] + list(run_path.parents):
        if parent.name in ("pytorch", "keras", "jax"):
            return parent.name
    return "unknown"


def _detect_dataset(run_path: Path) -> str:
    """Detect dataset from the run path."""
    for parent in [run_path] + list(run_path.parents):
        if "mind" in parent.name.lower():
            return "MIND-small"
        if "japanese" in parent.name.lower():
            return "Japanese News"
    return "unknown"


def _detect_seed(run_path: Path) -> str | None:
    """Detect seed from the run path (e.g. seed_42 -> 42)."""
    for parent in [run_path] + list(run_path.parents):
        if parent.name.startswith("seed_"):
            return parent.name.replace("seed_", "")
    # Fallback: check the Hydra config if available.
    for name in ["config.yaml"]:
        cfg_path = run_path / ".hydra" / name
        if cfg_path.exists():
            import yaml
            with open(cfg_path) as f:
                cfg = yaml.safe_load(f)
            if cfg and "seed" in cfg:
                return str(cfg["seed"])
    return None


def _load_summary(run_path: Path) -> dict:
    """Load training run summary if available."""
    for name in ["training_run_summary.json", "run_summary.json"]:
        p = run_path / name
        if p.exists():
            with open(p) as f:
                return json.load(f)
    return {}


def _find_weights(run_path: Path) -> list[Path]:
    """Find model weight files in the run directory."""
    models_dir = run_path / "models"
    if not models_dir.exists():
        return []
    patterns = ["best_model.pt", "*_best.weights.h5", "*.safetensors"]
    files = []
    for pattern in patterns:
        files.extend(models_dir.glob(pattern))
    return sorted(files)


def _find_config(run_path: Path) -> Path | None:
    """Find model config file."""
    for name in ["model_config.yaml", "config.yaml"]:
        p = run_path / "models" / name
        if p.exists():
            return p
    return None


def _generate_model_card(
    model_name: str,
    framework: str,
    dataset: str,
    summary: dict,
    weight_files: list[Path],
    seed: str | None = None,
    run_path: Path | None = None,
) -> str:
    """Generate a Hugging Face model card (README.md)."""
    # Extract metrics
    best_val = summary.get("best_validation_summary", {})
    test_metrics = summary.get("final_test_metrics", {})
    config = summary.get("configuration", {})
    train_cfg = config.get("train", {})

    card = f"""---
library_name: newsrex
tags:
- news-recommendation
- {model_name.lower()}
- mind
datasets:
- mind
license: apache-2.0
---

# NewsReX {model_name} — {dataset}

{model_name} news recommendation model trained on {dataset} using the
[NewsReX](https://github.com/igor17400/NewsReX) framework.

## Model Details

| Property | Value |
|----------|-------|
| Architecture | {model_name} |
| Framework | {framework} |
| Dataset | {dataset} |
| Seed | {seed or 'N/A'} |
| Parameters | {_format_size(weight_files)} |
"""

    # Include full spec from Hydra config if available.
    hydra_cfg_path = run_path / ".hydra" / "config.yaml"
    if hydra_cfg_path.exists():
        import yaml
        with open(hydra_cfg_path) as f:
            hydra_cfg = yaml.safe_load(f)
        spec = hydra_cfg.get("spec", {})
        if spec:
            spec_yaml = yaml.dump(spec, default_flow_style=False, sort_keys=False)
            card += f"""
## Experiment Configuration

```yaml
{spec_yaml}```
"""

    if best_val:
        card += f"""
## Validation Results (Best Epoch {int(best_val.get('epoch_number', 0))})

| Metric | Value |
|--------|-------|
| AUC | {best_val.get('val_auc', 'N/A'):.4f} |
| MRR | {best_val.get('val_mrr', 'N/A'):.4f} |
| NDCG@5 | {best_val.get('val_ndcg@5', 'N/A'):.4f} |
| NDCG@10 | {best_val.get('val_ndcg@10', 'N/A'):.4f} |
"""

    if test_metrics:
        card += f"""
## Test Results

| Metric | Value |
|--------|-------|
| AUC | {test_metrics.get('auc', 'N/A'):.4f} |
| MRR | {test_metrics.get('mrr', 'N/A'):.4f} |
| NDCG@5 | {test_metrics.get('ndcg@5', 'N/A'):.4f} |
| NDCG@10 | {test_metrics.get('ndcg@10', 'N/A'):.4f} |
"""

    repo_name = f"newsrex/{model_name}-{framework.upper()}-{dataset.replace(' ', '-')}"
    if seed:
        repo_name += f"-seed{seed}"

    card += f"""
## Usage

### 1. Download weights

```python
from huggingface_hub import hf_hub_download

weights_path = hf_hub_download(
    repo_id="{repo_name}",
    filename="model.safetensors",
)
```

### 2. Run evaluation on MIND test set

```bash
# Clone NewsReX and install dependencies
git clone https://github.com/igor17400/NewsReX.git
cd NewsReX && uv sync

# Run evaluation with downloaded weights
uv run python src/train.py \\
    experiment=mind/{model_name.lower()} \\
    framework={framework} \\
    train.num_epochs=0 \\
    weights=${{weights_path}}
```

### 3. Load weights in Python

```python
import jax
import jax.numpy as jnp
from flax import nnx
from safetensors.numpy import load_file

from src.core.models.spec import build_model_from_spec

# Build model from spec
model = build_model_from_spec(spec, "{framework}", processed_news)

# Load weights
weights = load_file(weights_path)
state = nnx.state(model)
new_leaves = []
for path, leaf in jax.tree_util.tree_leaves_with_path(state):
    name = ".".join(str(k) for k in path)
    new_leaves.append(jnp.asarray(weights[name]) if name in weights else leaf)
nnx.update(model, jax.tree_util.tree_unflatten(
    jax.tree_util.tree_structure(state), new_leaves,
))
```

## Citation

If you use this model, please cite NewsReX:

```bibtex
@misc{{newsrex2026,
  title={{NewsReX: An Open-Source Multi-Framework for Neural News Recommendation}},
  author={{Igor L. R. Azevedo and Toyotaro Suzumura and Yuichiro Yasui}},
  year={{2025}},
  eprint={{2508.21572}},
  archivePrefix={{arXiv}},
  primaryClass={{cs.IR}},
  url={{https://arxiv.org/abs/2508.21572}},
}}
```
"""
    return card


def _format_size(weight_files: list[Path]) -> str:
    """Format total weight file size."""
    total = sum(f.stat().st_size for f in weight_files)
    if total > 1e9:
        return f"{total / 1e9:.1f} GB"
    return f"{total / 1e6:.1f} MB"


def upload(
    run_path: Path,
    repo_id: str,
    model_name: str | None = None,
    private: bool = False,
    dry_run: bool = False,
):
    """Upload a trained model to Hugging Face Hub."""
    from huggingface_hub import HfApi, create_repo

    run_path = Path(run_path).resolve()
    if not run_path.exists():
        raise FileNotFoundError(f"Run path not found: {run_path}")

    # Auto-detect metadata.
    if model_name is None:
        model_name = _detect_model_name(run_path)
    framework = _detect_framework(run_path)
    dataset = _detect_dataset(run_path)
    seed = _detect_seed(run_path)

    # Find files to upload.
    weight_files = _find_weights(run_path)
    if not weight_files:
        raise FileNotFoundError(f"No weight files found in {run_path}/models/")

    config_file = _find_config(run_path)
    summary = _load_summary(run_path)

    print(f"Model: {model_name}")
    print(f"Framework: {framework}")
    print(f"Dataset: {dataset}")
    print(f"Seed: {seed}")
    print(f"Repo: {repo_id}")
    print(f"Weight files: {[f.name for f in weight_files]}")
    if config_file:
        print(f"Config: {config_file.name}")
    if summary:
        test = summary.get("final_test_metrics", {})
        if test:
            print(f"Test AUC: {test.get('auc', 'N/A')}")

    if dry_run:
        print("\n[DRY RUN] Would upload the above files. Exiting.")
        return

    # Create a temporary upload directory with organized contents.
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        # Copy weight files.
        for wf in weight_files:
            shutil.copy2(wf, tmp_path / wf.name)

        # Copy config.
        if config_file:
            shutil.copy2(config_file, tmp_path / config_file.name)

        # Copy training summary.
        for name in ["training_run_summary.json", "run_summary.json"]:
            src = run_path / name
            if src.exists():
                shutil.copy2(src, tmp_path / name)

        # Generate model card.
        card = _generate_model_card(
            model_name, framework, dataset, summary, weight_files, seed, run_path,
        )
        (tmp_path / "README.md").write_text(card)

        # Create repo and upload.
        api = HfApi()
        create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
        seed_str = f", seed={seed}" if seed else ""
        api.upload_folder(
            folder_path=str(tmp_path),
            repo_id=repo_id,
            commit_message=f"Upload {model_name} ({framework}{seed_str}) trained on {dataset}",
        )

    print(f"\nUploaded to: https://huggingface.co/{repo_id}")


def main():
    parser = argparse.ArgumentParser(
        description="Upload NewsReX model weights to Hugging Face Hub",
    )
    parser.add_argument(
        "--run-path", type=str, required=True,
        help="Path to the training run directory (e.g. outputs/mind_nrms/jax/2026-...)",
    )
    parser.add_argument(
        "--repo-id", type=str, required=True,
        help="Hugging Face repo ID (e.g. igor17400/NewsReX-NRMS)",
    )
    parser.add_argument(
        "--model-name", type=str, default=None,
        help="Model name (auto-detected from path if not provided)",
    )
    parser.add_argument(
        "--private", action="store_true",
        help="Create a private repository",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be uploaded without actually uploading",
    )
    args = parser.parse_args()

    upload(
        run_path=args.run_path,
        repo_id=args.repo_id,
        model_name=args.model_name,
        private=args.private,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
