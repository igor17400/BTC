#!/usr/bin/env python3
"""Upload NewsReX model weights to Hugging Face Hub.

One repo per model+framework, seeds as subfolders, best seed at the root:

    newsrex/NRMS-JAX-MIND-small/
    ├── model.safetensors          ← best seed (default download)
    ├── test_results.json
    ├── README.md
    ├── seed_42/model.safetensors
    ├── seed_123/model.safetensors
    └── seed_456/model.safetensors

Usage:
    # Upload all seeds at once (auto-detects best seed for root):
    uv run python scripts/upload_to_hf.py \
        --run-dir outputs/train/MIND-small/NRMS/jax \
        --org newsrex

    # Upload a single seed:
    uv run python scripts/upload_to_hf.py \
        --run-path outputs/train/MIND-small/NRMS/jax/seed_42 \
        --org newsrex

    # Dry run:
    uv run python scripts/upload_to_hf.py \
        --run-dir outputs/train/MIND-small/NRMS/jax \
        --org newsrex --dry-run
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path


def _detect_model_name(run_path: Path) -> str:
    """Detect model name from the run path."""
    for parent in [run_path] + list(run_path.parents):
        name = parent.name
        for model in [
            "nrms",
            "naml",
            "lstur",
            "pprec",
            "crown",
            "digat",
            "glory",
            "miner",
            "caum",
            "tccm",
        ]:
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
    return None


def _detect_split_strategy(run_path: Path) -> str:
    """Detect validation split strategy from the run path.

    Expects the post-migration layout where the split is the parent of
    ``seed_*``: ``.../{framework}/{split}/seed_*/``.
    Falls back to ``"random"`` for legacy paths missing the split segment.
    """
    known_splits = {"random", "chronological", "dev_as_val"}
    for parent in [run_path] + list(run_path.parents):
        if parent.name in known_splits:
            return parent.name
    return "random"


def _detect_encoder(run_path: Path) -> str:
    """Detect the encoder (glove / bert / ...) from the run path.

    Layout is ``.../{framework}/{encoder}/{split}/seed_*/`` — the encoder is
    the segment immediately below the framework dir. Falls back to ``"glove"``.
    """
    parts = run_path.resolve().parts
    frameworks = {"pytorch", "jax", "keras"}
    for i, p in enumerate(parts):
        if p in frameworks and i + 1 < len(parts):
            return parts[i + 1]
    return "glove"


def _load_test_results(run_path: Path) -> dict:
    """Load test results if available."""
    p = run_path / "test_results.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {}


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


def _format_size(weight_files: list[Path]) -> str:
    """Format total weight file size."""
    total = sum(f.stat().st_size for f in weight_files)
    if total > 1e9:
        return f"{total / 1e9:.1f} GB"
    return f"{total / 1e6:.1f} MB"


def _find_best_seed(seed_dirs: list[Path]) -> Path:
    """Find the seed with the best test AUC."""
    best_auc = -1.0
    best_dir = seed_dirs[0]
    for sd in seed_dirs:
        results = _load_test_results(sd)
        auc = results.get("auc", -1.0)
        if auc > best_auc:
            best_auc = auc
            best_dir = sd
    return best_dir


def _generate_readme(
    repo_id: str,
    model_name: str,
    framework: str,
    dataset: str,
    seed_results: list[dict],
    best_seed: str,
    run_path: Path | None = None,
) -> str:
    """Generate the repo README.md with results across all seeds."""
    card = f"""---
library_name: newsrex
tags:
- news-recommendation
- {model_name.lower()}
- {framework}
- mind
datasets:
- mind
license: apache-2.0
---

# NewsReX {model_name} — {framework.upper()} — {dataset}

{model_name} news recommendation model trained on {dataset} using the
[NewsReX](https://github.com/igor17400/NewsReX) framework ({framework.upper()}).

## Test Results

| Seed | AUC | MRR | NDCG@5 | NDCG@10 |
|------|-----|-----|--------|---------|
"""
    for sr in seed_results:
        marker = " *" if sr["seed"] == best_seed else ""
        card += f"| {sr['seed']}{marker} | {sr.get('auc', 0):.4f} | {sr.get('mrr', 0):.4f} | {sr.get('ndcg@5', 0):.4f} | {sr.get('ndcg@10', 0):.4f} |\n"

    # Compute mean ± std
    if len(seed_results) > 1:
        import numpy as np

        metrics = ["auc", "mrr", "ndcg@5", "ndcg@10"]
        means = {m: np.mean([sr[m] for sr in seed_results if m in sr]) for m in metrics}
        stds = {m: np.std([sr[m] for sr in seed_results if m in sr]) for m in metrics}
        card += f"| **mean ± std** | **{means['auc']:.4f}±{stds['auc']:.4f}** | **{means['mrr']:.4f}±{stds['mrr']:.4f}** | **{means['ndcg@5']:.4f}±{stds['ndcg@5']:.4f}** | **{means['ndcg@10']:.4f}±{stds['ndcg@10']:.4f}** |\n"

    card += "\n\\* Best seed (weights at repo root)\n"

    # Include spec from Hydra config if available.
    if run_path:
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

    card += f"""
## Repository Structure

```
{repo_id}/
├── model.safetensors          ← best seed ({best_seed})
├── test_results.json
├── training_run_summary.json
"""
    for sr in seed_results:
        card += f"├── seed_{sr['seed']}/model.safetensors\n"

    card += f"""└── README.md
```

## Usage

```bash
git clone https://github.com/igor17400/NewsReX.git
cd NewsReX && uv sync

# Run evaluation with best seed weights
uv run python src/eval.py \\
    experiment=mind/glove/{model_name.lower()} \\
    framework={framework} \\
    weights=hf://{repo_id}/model.safetensors

# Run evaluation with a specific seed
uv run python src/eval.py \\
    experiment=mind/glove/{model_name.lower()} \\
    framework={framework} \\
    weights=hf://{repo_id}/seed_42/model.safetensors
```

## Citation

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


def upload(
    run_dir: Path | None = None,
    run_path: Path | None = None,
    org: str = "newsrex",
    private: bool = False,
    dry_run: bool = False,
):
    """Upload trained model(s) to Hugging Face Hub.

    Either ``run_dir`` (directory containing seed_* folders) or
    ``run_path`` (single seed directory) must be provided.
    """
    from huggingface_hub import HfApi, create_repo

    # Collect seed directories.
    if run_dir:
        run_dir = Path(run_dir).resolve()
        seed_dirs = sorted(run_dir.glob("seed_*"))
        if not seed_dirs:
            raise FileNotFoundError(f"No seed_* directories found in {run_dir}")
    elif run_path:
        run_path = Path(run_path).resolve()
        seed_dirs = [run_path]
    else:
        raise ValueError("Provide either --run-dir or --run-path")

    # Auto-detect metadata from the first seed dir.
    ref_dir = seed_dirs[0]
    model_name = _detect_model_name(ref_dir)
    framework = _detect_framework(ref_dir)
    dataset = _detect_dataset(ref_dir)
    split_strategy = _detect_split_strategy(ref_dir)
    encoder = _detect_encoder(ref_dir)
    # Repo name includes the encoder AND split strategy so glove/bert and
    # dev_as_val/random weights don't collide in the same HF repo.
    repo_id = (
        f"{org}/{model_name}-{framework.upper()}-"
        f"{dataset.replace(' ', '-')}-{encoder}-{split_strategy}"
    )

    # Collect results per seed.
    seed_results = []
    for sd in seed_dirs:
        seed = _detect_seed(sd)
        weights = _find_weights(sd)
        if not weights:
            print(f"Warning: no weights in {sd}, skipping")
            continue
        results = _load_test_results(sd)
        results["seed"] = seed
        results["_dir"] = sd
        results["_weights"] = weights
        seed_results.append(results)

    if not seed_results:
        raise FileNotFoundError("No seeds with weights found")

    # Find best seed.
    best_idx = max(
        range(len(seed_results)), key=lambda i: seed_results[i].get("auc", -1)
    )
    best_seed = seed_results[best_idx]["seed"]
    best_dir = seed_results[best_idx]["_dir"]

    print(f"Model:     {model_name}")
    print(f"Framework: {framework}")
    print(f"Dataset:   {dataset}")
    print(f"Repo:      {repo_id}")
    print(f"Seeds:     {[sr['seed'] for sr in seed_results]}")
    print(
        f"Best seed: {best_seed} (AUC={seed_results[best_idx].get('auc', 'N/A'):.4f})"
    )

    if dry_run:
        print("\n[DRY RUN] Would upload the above. Exiting.")
        return

    # Build upload directory.
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        # Copy best seed weights to root.
        for wf in seed_results[best_idx]["_weights"]:
            shutil.copy2(wf, tmp_path / wf.name)

        # Copy best seed summary + test results to root.
        for name in ["training_run_summary.json", "test_results.json"]:
            src = best_dir / name
            if src.exists():
                shutil.copy2(src, tmp_path / name)

        # Copy each seed into its subfolder.
        for sr in seed_results:
            seed_subdir = tmp_path / f"seed_{sr['seed']}"
            seed_subdir.mkdir()
            for wf in sr["_weights"]:
                shutil.copy2(wf, seed_subdir / wf.name)
            for name in ["training_run_summary.json", "test_results.json"]:
                src = sr["_dir"] / name
                if src.exists():
                    shutil.copy2(src, seed_subdir / name)

        # Generate README.
        readme = _generate_readme(
            repo_id,
            model_name,
            framework,
            dataset,
            [
                {k: v for k, v in sr.items() if not k.startswith("_")}
                for sr in seed_results
            ],
            best_seed,
            best_dir,
        )
        (tmp_path / "README.md").write_text(readme)

        # Create repo and upload.
        api = HfApi()
        create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
        api.upload_folder(
            folder_path=str(tmp_path),
            repo_id=repo_id,
            commit_message=f"Upload {model_name} ({framework}) trained on {dataset} — {len(seed_results)} seeds",
        )

    print(f"\nUploaded to: https://huggingface.co/{repo_id}")


def main():
    parser = argparse.ArgumentParser(
        description="Upload NewsReX model weights to Hugging Face Hub",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--run-dir",
        type=str,
        help="Directory containing seed_* folders (uploads all seeds)",
    )
    group.add_argument(
        "--run-path",
        type=str,
        help="Single seed run directory",
    )
    parser.add_argument(
        "--org",
        type=str,
        default="newsrex",
        help="HuggingFace org (default: newsrex)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create a private repository",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be uploaded without actually uploading",
    )
    args = parser.parse_args()

    upload(
        run_dir=args.run_dir,
        run_path=args.run_path,
        org=args.org,
        private=args.private,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
