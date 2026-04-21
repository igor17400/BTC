# _NewsReX_: A Modular Framework for News Recommendation Research

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

NewsReX is a modular and extensible framework for news recommendation systems research, implementing state-of-the-art models with a focus on reproducibility and ease of use. The framework supports three backends — **JAX/Flax**, **Keras 3**, and **PyTorch** — behind a unified Hydra-based configuration system. This project draws inspiration from [newsreclib](https://github.com/andreeaiana/newsreclib).

## Features

- **6 SOTA news recommendation models** with a unified training and evaluation pipeline
- **Multi-framework support**: JAX/Flax (JIT + XLA), Keras 3, PyTorch — switchable via a single flag
- **Hydra-based configuration** with composable experiment, model spec, and dataset configs
- **Multi-seed training** with automatic mean ± std aggregation
- **Optuna hyperparameter search** with a two-phase search strategy
- **VewsX** — interactive Streamlit dashboard for dataset and prediction analysis
- **W&B integration** for experiment tracking
- **Cross-framework benchmarking** utilities

## Supported Models

| Model | Keras | JAX | PyTorch | Reference |
|-------|:-----:|:---:|:-------:|-----------|
| **NRMS** — Neural Recommendation with Multi-Head Self-Attention | ✓ | ✓ | ✓ | EMNLP 2019 |
| **NAML** — Attentive Multi-View Learning | ✓ | ✓ | ✓ | EMNLP 2019 |
| **LSTUR** — Long- and Short-term User Representations | ✓ | ✓ | ✓ | NAACL 2020 |
| **CROWN** — Intent Disentanglement + Bipartite GNN | ✓ | ✓ | ✓ | WWW 2025 |
| **PP-Rec** — Popularity-Aware Recommendation | ✓ | ✓ | ✓ | ACL 2021 |
| **DIGAT** — Dual Interactive Graph Attention Networks | — | — | ✓ | EMNLP 2022 |

## Supported Datasets

| Dataset | Description |
|---------|-------------|
| **MIND** | Microsoft News Dataset (small and large) — downloaded from HuggingFace |
| **Japanese** | Japanese news dataset with language-specific text processing |
| **Custom** | Generic loader for MIND-format datasets |
| **Synthetic** | In-memory randomly generated data — no downloads, used for smoke tests |

## Project Structure

```
NewsReX/
├── src/
│   ├── train.py                    # Main entry point (Hydra dispatcher)
│   ├── search.py                   # Hyperparameter search entry point
│   ├── core/                       # Framework-agnostic logic
│   │   ├── data/
│   │   │   ├── datasets/           # Dataset classes (MIND, Japanese, Custom, Synthetic)
│   │   │   ├── download/           # HuggingFace downloader
│   │   │   ├── encoders/           # GloVe, BPEmb embedding encoders
│   │   │   ├── loaders/            # Data caching
│   │   │   └── processing/         # Pipeline: news, behaviors, vocab, embeddings,
│   │   │                           #   sampling, popularity, SAG (DIGAT), knowledge graph
│   │   ├── models/
│   │   │   ├── configs.py          # Model config dataclasses (one per model)
│   │   │   ├── spec.py             # Spec → config factory
│   │   │   └── evaluations/        # Evaluation strategies (default, pp_rec)
│   │   ├── metrics/                # AUC, MRR, NDCG@5, NDCG@10
│   │   ├── losses.py               # Unified loss registry
│   │   ├── search/                 # Optuna-based HPO (optimizer, search spaces)
│   │   └── io/                     # Logging (Rich + W&B), config utils, saving
│   ├── frameworks/
│   │   ├── keras/                  # runner, models, dataloaders, losses, layers
│   │   ├── pytorch/                # runner, models, dataloaders, losses, layers
│   │   │   └── models/digat.py     # DIGAT (PyTorch-only)
│   │   └── jax/                    # runner, models, dataloaders, losses, layers
│   └── benchmarks/                 # Cross-framework benchmarking runner + reporting
├── configs/
│   ├── config.yaml                 # Base config (framework, device, train/eval defaults)
│   ├── spec/                       # Model architecture specs (nrms, naml, lstur, crown, pprec, digat)
│   ├── experiment/
│   │   ├── mind/                   # mind/{nrms,naml,lstur,crown,pprec,digat}.yaml
│   │   ├── japanese/               # japanese/{nrms,naml,lstur}.yaml
│   │   └── smoke/                  # Fast smoke-test configs for all models
│   ├── dataset/                    # mind, japanese, custom, synthetic
│   └── search.yaml                 # HPO config
├── vewsx/                          # Streamlit visualization platform
│   ├── app.py
│   ├── pages/raw/                  # Dataset overview, news corpus, temporal, user behavior
│   └── pages/processed/            # Tensor shapes, popularity features, sampling stats
├── tests/
│   ├── smoke.py                    # All-model smoke tests (all frameworks)
│   ├── smoke_digat.py              # DIGAT-specific tests
│   └── parity.py                   # Cross-framework parity verification
└── scripts/                        # Diagnostics and utility scripts
```

## Quick Start

### Prerequisites

Python 3.12–3.14 is required. The project uses [uv](https://docs.astral.sh/uv/) as the primary package manager.

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Installation

```bash
git clone https://github.com/igor17400/NewsReX.git
cd NewsReX

# Install with JAX backend (CPU)
uv sync --extra jax

# Install with JAX + CUDA 12 GPU support
uv sync --extra jax-cuda

# Install with PyTorch backend
uv sync --extra pytorch

# Install all frameworks
uv sync --extra all-cuda
```

To verify JAX sees your GPU:
```bash
uv run python -c "import jax; print(jax.default_backend())"
# Expected: gpu
```

### Training a Model

```bash
# JAX backend — NRMS on MIND-small
uv run python src/train.py experiment=mind/nrms framework=jax

# PyTorch backend — CROWN on MIND-small
uv run python src/train.py experiment=mind/crown framework=pytorch

# PyTorch — DIGAT (PyTorch-only model)
uv run python src/train.py experiment=mind/digat framework=pytorch

# Keras backend — NAML on MIND-small
uv run python src/train.py experiment=mind/naml framework=keras
```

Override any config value on the command line:

```bash
uv run python src/train.py experiment=mind/nrms framework=jax \
    train.batch_size=256 train.num_epochs=5 train.learning_rate=0.0002
```

### Multi-Seed Training

```bash
uv run python src/train.py experiment=mind/nrms framework=jax \
    multi_seed.enabled=true
```

Results (mean ± std across seeds) are saved to `outputs/{name}/{framework}/multi_seed/`.

### Hyperparameter Search

```bash
uv run python src/search.py search.model=nrms search.framework=pytorch search.n_trials=20

# Launch the Optuna dashboard
uv run optuna-dashboard sqlite:///outputs/search/optuna.db
```

### Smoke Tests

```bash
# All models, all frameworks (uses synthetic in-memory data — no downloads)
uv run python tests/smoke.py

# DIGAT only
uv run python tests/smoke_digat.py
```

### VewsX Visualization Platform

```bash
uv run streamlit run vewsx/app.py
```

Provides interactive dashboards for raw dataset exploration (news corpus, user behavior, temporal trends, user segments) and processed data analysis (tensor shapes, popularity/CTR distributions, sampling statistics).

## Configuration System

Configuration is composed from three layers via Hydra:

1. **Base** (`configs/config.yaml`) — framework, device, training/eval defaults, W&B, multi-seed
2. **Spec** (`configs/spec/{model}.yaml`) — model architecture, input lengths, feature flags, loss
3. **Dataset** (`configs/dataset/{dataset}.yaml`) — data paths, embedding type, preprocessing params

Experiment configs (`configs/experiment/{dataset}/{model}.yaml`) wire a spec to a dataset and can override any base setting.

## Metrics

All models are evaluated on:
- **AUC** — Area Under the ROC Curve
- **MRR** — Mean Reciprocal Rank
- **NDCG@5** — Normalized Discounted Cumulative Gain at 5
- **NDCG@10** — Normalized Discounted Cumulative Gain at 10

---

## Authors & Affiliations

- **Igor L.R. Azevedo** — The University of Tokyo · igorazevedo@acm.org · ORCID: 0000-0001-5144-825X
- **Toyotaro Suzumura** — The University of Tokyo · suzumura@acm.org · ORCID: 0000-0001-6412-8386
- **Yuichiro Yasui** — Nikkei Inc. · yuichiro.yasui@nex.nikkei.com · ORCID: 0000-0002-4175-9318

## Citation

```bibtex
@misc{azevedo2025newsrexefficientapproachnews,
      title={NewsReX: A More Efficient Approach to News Recommendation with Keras 3 and JAX},
      author={Igor L. R. Azevedo and Toyotaro Suzumura and Yuichiro Yasui},
      year={2025},
      eprint={2508.21572},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
      url={https://arxiv.org/abs/2508.21572},
}
```
