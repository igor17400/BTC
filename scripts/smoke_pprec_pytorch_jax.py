"""Smoke test: instantiate the PyTorch and JAX PP-Rec models and run
a synthetic forward pass + backward pass. Verifies the ports compile.

Doesn't need GPU — just checks code paths.
"""

from __future__ import annotations

import os
# Make sure we don't accidentally pick up an old keras backend env
os.environ.setdefault("KERAS_BACKEND", "jax")

import pickle
from pathlib import Path

import numpy as np

CACHE = Path("/home/igor/NewsReX/.data/mind-small/small/processed")


def load_processed_news() -> dict:
    """Load the same processed_news dict the runner builds."""
    pkl = CACHE / "processed_news_thresh5_english.pkl"
    npy = CACHE / "filtered_embeddings_thresh5_english.npy"
    with open(pkl, "rb") as f:
        pn = pickle.load(f)
    pn["embeddings"] = np.load(npy)
    # Add the entity + popularity data PP-Rec needs
    entity_pkl = CACHE / "entity_data_max5.pkl"
    if entity_pkl.exists():
        with open(entity_pkl, "rb") as f:
            ent = pickle.load(f)
        pn.update(ent)
    return pn


def make_fake_batch(pn: dict, B: int = 4, H: int = 50, C: int = 5,
                    title_len: int = 32, max_entities: int = 5):
    """Synthetic mini-batch with the same shape the model expects."""
    rng = np.random.default_rng(42)
    vocab = pn["vocab_size"]
    n_entities = pn.get("entity_vocab_size", 100)
    n_cats = pn.get("num_categories", 18)

    def make_feature_row(rows, n_token_max, n_entity_max, n_cat_max):
        # Concatenate [title_tokens, entity_indices, category_index]
        title = rng.integers(1, n_token_max, size=rows + (title_len,)).astype(np.int32)
        ent = rng.integers(0, n_entity_max, size=rows + (max_entities,)).astype(np.int32)
        cat = rng.integers(0, n_cat_max + 1, size=rows + (1,)).astype(np.int32)
        return np.concatenate([title, ent, cat], axis=-1)

    return {
        "hist_tokens": make_feature_row((B, H), vocab, n_entities, n_cats),
        "cand_tokens": make_feature_row((B, C), vocab, n_entities, n_cats),
        "hist_ctr": rng.integers(0, 200, size=(B, H)).astype(np.int32),
        "cand_ctr": rng.uniform(0, 1, size=(B, C)).astype(np.float32),
        "cand_recency": rng.integers(0, 100, size=(B, C)).astype(np.int32),
    }, rng.uniform(0, 1, size=(B, C)).astype(np.float32)


def smoke_pytorch():
    print("=" * 60)
    print("PyTorch PP-Rec smoke test")
    print("=" * 60)
    import torch
    from src.frameworks.pytorch.models.pprec import PPRec
    from src.core.models.configs import PPRecConfig

    pn = load_processed_news()
    cfg = PPRecConfig(
        embedding_size=300,
        max_title_length=32,
        max_history_length=50,
        max_impressions_length=5,
        max_entities=5,
    )
    model = PPRec(processed_news=pn, config=cfg)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    batch, labels = make_fake_batch(pn)
    batch = {k: torch.tensor(v) for k, v in batch.items()}
    labels_t = torch.tensor(labels)

    # Forward (training)
    model.train()
    out_train = model(batch, training=True)
    print(f"Training output shape: {tuple(out_train.shape)}")
    print(f"  mean={out_train.mean().item():.4f}  std={out_train.std().item():.4f}")
    assert out_train.shape == (4, 5), f"Expected (4, 5), got {tuple(out_train.shape)}"

    # Backward
    loss = torch.nn.functional.cross_entropy(out_train, labels_t.argmax(dim=-1))
    loss.backward()
    print(f"Backward OK. Loss: {loss.item():.4f}")

    # Forward (inference)
    model.eval()
    out_eval = model(batch, training=False)
    print(f"Inference output shape: {tuple(out_eval.shape)}")
    print(f"  mean={out_eval.mean().item():.4f}  std={out_eval.std().item():.4f}")
    assert (out_eval >= 0).all() and (out_eval <= 1).all(), "Eval output should be in [0,1] (sigmoid)"

    print("✓ PyTorch PP-Rec smoke test passed.\n")


def smoke_jax():
    print("=" * 60)
    print("JAX PP-Rec smoke test")
    print("=" * 60)
    import jax
    import jax.numpy as jnp
    from flax import nnx
    from src.frameworks.jax.models.pprec import PPRec
    from src.core.models.configs import PPRecConfig

    pn = load_processed_news()
    cfg = PPRecConfig(
        embedding_size=300,
        max_title_length=32,
        max_history_length=50,
        max_impressions_length=5,
        max_entities=5,
    )
    rngs = nnx.Rngs(42)
    model = PPRec(processed_news=pn, config=cfg, rngs=rngs)

    # Count params via the state
    state = nnx.state(model, nnx.Param)
    n_params = sum(int(np.prod(p.value.shape)) for p in jax.tree_util.tree_leaves(state) if hasattr(p, "value"))
    print(f"Model params (approx): {n_params:,}")

    batch, labels = make_fake_batch(pn)
    batch = {k: jnp.asarray(v) for k, v in batch.items()}

    # Forward (training)
    out_train = model(batch, training=True)
    print(f"Training output shape: {tuple(out_train.shape)}")
    print(f"  mean={float(out_train.mean()):.4f}  std={float(out_train.std()):.4f}")
    assert out_train.shape == (4, 5), f"Expected (4, 5), got {tuple(out_train.shape)}"

    # Forward (inference)
    out_eval = model(batch, training=False)
    print(f"Inference output shape: {tuple(out_eval.shape)}")
    print(f"  mean={float(out_eval.mean()):.4f}  std={float(out_eval.std()):.4f}")
    assert ((out_eval >= 0) & (out_eval <= 1)).all(), "Eval output should be in [0,1]"

    print("✓ JAX PP-Rec smoke test passed.\n")


if __name__ == "__main__":
    smoke_pytorch()
    smoke_jax()
    print("All smoke tests passed.")
