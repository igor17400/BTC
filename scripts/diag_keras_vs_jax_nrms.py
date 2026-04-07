"""Diagnostic: compare Keras NRMS vs JAX NRMS layer-by-layer outputs.

Both frameworks use the same underlying JAX runtime, so any difference in
output statistics points to a bug in the model implementation (Keras layers
vs Flax NNX layers) rather than the runtime.

We use:
- Real GloVe embeddings from processed_news (identical across frameworks)
- A fixed random integer batch as input
- seed=42 for any other random init

We compare per-layer output statistics (shape, mean, std, norm) of the
news encoder. If the embedding output matches but a later layer diverges,
that pinpoints the buggy component.
"""

from __future__ import annotations

import os

# Force JAX backend for Keras so both frameworks share the JAX runtime
os.environ["KERAS_BACKEND"] = "jax"

import pickle
from pathlib import Path

import numpy as np


def load_processed_news() -> dict:
    """Load the cached processed_news for MIND-small."""
    cache_dir = Path("/home/igor/NewsReX/.data/mind-small/small/processed")
    pkl = cache_dir / "processed_news_thresh5_english.pkl"
    npy = cache_dir / "filtered_embeddings_thresh5_english.npy"
    with open(pkl, "rb") as f:
        pn = pickle.load(f)
    pn["embeddings"] = np.load(npy)
    return pn


def stats(name: str, x) -> None:
    """Print shape + mean + std + norm of an array-like."""
    arr = np.asarray(x)
    print(
        f"  {name:32s} shape={str(arr.shape):20s} "
        f"mean={arr.mean():+.6f}  std={arr.std():.6f}  "
        f"norm={np.linalg.norm(arr):.4f}"
    )


def run_keras() -> dict:
    """Run a forward pass through Keras NRMS news_encoder."""
    import keras
    from keras import ops

    from src.frameworks.keras.models.nrms import NRMS

    pn = load_processed_news()
    keras.utils.set_random_seed(42)

    model = NRMS(
        processed_news=pn,
        embedding_size=300,
        multiheads=20,
        head_dim=15,
        attention_hidden_dim=200,
        dropout_rate=0.2,
        seed=42,
        max_title_length=32,
        max_history_length=50,
        max_impressions_length=5,
    )

    enc = model.news_encoder

    # Build a deterministic input
    rng = np.random.default_rng(42)
    fake = rng.integers(1, pn["vocab_size"], size=(4, 32)).astype(np.int32)
    fake_t = ops.convert_to_tensor(fake)

    out = {}
    out["input"] = fake

    # Walk the layers manually so we can capture intermediate stats
    embedded = enc.embedding_layer(fake_t)
    out["after_embedding"] = ops.convert_to_numpy(embedded)

    y = enc.dropout1(embedded, training=False)
    out["after_dropout1"] = ops.convert_to_numpy(y)

    pad_mask = ops.not_equal(fake_t, 0)
    y_mha = enc.multi_head_attention(
        y, y, y, key_mask=pad_mask, value_mask=pad_mask, training=False
    )
    out["after_mha"] = ops.convert_to_numpy(y_mha)

    y_drop = enc.dropout2(y_mha, training=False)
    out["after_dropout2"] = ops.convert_to_numpy(y_drop)

    final = enc.additive_attention(y_drop, mask=pad_mask)
    out["after_additive"] = ops.convert_to_numpy(final)

    # Also full forward
    full = enc(fake_t, training=False)
    out["full_forward"] = ops.convert_to_numpy(full)

    return out


def run_jax() -> dict:
    """Run a forward pass through JAX NRMS news_encoder."""
    import jax
    import jax.numpy as jnp
    from flax import nnx

    from src.frameworks.jax.models.nrms import NRMS

    pn = load_processed_news()
    rngs = nnx.Rngs(42)

    model = NRMS(
        processed_news=pn,
        embedding_size=300,
        multiheads=20,
        head_dim=15,
        attention_hidden_dim=200,
        dropout_rate=0.2,
        seed=42,
        max_title_length=32,
        max_history_length=50,
        max_impressions_length=5,
        rngs=rngs,
    )

    enc = model.news_encoder

    rng = np.random.default_rng(42)
    fake = rng.integers(1, pn["vocab_size"], size=(4, 32)).astype(np.int32)
    fake_t = jnp.asarray(fake)

    out = {}
    out["input"] = fake

    embedded = enc.embedding_layer(fake_t)
    out["after_embedding"] = np.asarray(embedded)

    y = enc.dropout1(embedded, deterministic=True)
    out["after_dropout1"] = np.asarray(y)

    pad_mask = jnp.not_equal(fake_t, 0)
    attn_mask = pad_mask[:, None, None, :]
    y_mha = enc.multi_head_attention(y, y, mask=attn_mask, deterministic=True)
    out["after_mha"] = np.asarray(y_mha)

    y_drop = enc.dropout2(y_mha, deterministic=True)
    out["after_dropout2"] = np.asarray(y_drop)

    final = enc.additive_attention(y_drop, mask=pad_mask)
    out["after_additive"] = np.asarray(final)

    full = enc(fake_t, training=False)
    out["full_forward"] = np.asarray(full)

    return out


def main() -> None:
    print("=" * 80)
    print("Keras NRMS news_encoder forward pass")
    print("=" * 80)
    keras_out = run_keras()
    for name, val in keras_out.items():
        if name == "input":
            continue
        stats(name, val)

    print()
    print("=" * 80)
    print("JAX NRMS news_encoder forward pass")
    print("=" * 80)
    jax_out = run_jax()
    for name, val in jax_out.items():
        if name == "input":
            continue
        stats(name, val)

    # Compare matching layer outputs
    print()
    print("=" * 80)
    print("Side-by-side comparison")
    print("=" * 80)
    for layer in [
        "after_embedding",
        "after_dropout1",
        "after_mha",
        "after_dropout2",
        "after_additive",
        "full_forward",
    ]:
        k = np.asarray(keras_out[layer])
        j = np.asarray(jax_out[layer])
        diff = np.abs(k - j) if k.shape == j.shape else None
        kn = np.linalg.norm(k)
        jn = np.linalg.norm(j)
        ratio = jn / max(kn, 1e-12)
        print(
            f"  {layer:20s}  keras_norm={kn:10.4f}  jax_norm={jn:10.4f}  "
            f"ratio={ratio:.4f}",
            end="",
        )
        if diff is not None:
            print(f"  max_abs_diff={diff.max():.6f}")
        else:
            print(f"  shapes_differ keras={k.shape} jax={j.shape}")


if __name__ == "__main__":
    main()
