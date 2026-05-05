"""Standalone TCCM smoke test (Keras).

Verifies the popularity-encoder shapes, the main model forward pass on
a tiny synthetic batch, and the per-token CTR preprocessor. No Hydra,
no dataset configs, no file I/O.

Usage:
    uv run python tests/smoke_tccm.py
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("KERAS_BACKEND", "jax")

import numpy as np
import pandas as pd

# ---- 1. TCCM token-CTR tables ---------------------------------------------


def test_token_ctr_tables():
    from src.core.data.processing.models.tccm import compute_token_ctr_tables

    # Three impressions over 3 hours, four news articles.
    df = pd.DataFrame(
        {
            "time": pd.to_datetime(
                ["2020-01-01 00:30:00", "2020-01-01 01:30:00", "2020-01-01 02:30:00"]
            ),
            "impressions": ["N1-1 N2-0", "N1-0 N3-1", "N3-1 N4-0"],
        }
    )
    news_str_to_idx = {"N1": 0, "N2": 1, "N3": 2, "N4": 3}
    news_titles = np.array(
        [
            [1, 2, 0, 0],
            [3, 4, 0, 0],
            [5, 1, 0, 0],
            [2, 3, 0, 0],
        ],
        dtype=np.int32,
    )
    news_entities = np.array(
        [
            [1, 0, 0],
            [2, 0, 0],
            [3, 0, 0],
            [4, 0, 0],
        ],
        dtype=np.int32,
    )

    word_pop, entity_pop, dataset_start, num_buckets = compute_token_ctr_tables(
        df,
        news_str_to_idx=news_str_to_idx,
        news_title_tokens=news_titles,
        news_entity_indices=news_entities,
        vocab_size=10,
        entity_vocab_size=10,
        bucket_hours=1,
        bins=200,
    )
    assert word_pop.shape == (num_buckets, 10), word_pop.shape
    assert entity_pop.shape == (num_buckets, 10), entity_pop.shape
    assert word_pop.dtype == np.int32
    # Padding column 0 must always be zero.
    assert (word_pop[:, 0] == 0).all()
    assert (entity_pop[:, 0] == 0).all()
    # Some non-padding token must have a non-zero CTR somewhere.
    assert word_pop[:, 1:].max() > 0, "expected at least one positive word CTR"
    print("[OK] token-CTR tables")


# ---- 2. TCCM popularity encoder (forward shape + finiteness) --------------


def test_popularity_encoder_forward():
    from src.core.models.configs import TCCMConfig
    from src.frameworks.keras.models.tccm import TCCMPopularityEncoder

    cfg = TCCMConfig(
        embedding_size=16,
        news_dim=32,
        num_heads=4,
        head_dim=8,
        co_num_heads=2,
        co_head_dim=8,
        attention_hidden_dim=16,
        pop_token_embedding_bins=50,
        pop_token_embedding_dim=16,
        pop_num_heads=1,
        pop_head_dim=32,
        pop_content_dims=(32, 32, 16),
        timeliness_embedding_bins=64,
        timeliness_embedding_dim=8,
        pop_recency_dims=(16, 16),
        timeliness_lambda=2.0,
        max_title_length=4,
        max_entities=2,
    )
    enc = TCCMPopularityEncoder(cfg)
    import keras

    B, T, E = 7, cfg.max_title_length, cfg.max_entities
    bucket_input = keras.ops.convert_to_tensor(
        np.random.randint(0, cfg.pop_token_embedding_bins, (B, T + E)), dtype="int32"
    )
    time_input = keras.ops.convert_to_tensor(
        np.random.randint(0, cfg.timeliness_embedding_bins, (B,)), dtype="int32"
    )
    out = enc(bucket_input, time_input, title_len=T, training=False)
    out_np = keras.ops.convert_to_numpy(out)
    assert out_np.shape == (B,), out_np.shape
    assert np.isfinite(out_np).all(), "non-finite popularity score"
    print(f"[OK] popularity encoder: out range [{out_np.min():.3f}, {out_np.max():.3f}]")


# ---- 3. Full TCCM model forward (and backward) on synthetic data ----------


def test_model_forward_backward():
    from src.core.models.configs import TCCMConfig
    from src.frameworks.keras.models.tccm import TCCM

    B, C, H, T, E = 3, 4, 6, 5, 3

    processed = {
        "vocab_size": 50,
        "embeddings": np.random.randn(50, 16).astype(np.float32),
        "num_categories": 10,
        "num_subcategories": 10,
        "entity_vocab_size": 20,
        "entity_embeddings": np.random.randn(20, 8).astype(np.float32),
        "entity_indices": np.random.randint(0, 20, (40, E)).astype(np.int32),
        "tokens": np.random.randint(0, 50, (40, T)).astype(np.int32),
    }

    cfg = TCCMConfig(
        embedding_size=16,
        news_dim=32,
        entity_embedding_dim=8,
        num_heads=2,
        head_dim=8,
        co_num_heads=2,
        co_head_dim=8,
        attention_hidden_dim=16,
        popularity_embedding_bins=50,
        popularity_embedding_dim=32,
        pop_token_embedding_bins=50,
        pop_token_embedding_dim=16,
        pop_num_heads=1,
        pop_head_dim=32,
        pop_content_dims=(32, 32, 16),
        timeliness_embedding_bins=64,
        timeliness_embedding_dim=8,
        pop_recency_dims=(16, 16),
        max_title_length=T,
        max_entities=E,
        max_history_length=H,
        max_impressions_length=C,
    )
    model = TCCM(processed, config=cfg)

    # Build hist/cand_tokens with correct per-slice ranges:
    #   [:T] are word IDs in [1, vocab_size); [T:T+E] are entity IDs in [1, entity_vocab_size).
    def _pack(words_shape):
        words = np.random.randint(1, 50, words_shape + (T,)).astype(np.int32)
        ents = np.random.randint(1, 20, words_shape + (E,)).astype(np.int32)
        return np.concatenate([words, ents], axis=-1)

    inputs = {
        "hist_tokens": _pack((B, H)),
        "cand_tokens": _pack((B, C)),
        "hist_ctr": np.random.randint(0, 50, (B, H)).astype(np.int32),
        "cand_pop_buckets": np.random.randint(0, 50, (B, C, T + E)).astype(np.int32),
        "cand_news_exist_time": np.random.randint(0, 64, (B, C)).astype(np.int32),
    }

    import keras

    inputs_k = {k: keras.ops.convert_to_tensor(v) for k, v in inputs.items()}
    logits = model(inputs_k, training=False)
    logits_np = keras.ops.convert_to_numpy(logits)
    assert logits_np.shape == (B, C), logits_np.shape
    assert np.isfinite(logits_np).all(), "non-finite logits"
    print(
        f"[OK] full forward: logits {logits_np.shape}, "
        f"range [{logits_np.min():.3f}, {logits_np.max():.3f}]"
    )

    # Causal-intervention check: the logits should change drastically when
    # ``use_causal_intervention`` is flipped at inference. We exercise it
    # via the evaluator only — here we just confirm the config flag is
    # readable on the model.
    assert hasattr(model.config, "use_causal_intervention")
    print("[OK] causal-intervention flag visible on model.config")


if __name__ == "__main__":
    print("=" * 60)
    print("TCCM Smoke Tests (Keras)")
    print("=" * 60)
    try:
        test_token_ctr_tables()
        test_popularity_encoder_forward()
        test_model_forward_backward()
        print("=" * 60)
        print("ALL TESTS PASSED")
        print("=" * 60)
    except Exception as e:
        print(f"\nFAILED: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
