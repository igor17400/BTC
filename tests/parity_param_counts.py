"""Phase-1 parity diagnostic: per-module trainable-param counts in JAX vs PyTorch.

Catches gross structural drift between the two framework ports of each
model (extra projection, wrong head/dim, missing component).
Runs in seconds with tiny dims; no real data needed.

Usage:
    python tests/parity_param_counts.py            # all models
    python tests/parity_param_counts.py nrms naml  # subset
"""

from __future__ import annotations

import sys
from collections import defaultdict
from typing import Any

import numpy as np

# Small, identical fixture for both frameworks.
SMALL = dict(
    vocab_size=200,
    embedding_size=64,
    num_categories=4,
    num_subcategories=8,
    num_users=20,
    num_entities=10,
    entity_emb_dim=32,
)


def _processed_news(use_entity: bool = False) -> dict[str, Any]:
    rng = np.random.default_rng(0)
    V, E = SMALL["vocab_size"], SMALL["embedding_size"]
    pn: dict[str, Any] = {
        "vocab_size": V,
        "embeddings": rng.standard_normal((V, E)).astype(np.float32),
        "num_categories": SMALL["num_categories"],
        "num_subcategories": SMALL["num_subcategories"],
        "num_users": SMALL["num_users"],
        "tokens": rng.integers(0, V, (50, 32), dtype=np.int32),
        "abstract_tokens": rng.integers(0, V, (50, 50), dtype=np.int32),
        "category_indices": rng.integers(
            0, SMALL["num_categories"], 50, dtype=np.int32
        ),
        "subcategory_indices": rng.integers(
            0, SMALL["num_subcategories"], 50, dtype=np.int32
        ),
        "news_ids_original_strings": [f"N{i}" for i in range(50)],
    }
    if use_entity:
        pn["entity_vocab_size"] = SMALL["num_entities"]
        pn["entity_embeddings"] = rng.standard_normal(
            (SMALL["num_entities"], SMALL["entity_emb_dim"])
        ).astype(np.float32)
        pn["entity_indices"] = rng.integers(
            0, SMALL["num_entities"], (50, 5), dtype=np.int32
        )
    return pn


def _shared_config(name: str) -> dict[str, Any]:
    """Minimal kwargs to make each model build with small dims.

    All values picked to match defaults in src/core/models/configs.py
    where divisibility / multiplicity constraints matter.
    """
    base = dict(
        embedding_size=SMALL["embedding_size"],
        dropout_rate=0.0,
        seed=0,
        max_title_length=32,
        max_history_length=10,
        max_impressions_length=5,
    )
    if name == "nrms":
        return {**base, "multiheads": 4, "head_dim": 16, "attention_hidden_dim": 64}
    if name == "naml":
        return {
            **base,
            "max_abstract_length": 50,
            "category_embedding_dim": 16,
            "subcategory_embedding_dim": 16,
            "cnn_filter_num": 64,
            "cnn_kernel_size": 3,
            "word_attention_query_dim": 64,
            "view_attention_query_dim": 64,
            "user_attention_query_dim": 64,
        }
    if name == "lstur":
        return {
            **base,
            "cnn_filter_num": 64,
            "cnn_kernel_size": 3,
            "cnn_activation": "relu",
            "attention_hidden_dim": 64,
            "gru_unit": 64,  # equal to news encoder output_dim when no cat
            "type": "ini",
            "use_category": False,
            "use_subcategory": False,
        }
    if name == "crown":
        return {
            **base,
            "max_abstract_length": 50,
            "category_embedding_dim": 16,
            "subcategory_embedding_dim": 16,
            "num_heads": 4,
            "head_dim": 16,
            "feedforward_dim": 64,
            "num_layers": 1,
            "intent_num": 3,
            "intent_embedding_dim": 64,
            "attention_dim": 64,
            "alpha": 0.3,
            "gnn_type": "gat",
            "graph_num_layers": 1,
            "graph_hidden_dim": 64,
            "gat_num_heads": 2,
            "gat_alpha": 0.2,
            "gat_concat_heads": True,
            "sage_aggregator": "mean",
            "sage_normalize": True,
            "user_attention_dim": 64,
        }
    if name == "caum":
        return {
            **base,
            "news_num_heads": 4,
            "news_head_dim": 16,
            "news_attention_hidden_dim": 64,
            "entity_embedding_dim": SMALL["entity_emb_dim"],
            "entity_num_heads": 2,
            "entity_head_dim": 16,
            "category_embedding_dim": 16,
            "use_entity": True,
            "use_category": True,
            "candi_selfatt_num_heads": 4,
            "candi_selfatt_head_dim": 16,
            "candi_cnn_half_window": 1,
            "candi_att_hidden_dim": 64,
            "candi_att_mid_dim": 32,
            "news_dim": 64,
            "max_entities": 5,
        }
    if name == "pprec":
        return {
            **base,
            "news_dim": 64,
            "entity_embedding_dim": SMALL["entity_emb_dim"],
            "category_embedding_dim": 32,
            "num_heads": 4,
            "head_dim": 16,
            "co_num_heads": 2,
            "co_head_dim": 16,
            "attention_hidden_dim": 64,
            "popularity_embedding_bins": 200,
            "popularity_embedding_dim": 64,
            "recency_embedding_bins": 1500,
            "recency_embedding_dim": 16,
            "ctr_scaler_init": 19.0,
            "pop_content_dims": (64, 64, 32),
            "pop_recency_dims": (16, 16),
            "pop_gate_dims": (32, 16),
            "activity_gate_hidden_dim": 16,
            "use_entity": True,
            "use_recency": True,
            "use_ctr": True,
            "use_activity_gate": True,
            "max_entities": 5,
        }
    if name == "tccm":
        return {
            **base,
            "news_dim": 64,
            "entity_embedding_dim": SMALL["entity_emb_dim"],
            "category_embedding_dim": 32,
            "num_heads": 4,
            "head_dim": 16,
            "co_num_heads": 2,
            "co_head_dim": 16,
            "attention_hidden_dim": 64,
            "popularity_embedding_bins": 200,
            "popularity_embedding_dim": 64,
            "pop_token_embedding_bins": 200,
            "pop_token_embedding_dim": 32,
            "pop_num_heads": 1,
            "pop_head_dim": 64,
            "pop_content_dims": (64, 64, 32),
            "timeliness_embedding_bins": 505,
            "timeliness_embedding_dim": 16,
            "pop_recency_dims": (16, 16),
            "timeliness_lambda": 2.0,
            "activity_gate_dims": (32, 16),
            "use_causal_intervention": False,
            "intervention_value": 0.5,
            "use_entity": True,
            "use_activity_gate": True,
            "max_entities": 5,
        }
    if name == "digat":
        return {
            **base,
            "msa_head_num": 4,
            "msa_head_dim": 16,
            "attention_dim": 64,
            "graph_depth": 1,
            "sag_hops": 1,
            "sag_neighbors": 2,
        }
    if name == "glory":
        return dict(
            word_emb_dim=SMALL["embedding_size"],
            head_num=4,
            head_dim=16,
            attention_hidden_dim=64,
            dropout_rate=0.0,
            title_size=30,
            entity_size=5,
            max_history_length=10,
            max_impressions_length=5,
            use_graph_type=0,
            directed=True,
            gnn_num_layers=1,
            k_hops=1,
            num_neighbors=2,
            use_entity=False,
        )
    raise ValueError(name)


def _build_jax(name: str, pn: dict[str, Any], cfg: dict[str, Any]):
    from flax import nnx

    from src.core.models.spec import get_model_class

    cls = get_model_class(name, "jax")
    extras = {"rngs": nnx.Rngs(0)}
    if name == "lstur":
        extras["num_users"] = SMALL["num_users"]
    return cls(processed_news=pn, **cfg, **extras)


def _build_pt(name: str, pn: dict[str, Any], cfg: dict[str, Any]):
    from src.core.models.spec import get_model_class

    cls = get_model_class(name, "pytorch")
    extras: dict[str, Any] = {}
    if name == "lstur":
        extras["num_users"] = SMALL["num_users"]
    return cls(processed_news=pn, **cfg, **extras)


def _jax_counts_by_module(model) -> tuple[dict[str, int], int]:
    """Group JAX/NNX trainable-param counts by top-level attribute name."""
    import jax
    import jax.numpy as jnp
    from flax import nnx

    state = nnx.state(model)
    by_module: dict[str, int] = defaultdict(int)
    total = 0
    for key_path, leaf in jax.tree_util.tree_leaves_with_path(state):
        arr = getattr(leaf, "value", leaf)
        if not hasattr(arr, "shape") or not hasattr(arr, "dtype"):
            continue
        if jnp.issubdtype(arr.dtype, jax.dtypes.prng_key):
            continue
        n = int(np.prod(arr.shape))
        # First DictKey is the top-level module name (e.g. "news_encoder").
        first = None
        for k in key_path:
            s = str(k)
            # JAX path entries look like "['name']" — strip quotes & brackets
            s = s.strip("[].").strip("'\"")
            if s and s not in ("value",):
                first = s
                break
        bucket = first or "<root>"
        by_module[bucket] += n
        total += n
    return dict(by_module), total


def _pt_counts_by_module(model) -> tuple[dict[str, int], int]:
    by_module: dict[str, int] = defaultdict(int)
    total = 0
    for name, p in model.named_parameters():
        n = p.numel()
        bucket = name.split(".", 1)[0]
        by_module[bucket] += n
        total += n
    return dict(by_module), total


def _print_table(name: str, jax_buckets: dict[str, int], pt_buckets: dict[str, int]):
    """Side-by-side render. Mismatch markers: ✓ equal, ! diff, J/P = only-in-one."""
    keys = sorted(set(jax_buckets) | set(pt_buckets))
    pt_total = sum(pt_buckets.values())
    jax_total = sum(jax_buckets.values())
    delta = jax_total - pt_total
    pct = 100.0 * delta / pt_total if pt_total else 0.0
    print()
    print(f"=== {name.upper()} ===")
    print(f"{'module':<24s}  {'JAX':>12s}  {'PyTorch':>12s}  {'Δ':>10s}  {'flag':>6s}")
    print("-" * 70)
    for k in keys:
        j = jax_buckets.get(k, 0)
        p = pt_buckets.get(k, 0)
        if j == p:
            flag = "✓"
        elif j == 0:
            flag = "PT-only"
        elif p == 0:
            flag = "JAX-only"
        else:
            flag = "MISMATCH"
        d = j - p
        print(f"{k:<24s}  {j:>12,d}  {p:>12,d}  {d:>+10,d}  {flag:>6s}")
    print("-" * 70)
    summary_flag = "✓ MATCH" if delta == 0 else f"DRIFT {pct:+.2f}%"
    print(
        f"{'TOTAL':<24s}  {jax_total:>12,d}  {pt_total:>12,d}  {delta:>+10,d}  {summary_flag}"
    )


_ALL = ["nrms", "naml", "lstur", "pprec", "crown", "caum", "tccm", "digat", "glory"]
_NEED_ENTITY = {"pprec", "tccm", "caum"}


def main(argv: list[str] | None = None) -> int:
    argv = argv or sys.argv[1:]
    names = argv or _ALL
    rc = 0
    for name in names:
        try:
            pn = _processed_news(use_entity=name in _NEED_ENTITY)
            cfg = _shared_config(name)
            jax_model = _build_jax(name, pn, cfg)
            pt_model = _build_pt(name, pn, cfg)
            j_b, _ = _jax_counts_by_module(jax_model)
            p_b, _ = _pt_counts_by_module(pt_model)
            _print_table(name, j_b, p_b)
            if sum(j_b.values()) != sum(p_b.values()):
                rc = 1
        except Exception as e:
            import traceback

            print(f"\n=== {name.upper()} ===")
            print(f"[ERROR] {type(e).__name__}: {e}")
            traceback.print_exc(limit=2)
            rc = 2
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
