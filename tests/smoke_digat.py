"""Standalone DIGAT smoke test.

Verifies the full pipeline: model instantiation, training forward/backward,
scatter ops, SAG BFS construction, and the custom DIGAT evaluator — all
with tiny synthetic data. No Hydra, no dataset configs, no file I/O.

Usage:
    uv run python tests/smoke_digat.py
"""

import sys
import numpy as np
import torch

# ---------- 1. Scatter operations ----------

def test_scatter_ops():
    from src.frameworks.pytorch.models.digat import scatter_softmax, scatter_sum

    src = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    idx = torch.tensor([[0, 0, 1, 1]])

    ss = scatter_softmax(src, idx, 2)
    assert ss.shape == (1, 4), f"scatter_softmax shape: {ss.shape}"
    # Group 0: softmax([1,2]) ≈ [0.269, 0.731]
    assert abs(ss[0, 0].item() - 0.2689) < 0.01, f"scatter_softmax val: {ss}"
    assert abs(ss[0, 1].item() - 0.7311) < 0.01

    vals = torch.tensor([[[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]]])
    ssum = scatter_sum(vals, idx, dim=1, dim_size=2)
    assert ssum.shape == (1, 2, 2)
    assert abs(ssum[0, 0, 0].item() - 3.0) < 0.01  # group 0: 1+2=3
    assert abs(ssum[0, 1, 0].item() - 7.0) < 0.01  # group 1: 3+4=7
    print("[OK] scatter ops")


# ---------- 2. SAG BFS construction ----------

def test_sag_bfs():
    from src.core.data.processing.sag import _build_news_graphs_bfs

    N = 20
    top_k = 5
    sim_indices = np.random.randint(0, N, (N, top_k))
    sim_values = np.random.uniform(0.6, 1.0, (N, top_k)).astype(np.float32)

    node_ids, graph, mask = _build_news_graphs_bfs(
        sim_indices, sim_values, num_news=N,
        news_graph_size=26, sag_hops=2, sag_neighbors=5,
    )
    assert node_ids.shape == (N, 26), f"node_ids shape: {node_ids.shape}"
    assert graph.shape == (N, 26, 26), f"graph shape: {graph.shape}"
    assert mask.shape == (N, 26), f"mask shape: {mask.shape}"
    assert mask[:, 0].all(), "center node should always be valid"
    print("[OK] SAG BFS construction")


# ---------- 3. Model forward + backward ----------

def test_model_forward_backward():
    from src.frameworks.pytorch.models.digat import DIGAT

    B, C, H, T = 4, 5, 50, 32
    num_news = 100
    num_categories = 10 + 1  # +1 padding

    processed = {
        "vocab_size": 200,
        "embeddings": np.random.randn(200, 300).astype(np.float32),
        "num_categories": 10,
    }
    model = DIGAT(processed)
    G_n = model.news_graph_size
    G_u = H + num_categories

    inputs = {
        "hist_tokens": torch.randint(1, 200, (B, H, T)),
        "hist_mask": torch.ones(B, H, T),
        "user_graph": torch.eye(G_u).unsqueeze(0).expand(B, -1, -1).float(),
        "user_category_mask": torch.ones(B, num_categories),
        "user_category_indices": torch.randint(0, num_categories, (B, H)),
        "cand_tokens": torch.randint(1, 200, (B, C, G_n, T)),
        "cand_mask": torch.ones(B, C, G_n, T),
        "cand_graph": torch.eye(G_n).unsqueeze(0).unsqueeze(0).expand(B, C, -1, -1).float(),
        "cand_graph_mask": torch.ones(B, C, G_n),
    }

    # Forward
    logits = model(inputs, training=True)
    assert logits.shape == (B, C), f"logits shape: {logits.shape}"
    assert not torch.isnan(logits).any(), "NaN in logits"
    print(f"  forward OK: logits {logits.shape}, range [{logits.min():.3f}, {logits.max():.3f}]")

    # Backward
    labels = torch.zeros(B, C)
    labels[:, 0] = 1.0  # first candidate is positive
    loss = torch.nn.functional.cross_entropy(logits, labels)
    loss.backward()
    assert not torch.isnan(loss), "NaN loss"

    # Check gradients exist
    grad_count = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    total = sum(1 for p in model.parameters())
    print(f"  backward OK: loss={loss.item():.4f}, {grad_count}/{total} params have non-zero grad")
    print("[OK] model forward + backward")


# ---------- 4. User graph construction ----------

def test_user_graph_construction():
    from src.core.data.processing.digat import build_user_graphs

    N, H = 8, 50
    num_categories = 11  # 10 + 1 padding
    hist_cats = np.random.randint(0, 10, (N, H)).astype(np.int32)
    hist_cats[:, -5:] = 0  # last 5 are padding

    result = build_user_graphs(hist_cats, H, num_categories)
    assert result["user_graph"].shape == (N, H + num_categories, H + num_categories)
    assert result["user_category_mask"].shape == (N, num_categories)
    assert result["user_category_indices"].shape == (N, H)

    # Diagonal should be True (self-loops)
    for b in range(N):
        assert np.all(np.diag(result["user_graph"][b])), f"missing self-loops at b={b}"
    print("[OK] user graph construction")


# ---------- 5. DIGAT evaluator (minimal) ----------

def test_evaluator():
    from src.frameworks.pytorch.models.digat import DIGAT
    from src.core.data.processing.digat import build_user_graphs

    B, H, T = 4, 50, 32
    num_news = 50
    num_categories = 11

    processed = {
        "vocab_size": 200,
        "embeddings": np.random.randn(200, 300).astype(np.float32),
        "num_categories": 10,
        "tokens": np.random.randint(1, 200, (num_news, T)).astype(np.int32),
        "category_indices": np.random.randint(0, 10, num_news).astype(np.int32),
    }

    model = DIGAT(processed)
    model.eval()
    G_n = model.news_graph_size
    D = model.D

    # Dummy SAG
    sag_data = {
        "news_node_ID": np.random.randint(0, num_news, (num_news, G_n)).astype(np.int32),
        "news_graph": np.eye(G_n, dtype=np.bool_)[None].repeat(num_news, axis=0),
        "news_graph_mask": np.ones((num_news, G_n), dtype=np.bool_),
    }

    # Pre-compute news embeddings (same as evaluator step 1)
    with torch.no_grad():
        tokens_t = torch.tensor(processed["tokens"], dtype=torch.long)
        emb = model.news_encoder(tokens_t.unsqueeze(1), (tokens_t != 0).float().unsqueeze(1))
        all_news_emb = emb.squeeze(1).numpy()

    # Pre-compute SAG contexts (step 2)
    with torch.no_grad():
        node_ids = sag_data["news_node_ID"][:5]
        sag_emb = torch.tensor(all_news_emb[node_ids], dtype=torch.float32)
        sag_mask = torch.tensor(sag_data["news_graph_mask"][:5], dtype=torch.float32)
        ctx = model.graph_encoder._news_graph_context(sag_emb, sag_mask)
        assert ctx.shape == (5, D), f"news ctx shape: {ctx.shape}"
        assert not torch.isnan(ctx).any(), "NaN in news ctx"

    # Run dual graph interaction for one "impression" (step 3)
    with torch.no_grad():
        C = 3
        cand_ids = np.array([1, 2, 3])
        cand_node_ids = sag_data["news_node_ID"][cand_ids]
        cand_sag_emb = torch.tensor(all_news_emb[cand_node_ids], dtype=torch.float32)
        cand_graph = torch.tensor(sag_data["news_graph"][cand_ids].astype(np.float32))
        cand_mask = torch.tensor(sag_data["news_graph_mask"][cand_ids].astype(np.float32))

        hist_ids = np.random.randint(0, num_news, H)
        user_hist_emb = torch.tensor(all_news_emb[hist_ids], dtype=torch.float32).unsqueeze(0).expand(C, -1, -1)

        hist_cats = np.random.randint(0, 10, (1, H)).astype(np.int32)
        ug = build_user_graphs(hist_cats, H, num_categories)
        u_graph = torch.tensor(ug["user_graph"][0], dtype=torch.float32).unsqueeze(0).expand(C, -1, -1)
        u_cat_mask = torch.tensor(ug["user_category_mask"][0], dtype=torch.float32).unsqueeze(0).expand(C, -1)
        u_cat_indices = torch.tensor(ug["user_category_indices"][0], dtype=torch.long).unsqueeze(0).expand(C, -1)

        news_ctx, user_ctx = model.graph_encoder(
            cand_sag_emb, cand_graph, cand_mask,
            user_hist_emb, u_graph, u_cat_mask, u_cat_indices,
            num_categories,
        )
        scores = (news_ctx * user_ctx).sum(dim=-1)
        assert scores.shape == (C,), f"scores shape: {scores.shape}"
        assert not torch.isnan(scores).any(), "NaN in scores"
        print(f"  evaluator OK: scores {scores.shape}, values {scores.tolist()}")

    print("[OK] DIGAT evaluator pipeline")


# ---------- Run all ----------

if __name__ == "__main__":
    print("=" * 50)
    print("DIGAT Smoke Tests")
    print("=" * 50)
    try:
        test_scatter_ops()
        test_sag_bfs()
        test_model_forward_backward()
        test_user_graph_construction()
        test_evaluator()
        print("=" * 50)
        print("ALL TESTS PASSED")
        print("=" * 50)
    except Exception as e:
        print(f"\nFAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
