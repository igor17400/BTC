"""Build DIGAT-specific training features from cached behaviors + SAG data.

Also provides ``build_id_remap`` to reconcile the two different integer
ID mappings used by the behaviors processing pipeline (``parse_news_id``)
and the news processing pipeline (``news_str_id_to_int_idx``).  DIGAT
needs both to be in the same space because it indexes SAG data by news
int ID.

Constructs the additional tensors DIGAT needs beyond what standard
models use: SAG subgraph data per candidate, user-topic graphs per
behavior, and category assignments.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def build_id_remap(
    dataset_provider,
    processed_news: dict,
    cache_path: "Path | None" = None,
) -> np.ndarray | None:
    """Build a remap array: ``remap[behaviors_int_id] → SAG_int_id``.

    The behaviors cache and the news/SAG cache use different int ID
    mappings for the same string news IDs.  This function matches them
    via token sequences (each news has a unique title token sequence).

    Returns None for synthetic datasets (no mismatch).
    """
    if cache_path is not None and cache_path.exists():
        logger.info(f"Loading ID remap from {cache_path}")
        return np.load(cache_path)

    if "tokens" not in processed_news:
        return None

    news_tokens = processed_news["tokens"]
    token_to_sag = {}
    for s_id in range(len(news_tokens)):
        token_to_sag[news_tokens[s_id].tobytes()] = s_id

    max_bid = 0
    for split in ["train_behaviors_data", "val_behaviors_data", "test_behaviors_data"]:
        data = getattr(dataset_provider, split, None)
        if data is None:
            continue
        for key in ["candidate_news_ids", "histories_news_ids"]:
            if key not in data:
                continue
            arr = data[key]
            if isinstance(arr, np.ndarray) and arr.dtype.kind in ("i", "u"):
                max_bid = max(max_bid, int(arr.max()))
            elif isinstance(arr, list):
                for item in arr:
                    a = np.asarray(item)
                    if a.dtype.kind in ("i", "u") and len(a) > 0:
                        max_bid = max(max_bid, int(a.max()))

    if max_bid == 0:
        return None

    remap = np.zeros(max_bid + 1, dtype=np.int64)

    for split in ["train_behaviors_data", "val_behaviors_data", "test_behaviors_data"]:
        data = getattr(dataset_provider, split, None)
        if data is None:
            continue
        for id_key, tok_key in [
            ("candidate_news_ids", "candidate_news_tokens"),
            ("histories_news_ids", "history_news_tokens"),
        ]:
            if id_key not in data or tok_key not in data:
                continue
            ids = data[id_key]
            toks = data[tok_key]
            N = len(ids)
            for i in range(N):
                id_arr = np.asarray(ids[i]).flatten().astype(np.int64)
                tok_arr = np.asarray(toks[i])
                if tok_arr.ndim == 1:
                    tok_arr = tok_arr.reshape(1, -1)
                for j in range(min(len(id_arr), len(tok_arr))):
                    bid = int(id_arr[j])
                    if bid > 0 and remap[bid] == 0:
                        key = tok_arr[j].tobytes()
                        if key in token_to_sag:
                            remap[bid] = token_to_sag[key]

    mapped = (remap > 0).sum()
    logger.info(f"Built ID remap: {mapped}/{max_bid} behaviors IDs mapped to SAG IDs")

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, remap)
        logger.info(f"Cached ID remap to {cache_path}")

    return remap


def build_user_graphs(
    history_categories: np.ndarray,
    max_history: int,
    num_categories: int,
) -> dict[str, np.ndarray]:
    """Build user-topic graph adjacency per behavior.

    Graph layout: nodes [0..H-1] = history news, [H..H+C-1] = topic nodes.
    Edge types: news↔topic, news↔news (same topic), topic↔topic,
    self-loops on diagonal.

    Args:
        history_categories: (N, H) int — category index per history news.
            Padding slots should have value 0 (the pad token from our
            category processing).
        max_history: H.
        num_categories: C (including +1 padding category).

    Returns:
        Dict with ``user_graph``, ``user_category_mask``, ``user_category_indices``.
    """
    N, H = history_categories.shape
    graph_size = max_history + num_categories

    user_graph = np.zeros((N, graph_size, graph_size), dtype=np.bool_)
    user_category_mask = np.zeros((N, num_categories), dtype=np.bool_)
    # Default: padding category for all history slots
    user_category_indices = np.full((N, max_history), num_categories - 1, dtype=np.int64)

    # Self-loops
    diag = np.eye(graph_size, dtype=np.bool_)
    user_graph[:] = diag[None, :, :]

    # Identify valid history (non-zero tokens means valid)
    valid = history_categories > 0  # (N, H) — 0 = padding

    for b in range(N):
        active_cats = set()
        for i in range(H):
            if not valid[b, i]:
                continue
            c = int(history_categories[b, i])
            if c >= num_categories:
                continue
            user_category_indices[b, i] = c
            user_category_mask[b, c] = True
            active_cats.add(c)

            # News ↔ Topic edge
            topic_node = max_history + c
            user_graph[b, i, topic_node] = True
            user_graph[b, topic_node, i] = True

            # News ↔ News intra-topic edges
            for j in range(i + 1, H):
                if not valid[b, j]:
                    continue
                cj = int(history_categories[b, j])
                if c == cj:
                    user_graph[b, i, j] = True
                    user_graph[b, j, i] = True
                else:
                    # Topic ↔ Topic inter-topic edges
                    ti = max_history + c
                    tj = max_history + cj
                    user_graph[b, ti, tj] = True
                    user_graph[b, tj, ti] = True

    return {
        "user_graph": user_graph,
        "user_category_mask": user_category_mask,
        "user_category_indices": user_category_indices,
    }


def build_digat_train_features(
    behaviors_data: dict,
    processed_news: dict,
    sag_data: dict,
    max_history: int,
    max_impressions: int,
    num_categories: int,
    news_graph_size: int,
    max_title_length: int,
    news_str_id_to_int: dict[str, int] | None = None,
    int_to_news_str_id: dict[int, str] | None = None,
    id_remap: np.ndarray | None = None,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Build all features DIGAT needs for training.

    Args:
        behaviors_data: Cached behaviors (from the standard pipeline).
        processed_news: Processed news data (tokens, categories, etc.).
        sag_data: SAG cache (news_node_ID, news_graph, news_graph_mask).
        max_history: Max history length.
        max_impressions: Number of candidates per behavior.
        num_categories: Number of categories (+1 for padding).
        news_graph_size: SAG subgraph size (26 for defaults).
        max_title_length: Title token length.

    Returns:
        (features_dict, labels) ready for the training dataloader.
    """
    news_tokens = processed_news["tokens"]  # (num_news, title_len)
    news_node_ID = sag_data["news_node_ID"]  # (num_news, G_n)
    news_graph = sag_data["news_graph"]  # (num_news, G_n, G_n)
    news_graph_mask = sag_data["news_graph_mask"]  # (num_news, G_n)

    hist_tokens = np.asarray(behaviors_data["history_news_tokens"])  # (N, H, T)
    hist_categories = np.asarray(behaviors_data["history_news_categories"])  # (N, H)
    raw_cand_ids = np.asarray(behaviors_data["candidate_news_ids"])
    if raw_cand_ids.dtype.kind in ("U", "S", "O") and news_str_id_to_int is not None:
        vfunc = np.vectorize(lambda x: news_str_id_to_int.get(str(x), 0))
        cand_ids = vfunc(raw_cand_ids).astype(np.int64)
    else:
        cand_ids = raw_cand_ids.astype(np.int64)

    # Remap behaviors int IDs → SAG int IDs (different mappings)
    num_sag_news = news_node_ID.shape[0]
    if id_remap is not None:
        cand_ids = np.where(cand_ids < len(id_remap), id_remap[cand_ids], 0).astype(np.int64)
        logger.info(f"Remapped candidate IDs via remap array ({(cand_ids == 0).sum()} unmapped)")
    elif cand_ids.max() >= num_sag_news:
        logger.warning(f"Clamping {(cand_ids >= num_sag_news).sum()} out-of-range IDs")
        cand_ids = np.clip(cand_ids, 0, num_sag_news - 1)
    labels = np.asarray(behaviors_data["labels"])  # (N, C)

    N = hist_tokens.shape[0]
    G_n = news_graph_size
    T = max_title_length

    logger.info("Building DIGAT candidate SAG features...")
    # Candidate SAG: expand each candidate to its SAG subgraph tokens
    # cand_ids: (N, C) → SAG node IDs: (N, C, G_n) → tokens: (N, C, G_n, T)
    cand_sag_node_ids = news_node_ID[cand_ids]  # (N, C, G_n)
    cand_tokens = news_tokens[cand_sag_node_ids]  # (N, C, G_n, T)
    cand_mask = (cand_tokens != 0).astype(np.float32)  # (N, C, G_n, T)
    cand_graph = news_graph[cand_ids]  # (N, C, G_n, G_n)
    cand_graph_mask = news_graph_mask[cand_ids]  # (N, C, G_n)

    logger.info("Building DIGAT user-topic graphs...")
    user_data = build_user_graphs(hist_categories, max_history, num_categories)

    # History mask for title tokens
    hist_mask = (hist_tokens != 0).astype(np.float32)  # (N, H, T)

    features = {
        "hist_tokens": hist_tokens,
        "hist_mask": hist_mask,
        "cand_tokens": cand_tokens,
        "cand_mask": cand_mask,
        "cand_graph": cand_graph.astype(np.float32),
        "cand_graph_mask": cand_graph_mask.astype(np.float32),
        "user_graph": user_data["user_graph"].astype(np.float32),
        "user_category_mask": user_data["user_category_mask"].astype(np.float32),
        "user_category_indices": user_data["user_category_indices"],
    }

    logger.info(
        f"DIGAT features: cand_tokens {cand_tokens.shape}, "
        f"user_graph {user_data['user_graph'].shape}, "
        f"N={N} behaviors"
    )
    return features, labels
