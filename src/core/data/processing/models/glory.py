"""GLORY-specific data preprocessing.

Pure numpy helpers shared across frameworks (PyTorch / JAX / Keras):

- :func:`build_news_feature_matrix` — assemble the per-news feature
  tensor ``(num_news, feature_dim)`` where feature_dim =
  ``title_size + entity_size + 3`` (title tokens, entity ids, category,
  subcategory, news_index).
- :func:`build_news_graph` — construct the global news graph from
  user click trajectories.  Returns a numpy dict with ``edge_index``,
  ``edge_attr``, ``num_nodes``; caches to ``processed/glory_news_graph.pkl``.
- :func:`build_neighbor_dict` — from the graph, produce a per-node
  sorted neighbor list (by edge weight) for on-the-fly subgraph
  sampling in the training loop.

No framework dependency — every output is plain numpy or Python
containers.  Structurally mirrors :mod:`src.core.data.processing.sag`
and :mod:`src.core.data.processing.digat`.
"""

from __future__ import annotations

import logging
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# News feature assembly
# ----------------------------------------------------------------------


def build_news_feature_matrix(
    processed_news: dict,
    title_size: int,
    entity_size: int,
) -> np.ndarray:
    """Assemble GLORY's per-news feature tensor.

    Layout (matches GLORY's reference ``nltk_token_news``):
    ``[title_tokens | entity_ids | category | subcategory | news_index]``
    with total width ``title_size + entity_size + 3``.

    Args:
        processed_news: Dict from the standard news processing pipeline.
            Must contain ``tokens``, ``category_indices``,
            ``subcategory_indices``.  ``entity_indices`` is optional —
            if missing, entity columns are zero-padded.
        title_size: Number of title token columns.
        entity_size: Number of entity id columns.

    Returns:
        ``(num_news, title_size + entity_size + 3)`` int32 array.
    """
    tokens = np.asarray(processed_news["tokens"])[:, :title_size].astype(np.int32)
    num_news = tokens.shape[0]

    if tokens.shape[1] < title_size:
        pad = np.zeros((num_news, title_size - tokens.shape[1]), dtype=np.int32)
        tokens = np.concatenate([tokens, pad], axis=1)

    if (
        "entity_indices" in processed_news
        and processed_news["entity_indices"] is not None
    ):
        entities = np.asarray(processed_news["entity_indices"])[:, :entity_size]
        if entities.shape[1] < entity_size:
            pad = np.zeros((num_news, entity_size - entities.shape[1]), dtype=np.int32)
            entities = np.concatenate([entities, pad], axis=1)
    else:
        entities = np.zeros((num_news, entity_size), dtype=np.int32)

    category = np.asarray(
        processed_news.get("category_indices", np.zeros(num_news))
    ).reshape(num_news, 1)
    subcategory = np.asarray(
        processed_news.get("subcategory_indices", np.zeros(num_news))
    ).reshape(num_news, 1)
    news_idx = np.arange(num_news, dtype=np.int32).reshape(num_news, 1)

    return np.concatenate(
        [
            tokens,
            entities.astype(np.int32),
            category.astype(np.int32),
            subcategory.astype(np.int32),
            news_idx,
        ],
        axis=1,
    ).astype(np.int32)


# ----------------------------------------------------------------------
# Global news graph
# ----------------------------------------------------------------------


def build_news_graph(
    train_behaviors: dict,
    num_news: int,
    use_graph_type: int = 0,
    cache_path: Path | None = None,
) -> dict:
    """Build the global news graph from user click trajectories.

    Mirrors ``prepare_news_graph`` in the reference GLORY.  One edge per
    ordered (or unordered, for co-occurrence) pair of consecutive clicks
    in each user's history; edge weights are pair counts.

    Args:
        train_behaviors: Dict with ``histories_news_ids`` — a sequence
            (one per impression) of arrays of int news IDs.  Duplicate
            users are filtered (first-seen-wins) matching the reference.
        num_news: Total number of news articles (for ``num_nodes``).
        use_graph_type: 0 = trajectory (directed consecutive edges),
            1 = co-occurrence (undirected all-pairs).
        cache_path: Optional path to load/save the result.

    Returns:
        Dict with ``edge_index`` ``(2, E)`` int64, ``edge_attr`` ``(E,)``
        int64, ``num_nodes`` int.
    """
    if cache_path is not None and cache_path.exists():
        logger.info(f"Loading cached GLORY news graph from {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    histories = train_behaviors.get("histories_news_ids")
    if histories is None:
        # Fallback for datasets (e.g. synthetic) that don't emit news
        # IDs for history — degenerate graph, still valid as a dataset.
        logger.warning(
            "train_behaviors missing 'histories_news_ids'; returning empty graph. "
            "GLORY's global encoder will effectively be an identity map."
        )
        graph = {
            "edge_index": np.zeros((2, 0), dtype=np.int64),
            "edge_attr": np.zeros((0,), dtype=np.int64),
            "num_nodes": int(num_news),
        }
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "wb") as f:
                pickle.dump(graph, f)
        return graph

    user_ids = train_behaviors.get("user_ids")
    seen_users: set = set()

    edge_counter: Counter = Counter()
    num_histories = len(histories)

    for idx in range(num_histories):
        # Dedup users (matches GLORY's first-seen-wins policy).
        if user_ids is not None:
            uid = int(user_ids[idx])
            if uid in seen_users:
                continue
            seen_users.add(uid)

        history = np.asarray(histories[idx]).astype(np.int64)
        # Strip padding zeros; keep order.
        history = history[history > 0]
        if history.size < 2:
            continue

        if use_graph_type == 0:
            # Trajectory: directed (h[i], h[i+1]) edges.
            for i in range(history.size - 1):
                edge_counter[(int(history[i]), int(history[i + 1]))] += 1
        elif use_graph_type == 1:
            # Co-occurrence: both directions for every pair in history.
            for i in range(history.size - 1):
                hi = int(history[i])
                for j in range(i + 1, history.size):
                    hj = int(history[j])
                    edge_counter[(hi, hj)] += 1
                    edge_counter[(hj, hi)] += 1
        else:
            raise ValueError(f"Unknown use_graph_type={use_graph_type}")

    if not edge_counter:
        logger.warning(
            "No edges built for GLORY news graph — empty or invalid behaviors"
        )
        edge_index = np.zeros((2, 0), dtype=np.int64)
        edge_attr = np.zeros((0,), dtype=np.int64)
    else:
        edges = list(edge_counter.keys())
        edge_index = np.asarray(edges, dtype=np.int64).T  # (2, E)
        edge_attr = np.asarray(
            [edge_counter[e] for e in edges],
            dtype=np.int64,
        )

    graph = {
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "num_nodes": int(num_news),
    }
    logger.info(
        f"Built GLORY news graph: {graph['num_nodes']} nodes, "
        f"{edge_index.shape[1]} directed edges"
    )

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(graph, f)
        logger.info(f"Cached GLORY news graph to {cache_path}")

    return graph


# ----------------------------------------------------------------------
# Neighbor dict (for on-the-fly subgraph sampling)
# ----------------------------------------------------------------------


def build_neighbor_dict(
    graph: dict,
    directed: bool = True,
    cache_path: Path | None = None,
) -> dict[int, list[int]]:
    """Build a per-node sorted neighbor list from a news graph.

    For each node ``i``, returns its in-neighbors sorted by edge weight
    (descending) — matching the reference's "dst edges, src nodes"
    indexing semantics.  When ``directed=False`` edges are symmetrised
    first.

    Args:
        graph: Output of :func:`build_news_graph`.
        directed: If ``False``, concatenate reverse edges before sorting.
        cache_path: Optional cache path.

    Returns:
        ``{node_id: [neighbor_ids]}`` dict.  Node ``0`` is a padding
        sentinel and gets an empty list.
    """
    if cache_path is not None and cache_path.exists():
        logger.info(f"Loading cached GLORY neighbor dict from {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    edge_index = np.asarray(graph["edge_index"])
    edge_attr = np.asarray(graph["edge_attr"])
    num_nodes = int(graph["num_nodes"])

    if not directed and edge_index.size > 0:
        rev = edge_index[[1, 0]]
        edge_index = np.concatenate([edge_index, rev], axis=1)
        edge_attr = np.concatenate([edge_attr, edge_attr], axis=0)

    # For each dst, gather (src, weight) pairs.
    if edge_index.size > 0:
        dst_to_src: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for i in range(edge_index.shape[1]):
            src = int(edge_index[0, i])
            dst = int(edge_index[1, i])
            w = int(edge_attr[i])
            dst_to_src[dst].append((src, w))
    else:
        dst_to_src = {}

    neighbor_dict: dict[int, list[int]] = {}
    for node in range(num_nodes):
        pairs = dst_to_src.get(node, [])
        pairs.sort(key=lambda p: p[1], reverse=True)
        neighbor_dict[node] = [p[0] for p in pairs]

    logger.info(
        f"Built GLORY neighbor dict: {sum(len(v) for v in neighbor_dict.values())} entries"
    )

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(neighbor_dict, f)
        logger.info(f"Cached GLORY neighbor dict to {cache_path}")

    return neighbor_dict


# ----------------------------------------------------------------------
# Subgraph construction (used by the dataloader)
# ----------------------------------------------------------------------


def sample_subgraph(
    history_ids: np.ndarray,
    neighbor_dict: dict[int, list[int]],
    k_hops: int,
    num_neighbors: int,
) -> tuple[np.ndarray, np.ndarray]:
    """k-hop neighbor expansion from a user's history.

    BFS-style expansion: at each hop, every current node contributes its
    top-``num_neighbors`` neighbors (by edge weight).  The resulting
    node list is de-duplicated via ``np.unique`` (sorted), and
    ``history_mapping`` gives the de-duplicated indices of the original
    history items.

    Args:
        history_ids: ``(H,)`` int array of news IDs (0 = padding).
        neighbor_dict: From :func:`build_neighbor_dict`.
        k_hops: Expansion depth.
        num_neighbors: Max neighbors per node per hop.

    Returns:
        ``(unique_node_ids, history_mapping)`` — both numpy int arrays.
        ``unique_node_ids`` is the sorted union of history + expanded
        neighbors; ``history_mapping[i]`` is the index of
        ``history_ids[i]`` within ``unique_node_ids``.
    """
    hist = np.asarray(history_ids, dtype=np.int64)
    hist_valid = hist[hist > 0]
    if hist_valid.size == 0:
        hist_valid = np.array([0], dtype=np.int64)

    all_nodes = list(hist_valid.tolist())
    current = all_nodes[:]
    for _ in range(k_hops):
        next_hop: list[int] = []
        for node in current:
            ns = neighbor_dict.get(int(node), [])
            next_hop.extend(ns[:num_neighbors])
        current = next_hop
        all_nodes.extend(next_hop)

    unique, inverse = np.unique(
        np.asarray(all_nodes, dtype=np.int64), return_inverse=True
    )
    # inverse[:len(hist_valid)] gives us the mapping for original history.
    hist_mapping = inverse[: hist_valid.size]
    return unique, hist_mapping


def build_csr_in_adjacency(
    edge_index: np.ndarray,
    num_nodes: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert ``(2, E)`` edge list to CSR incoming-adjacency.

    For each dst node, produces the list of src nodes that send edges
    into it.  Output layout::

        indptr[i:i+1] bounds the slice in ``in_srcs`` for dst=i.

    Returns:
        ``(indptr, in_srcs)`` both int64.
    """
    if edge_index.size == 0:
        return (
            np.zeros(num_nodes + 1, dtype=np.int64),
            np.zeros(0, dtype=np.int64),
        )

    src, dst = edge_index[0].astype(np.int64), edge_index[1].astype(np.int64)
    # Group edges by destination.
    order = np.argsort(dst, kind="stable")
    sorted_dst = dst[order]
    sorted_src = src[order]
    # Build cumulative counts: indptr[i] = start offset for dst=i.
    counts = np.bincount(sorted_dst, minlength=num_nodes + 1).astype(np.int64)
    indptr = np.concatenate([[0], np.cumsum(counts[:num_nodes])])
    # Append the total count as the sentinel end-pointer.
    indptr = np.concatenate([indptr, [sorted_src.size]])
    return indptr.astype(np.int64), sorted_src


def extract_edges_for_subgraph(
    node_ids: np.ndarray,
    edge_index: np.ndarray,
    edge_attr: np.ndarray,
    *,
    csr: tuple[np.ndarray, np.ndarray] | None = None,
    num_nodes: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract the node-induced subgraph on ``node_ids``.

    Both endpoints must lie in ``node_ids`` for an edge to be kept.
    Edge indices are relabeled to ``[0, len(node_ids))``.

    Args:
        node_ids: ``(N_sub,)`` global node ids in the subgraph (sorted
            ascending, matching ``sample_subgraph``'s return order).
        edge_index: full graph edges ``(2, E)``, used only when ``csr``
            is not provided.
        edge_attr: full graph edge weights ``(E,)``, used only when
            ``csr`` is not provided.
        csr: optional pre-built ``(indptr, in_srcs)`` from
            :func:`build_csr_in_adjacency`.  When provided, extraction
            is O(|subgraph| · avg_in_degree) instead of O(E log N).
        num_nodes: total node count in the full graph; required when
            ``csr`` is given.

    Returns:
        ``(sub_edges, sub_attr)`` where ``sub_edges`` is ``(2, E_sub)``
        with local indices and ``sub_attr`` is an empty placeholder
        (GLORY's ``GatedGraphConv`` ignores edge weights).
    """
    if node_ids.size == 0:
        return np.zeros((2, 0), dtype=np.int64), np.zeros((0,), dtype=np.int64)

    if csr is not None and num_nodes is not None:
        indptr, in_srcs = csr
        # Membership test in O(1) per lookup.
        node_set = np.zeros(num_nodes, dtype=bool)
        node_set[node_ids] = True
        global_to_local = np.full(num_nodes, -1, dtype=np.int64)
        global_to_local[node_ids] = np.arange(node_ids.size, dtype=np.int64)

        sub_src_chunks: list[np.ndarray] = []
        sub_dst_chunks: list[np.ndarray] = []
        for local_dst, dst in enumerate(node_ids):
            start = indptr[dst]
            end = indptr[dst + 1]
            if start == end:
                continue
            srcs = in_srcs[start:end]
            keep = node_set[srcs]
            if not keep.any():
                continue
            valid_srcs = srcs[keep]
            sub_src_chunks.append(global_to_local[valid_srcs])
            sub_dst_chunks.append(np.full(valid_srcs.size, local_dst, dtype=np.int64))

        if not sub_src_chunks:
            return np.zeros((2, 0), dtype=np.int64), np.zeros((0,), dtype=np.int64)
        sub_src = np.concatenate(sub_src_chunks)
        sub_dst = np.concatenate(sub_dst_chunks)
        return (
            np.stack([sub_src, sub_dst], axis=0).astype(np.int64),
            np.zeros(sub_src.size, dtype=np.int64),
        )

    # Fallback: searchsorted path (slow but keeps the public API
    # working for callers that don't supply a CSR).
    if edge_index.size == 0:
        return np.zeros((2, 0), dtype=np.int64), np.zeros((0,), dtype=np.int64)

    node_set = np.sort(node_ids)
    src, dst = edge_index[0], edge_index[1]
    pos_src = np.searchsorted(node_set, src)
    pos_dst = np.searchsorted(node_set, dst)
    valid_src = (pos_src < node_set.size) & (
        node_set[np.minimum(pos_src, node_set.size - 1)] == src
    )
    valid_dst = (pos_dst < node_set.size) & (
        node_set[np.minimum(pos_dst, node_set.size - 1)] == dst
    )
    mask = valid_src & valid_dst
    if not mask.any():
        return np.zeros((2, 0), dtype=np.int64), np.zeros((0,), dtype=np.int64)
    sub_src = pos_src[mask]
    sub_dst = pos_dst[mask]
    sub_attr = (
        edge_attr[mask] if edge_attr.size > 0 else np.zeros(mask.sum(), dtype=np.int64)
    )
    return np.stack([sub_src, sub_dst], axis=0).astype(np.int64), sub_attr
