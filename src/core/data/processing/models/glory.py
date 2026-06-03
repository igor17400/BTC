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


def _build_edges_from_raw_behaviors(
    raw_behaviors_path: Path,
    news_str_id_to_int_idx: dict[str, int],
    use_graph_type: int,
) -> Counter:
    """Build edge counter by reading raw behaviors.tsv (FULL history).

    Mirrors reference GLORY's ``prepare_news_graph``
    (``reference_codes/GLORY/src/dataload/data_preprocess.py:242-278``)
    which uses each user's full click history straight from
    behaviors.tsv -- NOT the truncated ``histories_news_ids`` array
    (which is capped at ``max_history_length=50`` for the model's
    user-encoder input). For users with >50 clicks the truncated
    version silently drops the bulk of their click trajectory, which
    severely depresses edge weights and starves GLORY's global graph
    encoder. See GLORY_PARITY_FINDINGS.md.
    """
    edge_counter: Counter = Counter()
    seen_users: set = set()
    with open(raw_behaviors_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 4:
                continue
            user_id = parts[1]
            if user_id in seen_users:
                continue
            seen_users.add(user_id)
            history_str_ids = parts[3].split() if parts[3] else []
            history = [
                news_str_id_to_int_idx[s]
                for s in history_str_ids
                if s in news_str_id_to_int_idx
            ]
            if len(history) < 2:
                continue
            if use_graph_type == 0:
                for i in range(len(history) - 1):
                    edge_counter[(history[i], history[i + 1])] += 1
            elif use_graph_type == 1:
                for i in range(len(history) - 1):
                    for j in range(i + 1, len(history)):
                        edge_counter[(history[i], history[j])] += 1
                        edge_counter[(history[j], history[i])] += 1
            else:
                raise ValueError(f"Unknown use_graph_type={use_graph_type}")
    return edge_counter


def build_news_graph(
    train_behaviors: dict,
    num_news: int,
    use_graph_type: int = 0,
    cache_path: Path | None = None,
    raw_behaviors_path: Path | None = None,
    news_str_id_to_int_idx: dict[str, int] | None = None,
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
        raw_behaviors_path: Optional path to the raw ``behaviors.tsv``
            file.  When provided (with ``news_str_id_to_int_idx``),
            edges are built from each user's FULL history (matching
            reference GLORY).  When not provided, falls back to the
            truncated ``histories_news_ids`` in ``train_behaviors``
            (capped at ``max_history_length``).
        news_str_id_to_int_idx: String news ID → processed_news row
            index map (required when ``raw_behaviors_path`` is given).

    Returns:
        Dict with ``edge_index`` ``(2, E)`` int64, ``edge_attr`` ``(E,)``
        int64, ``num_nodes`` int.
    """
    if cache_path is not None and cache_path.exists():
        logger.info(f"Loading cached GLORY news graph from {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    # Preferred path: read raw behaviors.tsv → full histories (matches
    # reference). Falls back to truncated ``histories_news_ids`` only
    # when raw data isn't reachable (e.g. synthetic datasets).
    if (
        raw_behaviors_path is not None
        and news_str_id_to_int_idx is not None
        and raw_behaviors_path.exists()
    ):
        logger.info(
            f"Building GLORY news graph from raw behaviors at "
            f"{raw_behaviors_path} (full histories)."
        )
        edge_counter = _build_edges_from_raw_behaviors(
            raw_behaviors_path, news_str_id_to_int_idx, use_graph_type
        )
        edges = list(edge_counter.keys())
        if edges:
            edge_index = np.asarray(edges, dtype=np.int64).T
            edge_attr = np.asarray([edge_counter[e] for e in edges], dtype=np.int64)
        else:
            edge_index = np.zeros((2, 0), dtype=np.int64)
            edge_attr = np.zeros((0,), dtype=np.int64)
        graph = {
            "edge_index": edge_index,
            "edge_attr": edge_attr,
            "num_nodes": int(num_news),
        }
        logger.info(
            f"Built GLORY news graph (raw-behaviors): {graph['num_nodes']} "
            f"nodes, {edge_index.shape[1]} directed edges, "
            f"total weight {int(edge_attr.sum())}."
        )
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "wb") as f:
                pickle.dump(graph, f)
            logger.info(f"Cached GLORY news graph to {cache_path}")
        return graph

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
# Entity graph (for entity-aware GLORY)
# ----------------------------------------------------------------------


def build_entity_graph(
    news_graph: dict,
    news_features: np.ndarray,
    title_size: int,
    entity_size: int,
    cache_path: Path | None = None,
) -> dict:
    """Build an entity-entity graph from the news graph.

    For each edge (src_news, dst_news) in the news graph, creates edges
    between all pairs of (src_entities, dst_entities) weighted by the
    news edge weight.  Matches the reference ``prepare_entity_graph``.

    Args:
        news_graph: Output of :func:`build_news_graph`.
        news_features: ``(num_news, feat_dim)`` from
            :func:`build_news_feature_matrix`.
        title_size: Number of title columns (entity columns start after).
        entity_size: Number of entity columns per news.
        cache_path: Optional cache path.

    Returns:
        Dict with ``edge_index`` ``(2, E)`` int64, ``edge_attr``
        ``(E,)`` int64, ``num_nodes`` int.
    """
    if cache_path is not None and cache_path.exists():
        logger.info(f"Loading cached GLORY entity graph from {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    edge_index = news_graph["edge_index"]
    edge_attr = news_graph["edge_attr"]

    # Extract entity columns from news features.
    entity_indices = news_features[:, title_size : title_size + entity_size]

    edge_counter: Counter = Counter()
    if edge_index.size > 0:
        news_src = edge_index[0]
        news_dst = edge_index[1]
        weights = edge_attr.astype(np.int64)

        for i in range(news_src.shape[0]):
            src_ents = entity_indices[news_src[i]]
            dst_ents = entity_indices[news_dst[i]]
            src_ents = src_ents[src_ents > 0]
            dst_ents = dst_ents[dst_ents > 0]
            if src_ents.size == 0 or dst_ents.size == 0:
                continue
            w = int(weights[i])
            for s in src_ents:
                for d in dst_ents:
                    edge_counter[(int(s), int(d))] += w

    # Determine num_nodes from max entity index seen.
    all_entity_ids = entity_indices[entity_indices > 0]
    num_entity_nodes = int(all_entity_ids.max()) + 1 if all_entity_ids.size > 0 else 1

    if not edge_counter:
        ent_edge_index = np.zeros((2, 0), dtype=np.int64)
        ent_edge_attr = np.zeros((0,), dtype=np.int64)
    else:
        edges = list(edge_counter.keys())
        ent_edge_index = np.asarray(edges, dtype=np.int64).T
        ent_edge_attr = np.asarray(
            [edge_counter[e] for e in edges],
            dtype=np.int64,
        )
        # Make undirected (matching reference).
        rev = ent_edge_index[[1, 0]]
        ent_edge_index = np.concatenate([ent_edge_index, rev], axis=1)
        ent_edge_attr = np.concatenate([ent_edge_attr, ent_edge_attr], axis=0)
        # Deduplicate after symmetrization.
        pairs = list(zip(ent_edge_index[0].tolist(), ent_edge_index[1].tolist()))
        dedup: dict[tuple[int, int], int] = {}
        for idx, p in enumerate(pairs):
            dedup[p] = dedup.get(p, 0) + int(ent_edge_attr[idx])
        edges_dedup = list(dedup.keys())
        ent_edge_index = np.asarray(edges_dedup, dtype=np.int64).T
        ent_edge_attr = np.asarray(
            [dedup[e] for e in edges_dedup],
            dtype=np.int64,
        )

    graph = {
        "edge_index": ent_edge_index,
        "edge_attr": ent_edge_attr,
        "num_nodes": num_entity_nodes,
    }
    logger.info(
        f"Built GLORY entity graph: {num_entity_nodes} nodes, "
        f"{ent_edge_index.shape[1]} edges"
    )

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(graph, f)
        logger.info(f"Cached GLORY entity graph to {cache_path}")

    return graph


def build_entity_neighbor_dict(
    entity_graph: dict,
    cache_path: Path | None = None,
) -> dict[int, list[int]]:
    """Build per-entity sorted neighbor list from the entity graph.

    Same logic as :func:`build_neighbor_dict` but for the entity graph.
    Neighbors are sorted by edge weight descending.

    Args:
        entity_graph: Output of :func:`build_entity_graph`.
        cache_path: Optional cache path.

    Returns:
        ``{entity_id: [neighbor_ids]}`` dict.
    """
    if cache_path is not None and cache_path.exists():
        logger.info(f"Loading cached GLORY entity neighbor dict from {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    edge_index = np.asarray(entity_graph["edge_index"])
    edge_attr = np.asarray(entity_graph["edge_attr"])
    num_nodes = int(entity_graph["num_nodes"])

    dst_to_src: dict[int, list[tuple[int, int]]] = defaultdict(list)
    if edge_index.size > 0:
        for i in range(edge_index.shape[1]):
            src = int(edge_index[0, i])
            dst = int(edge_index[1, i])
            w = int(edge_attr[i])
            dst_to_src[dst].append((src, w))

    neighbor_dict: dict[int, list[int]] = {}
    for node in range(num_nodes):
        pairs = dst_to_src.get(node, [])
        pairs.sort(key=lambda p: p[1], reverse=True)
        neighbor_dict[node] = [p[0] for p in pairs]

    logger.info(
        f"Built GLORY entity neighbor dict: "
        f"{sum(len(v) for v in neighbor_dict.values())} entries"
    )

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(neighbor_dict, f)
        logger.info(f"Cached GLORY entity neighbor dict to {cache_path}")

    return neighbor_dict


# ----------------------------------------------------------------------
# Subgraph construction (used by the dataloader)
# ----------------------------------------------------------------------


def sample_subgraph(
    history_ids: np.ndarray,
    neighbor_dict: dict[int, list[int]],
    k_hops: int,
    num_neighbors: int,
    max_nodes: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """k-hop neighbor expansion from a user's history.

    BFS-style expansion: at each hop, every current node contributes its
    top-``num_neighbors`` neighbors (by edge weight).  The resulting
    node list is de-duplicated and ``history_mapping`` gives the
    de-duplicated indices of the original history items.

    Args:
        history_ids: ``(H,)`` int array of news IDs (0 = padding).
        neighbor_dict: From :func:`build_neighbor_dict`.
        k_hops: Expansion depth.
        num_neighbors: Max neighbors per node per hop.
        max_nodes: If set, pad/truncate the subgraph to exactly this
            many nodes. Padding extends with ``0`` (the padding sentinel
            news id); truncation prefers history nodes first, then
            earlier hops, dropping the latest hop's neighbors first.
            **Used for JAX so that XLA compiles a single fixed-shape
            kernel across all impressions.** ``None`` (default) keeps
            the original variable-size behavior for PyTorch.

    Returns:
        ``(unique_node_ids, history_mapping)`` — both numpy int arrays.
        When ``max_nodes`` is ``None``, ``unique_node_ids`` is the sorted
        union of history + expanded neighbors; ``history_mapping[i]`` is
        the index of ``history_ids[i]`` within ``unique_node_ids``.
        When ``max_nodes`` is set, ``unique_node_ids`` has shape
        ``(max_nodes,)`` exactly, ordered as
        ``[history..., hop1_neighbors..., hop2_neighbors..., 0_padding...]``,
        and ``history_mapping[i]`` is the corresponding index.
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

    if max_nodes is None:
        # Existing path: sort-dedupe, hist_mapping is the inverse index.
        unique, inverse = np.unique(
            np.asarray(all_nodes, dtype=np.int64), return_inverse=True
        )
        hist_mapping = inverse[: hist_valid.size]
        return unique, hist_mapping

    # Padded path (for JAX): order-preserving dedupe, then truncate /
    # pad to exactly ``max_nodes``. History nodes occupy the first
    # ``hist_valid.size`` slots, so they survive truncation as long as
    # ``hist_valid.size <= max_nodes`` (always true with his_size=50,
    # max_nodes>=64).
    seen: dict[int, int] = {}
    ordered: list[int] = []
    for node in all_nodes:
        node_int = int(node)
        if node_int not in seen:
            seen[node_int] = len(ordered)
            ordered.append(node_int)
        if len(ordered) >= max_nodes:
            break  # truncate: drop remaining (latest-hop) neighbors

    while len(ordered) < max_nodes:
        ordered.append(0)  # pad with news id 0 (the padding sentinel)

    unique = np.asarray(ordered, dtype=np.int64)
    hist_mapping = np.asarray([seen[int(h)] for h in hist_valid], dtype=np.int64)
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
    max_edges: int | None = None,
    n_real_nodes: int | None = None,
    pad_node_idx: int | None = None,
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
        return _pad_edges(
            np.zeros((2, 0), dtype=np.int64),
            np.zeros((0,), dtype=np.int64),
            max_edges,
            pad_node_idx,
        )

    # When the caller padded ``node_ids``, only the first ``n_real_nodes``
    # entries are real news; the rest are sentinels that we must skip
    # during edge extraction (otherwise global_to_local[0] gets clobbered
    # by repeated padding entries).
    if n_real_nodes is not None:
        real_node_ids = node_ids[:n_real_nodes]
    else:
        real_node_ids = node_ids

    if csr is not None and num_nodes is not None:
        indptr, in_srcs = csr
        # Membership test in O(1) per lookup.
        node_set = np.zeros(num_nodes, dtype=bool)
        node_set[real_node_ids] = True
        global_to_local = np.full(num_nodes, -1, dtype=np.int64)
        global_to_local[real_node_ids] = np.arange(real_node_ids.size, dtype=np.int64)

        sub_src_chunks: list[np.ndarray] = []
        sub_dst_chunks: list[np.ndarray] = []
        for local_dst, dst in enumerate(real_node_ids):
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
            return _pad_edges(
                np.zeros((2, 0), dtype=np.int64),
                np.zeros((0,), dtype=np.int64),
                max_edges,
                pad_node_idx,
            )
        sub_src = np.concatenate(sub_src_chunks)
        sub_dst = np.concatenate(sub_dst_chunks)
        return _pad_edges(
            np.stack([sub_src, sub_dst], axis=0).astype(np.int64),
            np.zeros(sub_src.size, dtype=np.int64),
            max_edges,
            pad_node_idx,
        )

    # Fallback: searchsorted path (slow but keeps the public API
    # working for callers that don't supply a CSR).
    if edge_index.size == 0:
        return _pad_edges(
            np.zeros((2, 0), dtype=np.int64),
            np.zeros((0,), dtype=np.int64),
            max_edges,
            pad_node_idx,
        )

    node_set = np.sort(real_node_ids)
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
        return _pad_edges(
            np.zeros((2, 0), dtype=np.int64),
            np.zeros((0,), dtype=np.int64),
            max_edges,
            pad_node_idx,
        )
    sub_src = pos_src[mask]
    sub_dst = pos_dst[mask]
    sub_attr = (
        edge_attr[mask] if edge_attr.size > 0 else np.zeros(mask.sum(), dtype=np.int64)
    )
    return _pad_edges(
        np.stack([sub_src, sub_dst], axis=0).astype(np.int64),
        sub_attr,
        max_edges,
        pad_node_idx,
    )


def _pad_edges(
    sub_edges: np.ndarray,
    sub_attr: np.ndarray,
    max_edges: int | None,
    pad_node_idx: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Pad/truncate a subgraph's edge_index to ``max_edges`` (no-op when
    ``max_edges is None``).

    Padded slots are self-loops at ``pad_node_idx``, which should be a
    reserved position (e.g. ``max_nodes - 1``) that's never gathered by
    ``hist_mapping`` — keeps real nodes' aggregated messages unaffected.
    """
    if max_edges is None:
        return sub_edges, sub_attr
    e = sub_edges.shape[1]
    if e == max_edges:
        return sub_edges, sub_attr
    if e > max_edges:
        # Truncate (rare; only when the subgraph is unusually dense).
        return sub_edges[:, :max_edges], sub_attr[:max_edges]
    # Pad with (pad_node_idx, pad_node_idx) self-loops.
    pidx = pad_node_idx if pad_node_idx is not None else 0
    pad_count = max_edges - e
    pad_block = np.full((2, pad_count), pidx, dtype=sub_edges.dtype)
    sub_edges = np.concatenate([sub_edges, pad_block], axis=1)
    sub_attr = np.concatenate(
        [sub_attr, np.zeros(pad_count, dtype=sub_attr.dtype)], axis=0
    )
    return sub_edges, sub_attr
