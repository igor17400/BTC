"""Semantic Augmented Graph (SAG) construction for DIGAT.

Precomputes per-news subgraphs by:
1. Encoding all news titles via Sentence-BERT.
2. Computing top-k cosine-similar neighbors per news (within category).
3. BFS expansion to build a fixed-size subgraph per news.

Outputs are cached to disk and loaded at training time.

Reference: Mao et al., "DIGAT: Modeling News Recommendation with
Dual-Graph Interaction", EMNLP 2022 Findings.

Obs: This file is located at the part where theoretically we have
framework agnostic files. However in order to make things easier
the preprocessing is made using pytorch, but the rest of the training
is done by the define framework.
"""

from __future__ import annotations

import logging
import os
import pickle
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)

SBERT_MODEL = "sentence-transformers/all-mpnet-base-v2"
SIMILARITY_THRESHOLD = 0.5


def _encode_titles(titles: list[str], batch_size: int = 256) -> np.ndarray:
    """Encode titles using Sentence-BERT → (N, 768) float32."""
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(SBERT_MODEL)
    embeddings = model.encode(titles, batch_size=batch_size, show_progress_bar=True)
    return np.asarray(embeddings, dtype=np.float32)


def _compute_top_k_similarities(
    embeddings: np.ndarray, top_k: int, device: str = "cuda"
) -> tuple[np.ndarray, np.ndarray]:
    """Compute top-k cosine-similar neighbors per embedding.

    Args:
        embeddings: (N, D) float32.
        top_k: Number of neighbors to keep.

    Returns:
        (values, indices) — each (N, top_k) with self excluded.
    """
    emb = torch.from_numpy(embeddings)
    if device == "cuda" and torch.cuda.is_available():
        emb = emb.cuda()
    emb = emb / (emb.norm(dim=1, keepdim=True) + 1e-8)

    N = emb.size(0)
    top_k = min(top_k, N - 1)
    all_values = torch.zeros(N, top_k, device=emb.device)
    all_indices = torch.zeros(N, top_k, dtype=torch.long, device=emb.device)

    chunk = 512
    with torch.no_grad():
        for start in range(0, N, chunk):
            end = min(start + chunk, N)
            sim = torch.mm(emb[start:end], emb.t())  # (chunk, N)
            sim[
                torch.arange(end - start, device=sim.device),
                torch.arange(start, end, device=sim.device),
            ] = -1.0
            vals, idxs = sim.topk(top_k, dim=1)
            all_values[start:end] = vals
            all_indices[start:end] = idxs

    return all_values.cpu().numpy(), all_indices.cpu().numpy()


def _build_news_graphs_bfs(
    similarity_indices: np.ndarray,
    similarity_values: np.ndarray,
    num_news: int,
    news_graph_size: int,
    sag_hops: int,
    sag_neighbors: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """BFS graph construction per news article.

    For each news, expand a subgraph of ``news_graph_size`` nodes via
    BFS through the similarity neighbor list.

    Returns:
        news_node_ID: (num_news, news_graph_size) int32 — node-to-corpus mapping.
        news_graph: (num_news, news_graph_size, news_graph_size) bool — adjacency.
        news_graph_mask: (num_news, news_graph_size) bool — valid nodes.
    """
    G = news_graph_size
    node_ids = np.zeros((num_news, G), dtype=np.int32)
    graph = np.zeros((num_news, G, G), dtype=np.bool_)
    mask = np.zeros((num_news, G), dtype=np.bool_)
    mask[:, 0] = True

    for i in range(num_news):
        node_ids[i, 0] = i
        node_pos = {i: 0}
        depths = [0] * G
        head, rear = 0, 1

        while head < rear:
            if depths[head] >= sag_hops:
                head += 1
                continue
            cur_news = node_ids[i, head]
            for neighbors_used, j in enumerate(range(similarity_indices.shape[1])):
                neighbor = int(similarity_indices[cur_news, j])
                cos_sim = float(similarity_values[cur_news, j])

                if depths[head] > 0 and (
                    cos_sim < SIMILARITY_THRESHOLD
                    or neighbors_used >= sag_neighbors - 1
                ):
                    break
                if depths[head] == 0 and neighbors_used >= sag_neighbors:
                    break

                if neighbor not in node_pos:
                    if rear >= G:
                        break
                    node_ids[i, rear] = neighbor
                    mask[i, rear] = True
                    node_pos[neighbor] = rear
                    graph[i, head, rear] = True
                    graph[i, rear, head] = True
                    depths[rear] = depths[head] + 1
                    rear += 1
                else:
                    pos = node_pos[neighbor]
                    graph[i, head, pos] = True
                    graph[i, pos, head] = True
            head += 1

    return node_ids, graph, mask


def construct_sag(
    all_news_df,
    news_str_id_to_int_idx: dict[str, int],
    sag_hops: int = 2,
    sag_neighbors: int = 5,
    top_k: int = 20,
    cache_dir: Path | None = None,
) -> dict[str, np.ndarray]:
    """Build the Semantic Augmented Graph for all news.

    Args:
        all_news_df: DataFrame with columns 'id', 'title', 'category'.
        news_str_id_to_int_idx: Mapping from string news IDs to int indices.
        sag_hops: BFS depth.
        sag_neighbors: Neighbors per hop.
        top_k: Top-k similar news to consider (before BFS filtering).
        cache_dir: Where to cache intermediate results.

    Returns:
        Dict with keys ``news_node_ID``, ``news_graph``, ``news_graph_mask``.
    """
    # Compute news_graph_size
    news_graph_size = 1
    neighbors = 1
    for i in range(sag_hops):
        if i == 0:
            neighbors *= sag_neighbors
        else:
            neighbors *= sag_neighbors - 1
        news_graph_size += neighbors

    num_news = len(news_str_id_to_int_idx)
    cache_file = None
    if cache_dir is not None:
        cache_file = cache_dir / f"sag_h{sag_hops}_n{sag_neighbors}_k{top_k}.pkl"
        if cache_file.exists():
            logger.info(f"Loading cached SAG from {cache_file}")
            with open(cache_file, "rb") as f:
                return pickle.load(f)

    # Build title list indexed by int ID
    titles = [""] * num_news
    for _, row in all_news_df.drop_duplicates("id").iterrows():
        str_id = row["id"]
        if str_id in news_str_id_to_int_idx:
            idx = news_str_id_to_int_idx[str_id]
            title = str(row.get("title", "")).strip()
            if not title:
                title = str(row.get("abstract", "")).strip() or "empty"
            titles[idx] = title

    for i, t in enumerate(titles):
        if not t:
            titles[i] = "empty"

    logger.info(f"Encoding {num_news} news titles with Sentence-BERT...")
    embeddings = _encode_titles(titles)

    logger.info(f"Computing top-{top_k} similarities...")
    sim_values, sim_indices = _compute_top_k_similarities(embeddings, top_k)

    logger.info(
        f"Building BFS graphs (hops={sag_hops}, neighbors={sag_neighbors}, graph_size={news_graph_size})..."
    )
    node_ids, graph, mask = _build_news_graphs_bfs(
        sim_indices,
        sim_values,
        num_news,
        news_graph_size,
        sag_hops,
        sag_neighbors,
    )

    result = {
        "news_node_ID": node_ids,
        "news_graph": graph,
        "news_graph_mask": mask,
    }

    if cache_file is not None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Caching SAG to {cache_file}")
        with open(cache_file, "wb") as f:
            pickle.dump(result, f)

    return result
