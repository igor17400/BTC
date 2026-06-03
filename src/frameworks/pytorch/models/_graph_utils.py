"""Shared graph operations for graph-based news-recommendation models.

These utilities are used across CROWN / DIGAT / GLORY where PyG's
high-level GNN classes (``GatedGraphConv``, ``SAGEConv``, ``GATConv``)
don't directly cover the needed shape — most often when an op needs
**batched, per-row scatter** (each row has its own group structure)
rather than 1D scatter over a flat edge list.

The two scatter helpers below match the semantics of the
``torch_scatter.scatter_softmax`` and ``torch_scatter.scatter_sum``
calls used in DIGAT's reference implementation
(``reference_codes/digat/graphEncoders.py:7,129``), without requiring
the compiled ``torch_scatter`` wheel — they use ``torch.scatter_reduce_``
plus ``gather``.

``torch_geometric.utils.softmax`` / ``torch_geometric.utils.scatter``
provide the same semantics for flat (1D) edge lists; reach for those
when a model emits an unbatched edge representation.
"""

from __future__ import annotations

import torch

__all__ = ["scatter_softmax_batched", "scatter_sum_batched"]


def scatter_softmax_batched(
    src: torch.Tensor, index: torch.Tensor, num_groups: int
) -> torch.Tensor:
    """Per-group softmax with batched, per-row group indices.

    Args:
        src: ``(B, N)`` scores.
        index: ``(B, N)`` int group assignments in ``[0, num_groups)``.
        num_groups: Total number of groups.

    Returns:
        ``(B, N)`` softmax-normalised within each group.

    Equivalent to ``torch_scatter.scatter_softmax(src, index, dim=1)``.
    """
    batch_size, _ = src.shape
    idx = index.long()

    group_max = torch.full(
        (batch_size, num_groups), float("-inf"), device=src.device, dtype=src.dtype
    )
    group_max.scatter_reduce_(1, idx, src, reduce="amax", include_self=True)
    element_max = group_max.gather(1, idx)

    exp_src = torch.exp(src - element_max)

    group_sum = torch.zeros(batch_size, num_groups, device=src.device, dtype=src.dtype)
    group_sum.scatter_add_(1, idx, exp_src)
    element_sum = group_sum.gather(1, idx)

    return exp_src / (element_sum + 1e-10)


def scatter_sum_batched(
    src: torch.Tensor, index: torch.Tensor, dim_size: int
) -> torch.Tensor:
    """Grouped sum with batched, per-row group indices (scatters along dim 1).

    Args:
        src: ``(B, N, D)`` values.
        index: ``(B, N)`` int group assignments in ``[0, dim_size)``.
        dim_size: Output size along the scatter dimension.

    Returns:
        ``(B, dim_size, D)`` summed values per group.

    Equivalent to ``torch_scatter.scatter_sum(src, index, dim=1, dim_size=dim_size)``.
    """
    batch_size, _, feat_dim = src.shape
    out = torch.zeros(
        batch_size, dim_size, feat_dim, device=src.device, dtype=src.dtype
    )
    idx = index.unsqueeze(-1).expand_as(src)
    return out.scatter_add_(1, idx, src)
