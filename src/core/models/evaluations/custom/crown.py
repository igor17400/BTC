"""CROWN evaluation with candidate-aware user attention.

Differs from :func:`fast_evaluate` because CROWN's user representation
depends on the candidate (reference repo
``seongeunryu/crown-www25/userEncoders.py:CROWN.forward`` uses
``Q = self.Q(candidate_news_representation)``). The standard
"precompute user vector once per impression" trick does **not** apply.

Two-stage design that keeps eval fast:

1. **Precompute news vectors** for every article (same as default).
2. **Precompute per-impression GCN-updated history reps** via
   :meth:`FrameworkAdapter.encode_crown_history_graph`. The expensive
   part — news encoder + 1-layer SAGE — runs ONCE per impression and is
   cached.
3. **Per-impression scoring**: gather candidate news vectors from cache,
   run candidate-aware attention + dot product via
   :meth:`FrameworkAdapter.run_crown_candidate_attention`. With cached
   ``gcn_news`` the per-impression call is essentially a few small mat-
   muls plus a softmax.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.core.io.progress import ProgressManager
from src.core.io.saving import save_predictions_to_file_fn
from src.core.models.adapter import FrameworkAdapter

from ..utils import compute_metrics, precompute_news_vectors


def crown_fast_evaluate(
    *,
    model: Any,
    news_dataloader: Any,
    user_hist_dataloader: Any,
    impression_iterator: Any,
    metrics_calculator: Any,
    progress: ProgressManager,
    adapter: FrameworkAdapter,
    behaviors_data: dict | None = None,
    int_to_news_id_map: dict[int, str] | None = None,
    save_predictions_path: str | None = None,
    epoch: int | None = None,
    mode: str = "val",
) -> dict[str, float]:
    """CROWN-specific candidate-aware fast evaluation.

    Args:
        model: Full CROWN model (needs ``news_encoder``, ``user_encoder``).
        news_dataloader: Yields ``{"news_id", "news_features"}``.
        user_hist_dataloader: Yields ``(imp_ids, user_ids, features)``
            where ``features`` is the packed history.
        impression_iterator: Yields ``(_, labels, imp_id, cand_ids)``.
        metrics_calculator: Object with ``METRIC_NAMES`` + ``compute_metrics``.
        progress: Progress bar manager.
        adapter: Framework adapter implementing
            :meth:`encode_crown_history_graph` and
            :meth:`run_crown_candidate_attention`.
        behaviors_data: Unused (signature parity with other custom evaluators).
        int_to_news_id_map: ``int -> "Nxxx"`` lookup for the news cache.
        save_predictions_path: Optional path for dumping predictions.
        epoch: Current epoch (for prediction filenames).
        mode: ``"val"`` or ``"test"``.

    Returns:
        ``{metric_name: value}`` dict (canonical numpy softmax-CE loss
        + ranking metrics from ``metrics_calculator``).
    """
    news_encoder = model.news_encoder
    user_encoder = model.user_encoder

    # 1. Precompute news vectors (full news_repr per article).
    news_vecs = precompute_news_vectors(
        news_encoder, news_dataloader, adapter, progress
    )

    # 2. Precompute per-impression GCN-updated history reps + masks.
    impression_state: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    hist_task = progress.add_task(
        "Precomputing CROWN history GNN reps...",
        total=len(user_hist_dataloader),
        visible=True,
    )
    for impression_ids, _user_ids, features in user_hist_dataloader:
        gcn_np, mask_np = adapter.encode_crown_history_graph(user_encoder, features)
        for i, imp_id in enumerate(impression_ids):
            impression_state[int(imp_id)] = (gcn_np[i], mask_np[i])
        progress.update(hist_task, advance=len(impression_ids))
    progress.remove_task(hist_task)

    # 3. Score every impression.
    #
    # Padding-to-max_C strategy: JAX-CROWN at PLM/A=128 OOM'd because
    # candidate counts vary per impression (5 to 100+ on MIND-small).
    # Every unique ``C`` triggered a fresh JIT compile of
    # ``candidate_attention``, accumulating dozens of cached binaries
    # on GPU until the next training-step allocation couldn't find
    # contiguous memory. Solution: compute the dataset-wide max C in
    # one pass over ``impression_iterator``, then pad every per-
    # impression cand_attention call to that same shape. JAX caches
    # one compiled binary for one shape; PT is unaffected.
    max_C = 0
    for impression in impression_iterator:
        cand_ids_np = adapter.to_numpy(impression[3])
        if cand_ids_np.shape[0] > max_C:
            max_C = int(cand_ids_np.shape[0])

    group_labels: list[np.ndarray] = []
    group_preds: list[np.ndarray] = []
    predictions_to_save: dict = {}

    imp_task = progress.add_task(
        "Scoring impressions (CROWN)...",
        total=len(impression_iterator),
        visible=True,
    )

    for impression in impression_iterator:
        _, labels, impression_id, cand_ids = impression
        labels_np = adapter.to_numpy(labels)

        state = impression_state.get(int(impression_id))
        if state is None:
            group_labels.append(labels_np)
            group_preds.append(np.array([]))
            progress.update(imp_task, advance=1)
            continue
        gcn_news_np, hist_mask_np = state

        cand_ids_np = adapter.to_numpy(cand_ids)
        cand_vecs: list[np.ndarray] = []
        for nid in cand_ids_np:
            if isinstance(nid, (str, np.str_)):
                key = str(nid)
            elif int_to_news_id_map and nid in int_to_news_id_map:
                key = int_to_news_id_map[nid]
            else:
                key = f"N{nid}"
            vec = news_vecs.get(key)
            if vec is not None:
                cand_vecs.append(vec)

        if not cand_vecs:
            scores = np.array([])
        else:
            cand_mat = np.stack(cand_vecs, axis=0).astype(np.float32)
            scores = adapter.run_crown_candidate_attention(
                user_encoder,
                gcn_news_np,
                hist_mask_np,
                cand_mat,
                max_C=max_C,
            )

        group_labels.append(labels_np)
        group_preds.append(scores)
        if save_predictions_path:
            predictions_to_save[str(cand_ids_np)] = (
                labels_np.tolist(),
                scores.tolist(),
            )
        progress.update(imp_task, advance=1)

    progress.remove_task(imp_task)

    final_metrics = compute_metrics(
        group_labels, group_preds, metrics_calculator, progress
    )
    final_metrics["_num_impressions"] = len(group_labels)

    if save_predictions_path:
        save_predictions_to_file_fn(
            predictions_to_save, save_predictions_path, epoch, mode
        )

    return final_metrics
