"""FastMIND dataset — a user-subset wrapper over MIND for fast iteration.

Samples a deterministic fraction of users from MIND-{small,large} and
restricts train+val behaviors to those users. Test split is left untouched
so eval metrics stay comparable to full-MIND runs. News, vocabulary and
popularity features are computed on the full dataset (the MIND files are
already downloaded — restricting users only changes which behaviors flow
into ``process_behaviors``).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from src.core.data.datasets.mind import MINDDataset

logger = logging.getLogger(__name__)


class FastMINDDataset(MINDDataset):
    """MIND with a deterministic user-level subset for train+val."""

    def __init__(
        self,
        *args,
        user_subset_ratio: float = 0.25,
        user_subset_seed: int = 42,
        **kwargs,
    ):
        if not 0.0 < user_subset_ratio <= 1.0:
            raise ValueError(
                f"user_subset_ratio must be in (0, 1]; got {user_subset_ratio}"
            )
        self.user_subset_ratio = float(user_subset_ratio)
        self.user_subset_seed = int(user_subset_seed)
        self._train_user_subset_cache: set[str] | None = None
        super().__init__(*args, **kwargs)

    def _get_train_user_subset(self) -> set[str] | None:
        if self._train_user_subset_cache is not None:
            return self._train_user_subset_cache
        if self.user_subset_ratio >= 1.0:
            return None

        train_path = self.dataset_path / "train" / "behaviors.tsv"
        valid_path = self.dataset_path / "valid" / "behaviors.tsv"
        col_names = ["impression_id", "user_id", "time", "history", "impressions"]

        user_ids: set[str] = set()
        for path in (train_path, valid_path):
            if not path.exists():
                continue
            df = pd.read_csv(
                path, sep="\t", header=None, names=col_names, usecols=["user_id"]
            )
            user_ids.update(df["user_id"].astype(str).unique().tolist())

        if not user_ids:
            logger.warning(
                "FastMIND: no users found in train/valid behaviors — falling "
                "back to full dataset"
            )
            return None

        sorted_users = sorted(user_ids)
        n_keep = max(1, int(len(sorted_users) * self.user_subset_ratio))
        rng = np.random.default_rng(self.user_subset_seed)
        chosen_idx = rng.choice(len(sorted_users), size=n_keep, replace=False)
        subset = {sorted_users[i] for i in chosen_idx}
        logger.info(
            f"FastMIND: sampled {n_keep:,}/{len(sorted_users):,} users "
            f"(ratio={self.user_subset_ratio:.3f}, seed={self.user_subset_seed})"
        )
        self._train_user_subset_cache = subset
        return subset

    def _user_subset_cache_signature(self) -> dict:
        return {
            "user_subset_ratio": round(self.user_subset_ratio, 6),
            "user_subset_seed": self.user_subset_seed,
        }
