import numpy as np
from omegaconf import DictConfig


class ImpressionSampler:
    """Handles sampling strategies for news impressions."""

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.max_length = cfg.max_impressions_length
        self.strategy = cfg.strategy
        np.random.seed(cfg.random_seed)

    def sample_candidates_news(
        self,
        stage: str,
        candidates: list[str],
        parse_news_id,
        random_train_samples: bool = False,
        news_info: dict | None = None,
        timestamp: str | None = None,
        available_news_ids: set | None = None,
    ) -> tuple[list[list[int]], list[list[int]]]:
        """Sample candidate news with fixed ratio of positive to negative samples.

        Args:
            candidates: List of candidtes news strings in format "<news_id>-<label>"
            parse_news_id: parser news id to int
            news_info: Optional dictionary containing news metadata
            timestamp: Optional timestamp for temporal sampling
        Returns:
            Tuple containing:
            - List of candidates news groups, each with 1 positive and k negatives (training)
            - List of label groups corresponding to each candidate news group (indicating if the news is clicked or not)
        """
        # Split impressions into positives and negatives
        positives = []
        negatives = []
        for can_news in candidates:
            news_id, label = can_news.split("-")
            news_id = parse_news_id(news_id)

            # Filter out news IDs that don't exist in the news data
            if available_news_ids is not None and news_id not in available_news_ids:
                continue

            if int(label) == 1:
                positives.append(news_id)
            else:
                negatives.append(news_id)

        if stage == "val" or stage == "test":
            # For validation/testing, return all impressions in a single group
            # Combine and shuffle all impressions
            all_impressions = positives + negatives
            all_labels = [1] * len(positives) + [0] * len(negatives)

            # Shuffle together to avoid order bias
            combined = list(zip(all_impressions, all_labels))
            np.random.shuffle(combined)
            all_impressions, all_labels = zip(*combined)

            return list(all_impressions), list(all_labels)

        # Training sampling strategy
        k = self.max_length - 1  # number of negatives per positive
        all_samples = []
        all_labels = []

        for pos in positives:
            # Select negatives based on strategy
            neg_samples = self._sample_negatives(
                negatives, k, news_info=news_info, timestamp=timestamp
            )

            # Combine positive and negatives
            sample = [pos] + neg_samples
            labels = [1] + [0] * len(neg_samples)

            if random_train_samples:
                # Shuffle together
                combined = list(zip(sample, labels))
                np.random.shuffle(combined)
                sample, labels = zip(*combined)

            # Convert tuples to lists and add to groups
            all_samples.append(list(sample))
            all_labels.append(list(labels))

        return all_samples, all_labels

    def _sample_negatives(
        self,
        negatives: list[int],
        k: int,
        news_info: dict | None = None,
        timestamp: str | None = None,
    ) -> list[int]:
        """Sample k negatives using the configured strategy"""
        if self.strategy == "random":
            return self._random_sample_negatives(negatives, k)
        elif self.strategy == "popularity":
            return self._popularity_based_negatives(negatives, k, news_info)
        elif self.strategy == "temporal":
            return self._temporal_based_negatives(negatives, k, timestamp)
        elif self.strategy == "topic_diverse":
            return self._topic_diverse_negatives(negatives, k, news_info)
        else:
            return self._random_sample_negatives(negatives, k)

    def _random_sample_negatives(self, negatives: list[int], k: int) -> list[int]:
        """Random sampling strategy.

        Args:
            negatives: List of negative samples
            k: Number of samples to return

        Returns:
            List of sampled negative items
        """
        if k > len(negatives):
            # If we need more samples than available, repeat the list and sample
            repeated_negatives = negatives * (k // len(negatives) + 1)
            return list(np.random.choice(repeated_negatives, size=k, replace=False))
        else:
            return list(np.random.choice(negatives, size=k, replace=False))

    def _popularity_based_negatives(
        self, negatives: list[int], k: int, news_info: dict
    ) -> list[int]:
        """Sample negatives based on popularity"""
        if not news_info:
            return self._random_sample_negatives(negatives, k)

        # Get popularity scores
        popularity_scores = np.array(
            [news_info.get(nid, {}).get("popularity", 0) for nid in negatives]
        )

        # Normalize scores to probabilities
        if popularity_scores.sum() > 0:
            probs = popularity_scores / popularity_scores.sum()
        else:
            return self._random_sample_negatives(negatives, k)

        # Handle case where we need more samples than available
        if k > len(negatives):
            repeated_negatives = negatives * (k // len(negatives) + 1)
            repeated_probs = np.tile(probs, k // len(negatives) + 1)[:k]
            return list(
                np.random.choice(
                    repeated_negatives, size=k, replace=False, p=repeated_probs
                )
            )
        else:
            return list(np.random.choice(negatives, size=k, replace=False, p=probs))

    def _temporal_based_negatives(
        self, negatives: list[int], k: int, timestamp: str
    ) -> list[int]:
        """Sample negatives based on temporal proximity"""
        if not timestamp:
            return self._random_sample_negatives(negatives, k)

        # Implementation for temporal sampling
        # Could consider time difference between impression and news publication
        return self._random_sample_negatives(negatives, k)  # Placeholder

    def _topic_diverse_negatives(
        self, negatives: list[int], k: int, news_info: dict
    ) -> list[int]:
        """Sample negatives to ensure topic diversity"""
        if not news_info:
            return self._random_sample_negatives(negatives, k)

        # Get categories for negative samples
        categories = [
            news_info.get(nid, {}).get("category", "unknown") for nid in negatives
        ]
        unique_cats = list(set(categories))

        # Try to sample from different categories
        samples = []
        samples_per_cat = max(1, k // len(unique_cats))

        for cat in unique_cats:
            cat_negatives = [nid for nid, c in zip(negatives, categories) if c == cat]
            if cat_negatives:
                # Use the same sampling strategy as random sampling
                if samples_per_cat > len(cat_negatives):
                    repeated_negatives = cat_negatives * (
                        samples_per_cat // len(cat_negatives) + 1
                    )
                    samples.extend(
                        np.random.choice(
                            repeated_negatives,
                            size=min(samples_per_cat, len(repeated_negatives)),
                            replace=False,
                        )
                    )
                else:
                    samples.extend(
                        np.random.choice(
                            cat_negatives,
                            size=min(samples_per_cat, len(cat_negatives)),
                            replace=False,
                        )
                    )

        # Fill remaining slots randomly
        remaining = k - len(samples)
        if remaining > 0:
            remaining_samples = self._random_sample_negatives(
                [n for n in negatives if n not in samples], remaining
            )
            samples.extend(remaining_samples)

        return samples[:k]  # Ensure we return exactly k samples

    def _random_sample(self, impressions: list[int]) -> list[int]:
        """Random sampling strategy."""
        return np.random.choice(
            impressions, size=self.max_length, replace=self.cfg.random.replace
        ).tolist()

    def _topic_diverse_sample(
        self, impressions: list[int], news_info: dict
    ) -> list[int]:
        """Topic-diverse sampling strategy."""
        # Implementation for topic diversity
        categories = [news_info[imp]["category"] for imp in impressions]
        unique_cats = set(categories)

        # Ensure minimum category diversity
        sampled = []
        for cat in unique_cats:
            cat_impressions = [
                imp for imp, c in zip(impressions, categories) if c == cat
            ]
            weight = self.cfg.topic_diverse.category_weights.get(cat, 1.0)
            n_samples = max(1, int(self.max_length * weight / len(unique_cats)))
            if cat_impressions:
                sampled.extend(
                    np.random.choice(
                        cat_impressions, size=min(n_samples, len(cat_impressions))
                    )
                )

        # Fill remaining slots randomly
        if len(sampled) < self.max_length:
            remaining = [imp for imp in impressions if imp not in sampled]
            sampled.extend(
                np.random.choice(remaining, size=self.max_length - len(sampled))
            )

        return sampled[: self.max_length]

    # Add other sampling strategies as needed
