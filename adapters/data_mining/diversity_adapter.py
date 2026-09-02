# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Data Mining diversity adapter — farthest-point sampling.

Pure torch/numpy greedy max-min selection. Cluster seeds optional.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from packages.domain.types import EmbeddingRecord, SubsetSelection


class DataMiningDiversityAdapter:
    """Implements SubsetSelectionPort with farthest-point semantics."""

    def select_diverse(
        self,
        records: Sequence[EmbeddingRecord],
        n_samples: int,
        cluster_labels: Sequence[int] | None = None,
    ) -> SubsetSelection:
        if not records:
            return SubsetSelection(record_ids=[], strategy="farthest_point")
        if n_samples <= 0:
            raise ValueError("n_samples must be positive")
        n_samples = min(n_samples, len(records))

        matrix = np.asarray([list(r.embedding) for r in records], dtype=np.float64)
        ids = self._seed_ids(cluster_labels, len(records))
        selected = self._farthest_point(matrix, ids, n_samples)
        return SubsetSelection(
            record_ids=[records[i].record_id for i in selected],
            strategy="farthest_point",
            params={"n_samples": n_samples, "seeded_by_clusters": cluster_labels is not None},
        )

    @staticmethod
    def _seed_ids(cluster_labels: Sequence[int] | None, n: int) -> list[int]:
        if cluster_labels is None:
            return [0]
        labels = np.asarray(cluster_labels)
        if len(labels) != n:
            raise ValueError("cluster_labels length must match records")
        # Largest clusters first — same ordering idea as TAO diversity_sampling
        unique = np.unique(labels)
        # Sort unique labels by descending count
        order = sorted(unique, key=lambda lab: -int(np.sum(labels == lab)))
        rng = np.random.default_rng(seed=0)
        seeds: list[int] = []
        for lab in order:
            candidates = np.where(labels == lab)[0]
            seeds.append(int(rng.choice(candidates)))
        return seeds or [0]

    @staticmethod
    def _farthest_point(matrix: np.ndarray, seed_ids: list[int], n_samples: int) -> list[int]:
        selected = list(dict.fromkeys(seed_ids))  # preserve order, unique
        if len(selected) >= n_samples:
            return selected[:n_samples]

        # min squared distance to selected set
        min_dist = None
        for _ in range(len(selected), n_samples):
            current = matrix[selected[-1]]
            new_dist = ((matrix - current) ** 2).sum(axis=1)
            if min_dist is None:
                min_dist = new_dist
            else:
                min_dist = np.minimum(min_dist, new_dist)
            # mask already selected
            masked = min_dist.copy()
            masked[selected] = -1.0
            farthest = int(np.argmax(masked))
            selected.append(farthest)
        return selected
