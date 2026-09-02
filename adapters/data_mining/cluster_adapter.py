# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Data Mining clustering adapter — thin wrapper around clustering libraries.

Does not change algorithm behavior; only maps EmbeddingRecord ↔ numpy/cuDF inputs.
When RAPIDS is unavailable, falls back to sklearn KMeans with the same n_clusters.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from packages.domain.types import ClusterAssignment, ClusterResult, EmbeddingRecord


class DataMiningClusterAdapter:
    """Implements ClusterAnalyticsPort."""

    def fit_predict(
        self,
        records: Sequence[EmbeddingRecord],
        method: str = "kmeans",
        n_clusters: int = 8,
    ) -> ClusterResult:
        if not records:
            return ClusterResult(assignments=[], method=method)

        matrix = np.asarray([list(r.embedding) for r in records], dtype=np.float64)
        labels = self._fit_predict_labels(matrix, method, n_clusters)
        assignments = [
            ClusterAssignment(record_id=records[i].record_id, label=int(labels[i]))
            for i in range(len(records))
        ]
        return ClusterResult(assignments=assignments, method=method)

    def _fit_predict_labels(
        self,
        matrix: np.ndarray,
        method: str,
        n_clusters: int,
    ) -> np.ndarray:
        method_l = method.lower()
        if method_l != "kmeans":
            # Other methods require cuML; keep explicit rather than silent wrong algo.
            try:
                return self._cuml_fit_predict(matrix, method_l, n_clusters)
            except ImportError as exc:
                raise ValueError(
                    f"method={method!r} requires cuML; only kmeans has a CPU fallback"
                ) from exc

        try:
            return self._cuml_fit_predict(matrix, "kmeans", n_clusters)
        except ImportError:
            from sklearn.cluster import KMeans

            model = KMeans(n_clusters=min(n_clusters, len(matrix)), random_state=0, n_init=10)
            return np.asarray(model.fit_predict(matrix), dtype=np.int64)

    def _cuml_fit_predict(self, matrix: np.ndarray, method: str, n_clusters: int) -> np.ndarray:
        import cudf

        # Use the stable cuML API because TAO clustering signatures vary by release.
        from cuml.cluster import KMeans as CumlKMeans

        if method != "kmeans":
            raise ImportError(f"Adapter CPU/cuML path only wires kmeans; got {method}")
        gdf = cudf.DataFrame(matrix)
        model = CumlKMeans(n_clusters=min(n_clusters, len(matrix)), random_state=0)
        return np.asarray(model.fit_predict(gdf).to_numpy())
