# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Data Mining DivKNN adapter — wraps uniqueness domain logic + optional GPU kNN.

When cuML is unavailable, callers can supply precomputed neighbor rows
(from an external kNN) and still get identical uniqueness behavior.
When cuML is available and ``use_gpu_knn=True``, delegates to the same
NearestNeighbors path as ``tao_data_curation.DivKNN.div_knn``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from packages.analytics.uniqueness.divknn_uniqueness import (
    apply_uniqueness_pipeline,
    unique_source_ids_from_frame,
)
from packages.domain.types import BatchSimilarityResult, EmbeddingRecord, NeighborMatch


class DataMiningDivKnnAdapter:
    """Implements BatchSimilarityPort with Data Mining DivKNN semantics."""

    def __init__(self, use_gpu_knn: bool = False) -> None:
        self._use_gpu_knn = use_gpu_knn

    def select_similar(
        self,
        targets: Sequence[EmbeddingRecord],
        sources: Sequence[EmbeddingRecord],
        top_n: int = 5,
        backup_candidates: int = 15,
        metric: str = "cosine",
        apply_uniqueness: bool = True,
    ) -> BatchSimilarityResult:
        total_n = top_n + backup_candidates
        if total_n < 1:
            raise ValueError("top_n + backup_candidates must be >= 1")
        if not sources or not targets:
            return BatchSimilarityResult(matches=[], unique_source_ids=[])

        if self._use_gpu_knn:
            neighbor_rows = self._gpu_knn_rows(targets, sources, total_n, metric)
        else:
            neighbor_rows = self._numpy_knn_rows(targets, sources, total_n, metric)

        if apply_uniqueness:
            if top_n >= total_n:
                raise ValueError("top_n must be less than total_n when uniqueness is on")
            df = apply_uniqueness_pipeline(neighbor_rows, total_n, top_n)
        else:
            df = pd.DataFrame(neighbor_rows)

        matches: list[NeighborMatch] = []
        for _, row in df.iterrows():
            target_id = str(row["target_file_name"])
            for rank in range(1, top_n + 1):
                src = row.get(f"top{rank}_file_name")
                if src is None or (isinstance(src, float) and np.isnan(src)):
                    continue
                dist = row.get(f"top{rank}_distance", 0.0)
                matches.append(
                    NeighborMatch(
                        target_id=target_id,
                        source_id=str(src),
                        distance=float(dist),
                        rank=rank,
                    )
                )

        unique_ids = unique_source_ids_from_frame(df, top_n)
        return BatchSimilarityResult(matches=matches, unique_source_ids=unique_ids)

    def _numpy_knn_rows(
        self,
        targets: Sequence[EmbeddingRecord],
        sources: Sequence[EmbeddingRecord],
        total_n: int,
        metric: str,
    ) -> list[dict[str, Any]]:
        """CPU fallback with the same L2-normalize + distance ranking as DivKNN."""
        src_ids = [s.record_id for s in sources]
        src_mat = self._l2_normalize(
            np.asarray([list(s.embedding) for s in sources], dtype=np.float64)
        )
        tgt_mat = self._l2_normalize(
            np.asarray([list(t.embedding) for t in targets], dtype=np.float64)
        )

        if metric == "cosine":
            # After L2-norm, cosine distance = 1 - dot
            sims = tgt_mat @ src_mat.T
            distances = 1.0 - sims
        elif metric == "euclidean":
            # ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a·b; with unit norms → 2-2dot
            distances = np.sqrt(np.maximum(0.0, 2.0 - 2.0 * (tgt_mat @ src_mat.T)))
        else:
            raise ValueError(f"Unsupported metric for CPU path: {metric}")

        k = min(total_n, src_mat.shape[0])
        rows: list[dict[str, Any]] = []
        for i, target in enumerate(targets):
            order = np.argpartition(distances[i], kth=k - 1)[:k]
            order = order[np.argsort(distances[i, order])]
            row: dict[str, Any] = {
                "target_file_name": target.record_id,
                "target_embedding": list(target.embedding),
            }
            for rank, idx in enumerate(order, start=1):
                row[f"top{rank}_file_name"] = src_ids[int(idx)]
                row[f"top{rank}_distance"] = float(distances[i, idx])
                row[f"top{rank}_embed"] = list(sources[int(idx)].embedding)
            for rank in range(len(order) + 1, total_n + 1):
                row[f"top{rank}_file_name"] = None
                row[f"top{rank}_distance"] = float("nan")
                row[f"top{rank}_embed"] = None
            rows.append(row)
        return rows

    def _gpu_knn_rows(
        self,
        targets: Sequence[EmbeddingRecord],
        sources: Sequence[EmbeddingRecord],
        total_n: int,
        metric: str,
    ) -> list[dict[str, Any]]:
        """Delegate to cuML NearestNeighbors — same as Data Mining DivKNN main path."""
        try:
            import cuml.neighbors
            import cupy as cp
        except ImportError as exc:
            raise ImportError(
                "cuML/cupy required for use_gpu_knn=True; install RAPIDS or use CPU path"
            ) from exc

        src_ids = [s.record_id for s in sources]
        src_mat = cp.asarray([list(s.embedding) for s in sources], dtype=cp.float32)
        tgt_mat = cp.asarray([list(t.embedding) for t in targets], dtype=cp.float32)
        src_mat = self._l2_normalize_cupy(src_mat)
        tgt_mat = self._l2_normalize_cupy(tgt_mat)

        knn = cuml.neighbors.NearestNeighbors(n_neighbors=total_n, metric=metric)
        knn.fit(src_mat)
        distances, indices = knn.kneighbors(tgt_mat)
        distances_host = cp.asnumpy(distances)
        indices_host = cp.asnumpy(indices)

        rows: list[dict[str, Any]] = []
        for i, target in enumerate(targets):
            row: dict[str, Any] = {
                "target_file_name": target.record_id,
                "target_embedding": list(target.embedding),
            }
            for rank in range(total_n):
                idx = int(indices_host[i, rank])
                row[f"top{rank + 1}_file_name"] = src_ids[idx]
                row[f"top{rank + 1}_distance"] = float(distances_host[i, rank])
                row[f"top{rank + 1}_embed"] = list(sources[idx].embedding)
            rows.append(row)
        return rows

    @staticmethod
    def _l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        norms = np.maximum(norms, eps)
        return np.asarray(x / norms, dtype=np.float64)

    @staticmethod
    def _l2_normalize_cupy(x: Any, eps: float = 1e-12) -> Any:
        import cupy as cp

        norms = cp.linalg.norm(x, axis=1, keepdims=True)
        norms = cp.maximum(norms, eps)
        return x / norms
