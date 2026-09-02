# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DivKNN-style uniqueness selection over embedding matrices (CPU)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from packages.analytics.embeddings.vectors import l2_normalize


def _ensure_top_n_unique(
    row_files: list[str],
    row_dists: list[float],
    top_n: int,
) -> list[int]:
    seen = set()
    keep: list[int] = []
    for idx in np.argsort(row_dists):
        name = row_files[int(idx)]
        if name in seen:
            continue
        seen.add(name)
        keep.append(int(idx))
        if len(keep) >= top_n:
            break
    return keep


def _global_unique(candidates: list[tuple[int, str, float, int]], top_n: int) -> list[str]:
    best: dict = {}
    for t_idx, src, dist, rank in candidates:
        prev = best.get(src)
        if prev is None or dist < prev[0]:
            best[src] = (dist, t_idx, rank)

    by_target: dict = {}
    for src, (dist, t_idx, _rank) in best.items():
        by_target.setdefault(t_idx, []).append((dist, src))

    selected: list[str] = []
    selected_set = set()
    for t_idx in sorted(by_target.keys()):
        items = sorted(by_target[t_idx], key=lambda x: x[0])
        for _dist, src in items[:top_n]:
            if src not in selected_set:
                selected_set.add(src)
                selected.append(src)
    return selected


def knn_unique_select(
    target_names: Sequence[str],
    target_emb: np.ndarray,
    source_names: Sequence[str],
    source_emb: np.ndarray,
    top_n: int = 3,
    backup: int = 15,
) -> list[str]:
    """Target-guided kNN with row + global uniqueness (DivKNN semantics).

    Uses Euclidean distance on L2-normalized vectors (monotonic with cosine).
    """
    if top_n < 1:
        raise ValueError("top_n must be >= 1")
    t = l2_normalize(np.asarray(target_emb, dtype=np.float32))
    s = l2_normalize(np.asarray(source_emb, dtype=np.float32))
    source_names = [str(n) for n in source_names]
    n_src = s.shape[0]
    total_n = min(top_n + backup, n_src)
    if total_n < 1:
        return []

    candidates: list[tuple[int, str, float, int]] = []
    chunk = 64
    for start in range(0, t.shape[0], chunk):
        end = min(start + chunk, t.shape[0])
        sims = t[start:end] @ s.T
        dists = np.sqrt(np.maximum(0.0, 2.0 - 2.0 * sims))
        for local_i, global_i in enumerate(range(start, end)):
            order = np.argsort(dists[local_i])[:total_n]
            row_files = [source_names[j] for j in order]
            row_dists = [float(dists[local_i, j]) for j in order]
            keep = _ensure_top_n_unique(row_files, row_dists, top_n)
            for rank, idx in enumerate(keep, start=1):
                candidates.append((global_i, row_files[idx], row_dists[idx], rank))
    return _global_unique(candidates, top_n)
