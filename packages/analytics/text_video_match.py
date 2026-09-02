# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Text↔video matching over precomputed video embeddings.

Scores each gallery video by max (or mean) cosine similarity to one or more
text query embeddings, then ranks / thresholds for selection.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from packages.analytics.embeddings.vectors import cosine_scores, l2_normalize

ScoreReduce = Literal["max", "mean"]


@dataclass(frozen=True)
class TextVideoMatchResult:
    """Ranked gallery scores against text queries."""

    file_names: list[str]
    scores: np.ndarray
    per_query_scores: dict[str, np.ndarray] = field(default_factory=dict)
    threshold: float | None = None
    matched_indices: list[int] = field(default_factory=list)

    @property
    def matched_file_names(self) -> list[str]:
        return [self.file_names[i] for i in self.matched_indices]


def score_gallery_against_texts(
    gallery_embeddings: np.ndarray,
    text_embeddings: Sequence[np.ndarray],
    query_labels: Sequence[str] | None = None,
    reduce: ScoreReduce = "max",
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Score each gallery row against one or more text vectors.

    Parameters
    ----------
    gallery_embeddings:
        ``(N, D)`` video embeddings (will be L2-normalized).
    text_embeddings:
        Sequence of ``(D,)`` or ``(1, D)`` text embeddings.
    query_labels:
        Optional names for each text query (defaults to ``q1``, ``q2``, …).
    reduce:
        ``max`` or ``mean`` across text queries per video.
    """
    if len(text_embeddings) == 0:
        raise ValueError("At least one text embedding is required")
    gallery = l2_normalize(np.asarray(gallery_embeddings, dtype=np.float64))
    labels = (
        list(query_labels)
        if query_labels is not None
        else [f"q{i + 1}" for i in range(len(text_embeddings))]
    )
    if len(labels) != len(text_embeddings):
        raise ValueError("query_labels length must match text_embeddings")

    per_query: dict[str, np.ndarray] = {}
    stacked = []
    for label, te in zip(labels, text_embeddings, strict=True):
        t = np.asarray(te, dtype=np.float64).reshape(1, -1)
        sims = cosine_scores(t, gallery)[0]
        per_query[label] = sims
        stacked.append(sims)

    mat = np.stack(stacked, axis=0)  # (Q, N)
    if reduce == "max":
        scores = mat.max(axis=0)
    elif reduce == "mean":
        scores = mat.mean(axis=0)
    else:
        raise ValueError(f"Unknown reduce={reduce!r}")
    return scores, per_query


def mean_std_threshold(scores: np.ndarray, k_std: float = 1.0) -> float:
    """Threshold at ``mean + k_std * std`` (population std)."""
    scores = np.asarray(scores, dtype=np.float64)
    if scores.size == 0:
        raise ValueError("scores must be non-empty")
    return float(scores.mean() + k_std * scores.std())


def select_by_threshold(
    file_names: Sequence[str],
    scores: np.ndarray,
    threshold: float,
) -> list[int]:
    """Return indices with score >= threshold, sorted by score descending."""
    scores = np.asarray(scores, dtype=np.float64)
    if len(file_names) != len(scores):
        raise ValueError("file_names and scores length mismatch")
    idx = np.where(scores >= threshold)[0]
    order = idx[np.argsort(-scores[idx])]
    return [int(i) for i in order]


def select_top_k(
    file_names: Sequence[str],
    scores: np.ndarray,
    top_k: int,
) -> list[int]:
    """Return indices of the top-k scores (descending)."""
    scores = np.asarray(scores, dtype=np.float64)
    if len(file_names) != len(scores):
        raise ValueError("file_names and scores length mismatch")
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    k = min(top_k, len(scores))
    order = np.argsort(-scores)[:k]
    return [int(i) for i in order]


def match_text_to_videos(
    file_names: Sequence[str],
    gallery_embeddings: np.ndarray,
    text_embeddings: Sequence[np.ndarray],
    query_labels: Sequence[str] | None = None,
    reduce: ScoreReduce = "max",
    mode: Literal["threshold", "top_k"] = "threshold",
    k_std: float = 1.0,
    top_k: int = 50,
    threshold: float | None = None,
) -> TextVideoMatchResult:
    """End-to-end text→video match: score, then threshold or top-k."""
    scores, per_query = score_gallery_against_texts(
        gallery_embeddings,
        text_embeddings,
        query_labels=query_labels,
        reduce=reduce,
    )
    names = [str(n) for n in file_names]
    if mode == "threshold":
        thr = float(threshold) if threshold is not None else mean_std_threshold(scores, k_std)
        matched = select_by_threshold(names, scores, thr)
    elif mode == "top_k":
        thr = None
        matched = select_top_k(names, scores, top_k)
    else:
        raise ValueError(f"Unknown mode={mode!r}")
    return TextVideoMatchResult(
        file_names=names,
        scores=scores,
        per_query_scores=per_query,
        threshold=thr,
        matched_indices=matched,
    )
