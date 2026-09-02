# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Embedding-space distribution analysis (pure numerics, no plotting)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA

from packages.analytics.embeddings.vectors import cosine_scores, l2_normalize


@dataclass(frozen=True)
class DistanceStats:
    """Nearest-neighbor cosine stats for set S against gallery B."""

    s_to_b_top1: np.ndarray
    within_s_nn: np.ndarray
    s_to_s_centroid: np.ndarray
    s_to_b_centroid: np.ndarray
    b_to_s_centroid: np.ndarray
    b_to_b_centroid: np.ndarray


@dataclass(frozen=True)
class ProjectionResult:
    """2D/3D projection of stacked [S; B] embeddings."""

    coords: np.ndarray
    n_s: int
    method: str
    explained_variance_pct: list[float] | None = None


def compute_distance_stats(
    embeddings_s: np.ndarray,
    embeddings_b: np.ndarray,
) -> DistanceStats:
    """Compute S↔B and centroid cosine statistics."""
    xs = l2_normalize(np.asarray(embeddings_s, dtype=np.float64))
    xb = l2_normalize(np.asarray(embeddings_b, dtype=np.float64))
    if xs.ndim != 2 or xb.ndim != 2:
        raise ValueError("embeddings must be 2D")
    if xs.shape[0] == 0 or xb.shape[0] == 0:
        raise ValueError("S and B must be non-empty")
    if xs.shape[1] != xb.shape[1]:
        raise ValueError("S and B embedding dims must match")

    sims_sb = cosine_scores(xs, xb)
    s_to_b_top1 = sims_sb.max(axis=1)

    sims_ss = cosine_scores(xs, xs).copy()
    np.fill_diagonal(sims_ss, -np.inf)
    within_s_nn = sims_ss.max(axis=1)

    cent_s = l2_normalize(xs.mean(axis=0, keepdims=True))[0]
    cent_b = l2_normalize(xb.mean(axis=0, keepdims=True))[0]
    return DistanceStats(
        s_to_b_top1=s_to_b_top1,
        within_s_nn=within_s_nn,
        s_to_s_centroid=xs @ cent_s,
        s_to_b_centroid=xs @ cent_b,
        b_to_s_centroid=xb @ cent_s,
        b_to_b_centroid=xb @ cent_b,
    )


def project_pca(
    embeddings_s: np.ndarray,
    embeddings_b: np.ndarray,
    n_components: int = 3,
    random_state: int = 42,
) -> ProjectionResult:
    """PCA on stacked [S; B] (sklearn)."""
    xs = l2_normalize(np.asarray(embeddings_s, dtype=np.float64))
    xb = l2_normalize(np.asarray(embeddings_b, dtype=np.float64))
    x_all = np.vstack([xs, xb])
    n_comp = min(n_components, x_all.shape[0], x_all.shape[1])
    pca = PCA(n_components=n_comp, random_state=random_state)
    coords = pca.fit_transform(x_all)
    expl = (pca.explained_variance_ratio_ * 100.0).tolist()
    return ProjectionResult(
        coords=coords,
        n_s=xs.shape[0],
        method="pca",
        explained_variance_pct=expl,
    )


def project_tsne(
    embeddings_s: np.ndarray,
    embeddings_b: np.ndarray,
    perplexity: float = 30.0,
    random_state: int = 42,
) -> ProjectionResult:
    """t-SNE on stacked [S; B] (sklearn)."""
    from sklearn.manifold import TSNE

    xs = l2_normalize(np.asarray(embeddings_s, dtype=np.float64))
    xb = l2_normalize(np.asarray(embeddings_b, dtype=np.float64))
    x_all = np.vstack([xs, xb])
    n = x_all.shape[0]
    perp = min(perplexity, max(5.0, (n - 1) / 3.0))
    tsne = TSNE(
        n_components=2,
        perplexity=perp,
        metric="euclidean",
        random_state=random_state,
        init="pca",
    )
    coords = tsne.fit_transform(x_all)
    return ProjectionResult(coords=coords, n_s=xs.shape[0], method="tsne")


def project_umap(
    embeddings_s: np.ndarray,
    embeddings_b: np.ndarray,
    n_neighbors: int = 30,
    min_dist: float = 0.1,
    random_state: int = 42,
) -> ProjectionResult:
    """UMAP on stacked [S; B] (optional dependency)."""
    try:
        import umap
    except ImportError as exc:
        raise ImportError(
            "umap-learn is required for UMAP projections; poetry add umap-learn"
        ) from exc

    xs = l2_normalize(np.asarray(embeddings_s, dtype=np.float64))
    xb = l2_normalize(np.asarray(embeddings_b, dtype=np.float64))
    x_all = np.vstack([xs, xb])
    n_neighbors = min(n_neighbors, max(2, x_all.shape[0] - 1))
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="cosine",
        random_state=random_state,
    )
    coords = reducer.fit_transform(x_all)
    return ProjectionResult(coords=coords, n_s=xs.shape[0], method="umap")


def summarize_distance_stats(stats: DistanceStats) -> dict[str, float]:
    """Compact JSON-serializable summary of distance stats."""
    return {
        "s_to_b_top1_mean": float(stats.s_to_b_top1.mean()),
        "s_to_b_top1_min": float(stats.s_to_b_top1.min()),
        "s_to_b_top1_max": float(stats.s_to_b_top1.max()),
        "within_s_nn_mean": float(stats.within_s_nn.mean()),
        "within_s_nn_min": float(stats.within_s_nn.min()),
        "within_s_nn_max": float(stats.within_s_nn.max()),
        "s_to_s_centroid_mean": float(stats.s_to_s_centroid.mean()),
        "s_to_b_centroid_mean": float(stats.s_to_b_centroid.mean()),
        "b_to_s_centroid_mean": float(stats.b_to_s_centroid.mean()),
    }


def text_similarity_to_gallery(
    gallery_embeddings: np.ndarray,
    text_embedding: np.ndarray,
) -> np.ndarray:
    """Cosine of each gallery row to a single text vector."""
    similarities = cosine_scores(
        np.asarray(text_embedding, dtype=np.float64).reshape(1, -1),
        gallery_embeddings,
    )
    return np.asarray(similarities[0], dtype=np.float64)
