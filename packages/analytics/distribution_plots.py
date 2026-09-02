# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Matplotlib plotting for embedding distribution analysis."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from packages.analytics.embedding_distribution import (
    DistanceStats,
    ProjectionResult,
    summarize_distance_stats,
)

PathLike = str | Path


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_distance_histograms(
    stats: DistanceStats,
    output_path: PathLike,
    title: str = "Embedding distance distributions (S vs B)",
) -> Path:
    """Three-panel histogram: S→B top1, within-S vs S→B, density overlay."""
    output_path = Path(output_path)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(
        stats.s_to_b_top1,
        bins=min(20, max(5, len(stats.s_to_b_top1))),
        color="#1F8A65",
        alpha=0.85,
        edgecolor="white",
    )
    axes[0].axvline(
        stats.s_to_b_top1.mean(),
        color="black",
        ls="--",
        lw=1,
        label=f"mean={stats.s_to_b_top1.mean():.3f}",
    )
    axes[0].set_title("Each S: top-1 cosine to B")
    axes[0].set_xlabel("cosine similarity")
    axes[0].set_ylabel("count")
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].hist(
        stats.within_s_nn,
        bins=min(15, max(5, len(stats.within_s_nn))),
        color="#5A6CC0",
        alpha=0.75,
        edgecolor="white",
        label="S→nearest other S",
    )
    axes[1].hist(
        stats.s_to_b_top1,
        bins=min(15, max(5, len(stats.s_to_b_top1))),
        color="#1F8A65",
        alpha=0.55,
        edgecolor="white",
        label="S→nearest B",
    )
    axes[1].set_title("Within-S NN vs S→B NN")
    axes[1].set_xlabel("cosine similarity")
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle(title, y=1.02)
    return _save(fig, output_path)


def plot_centroid_scatter(
    stats: DistanceStats,
    output_path: PathLike,
    title: str = "Centroid similarity (S vs B)",
) -> Path:
    output_path = Path(output_path)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].scatter(
        stats.b_to_s_centroid, stats.b_to_b_centroid, s=8, c="#70B0D8", alpha=0.35, label="B"
    )
    axes[0].scatter(
        stats.s_to_s_centroid,
        stats.s_to_b_centroid,
        s=60,
        c="#1F8A65",
        edgecolors="black",
        linewidths=0.5,
        label="S",
        zorder=5,
    )
    axes[0].set_xlabel("cosine to S centroid")
    axes[0].set_ylabel("cosine to B centroid")
    axes[0].set_title("Prototype / centroid similarity")
    axes[0].legend(frameon=False, fontsize=8)

    margin_b = stats.b_to_s_centroid - stats.b_to_b_centroid
    margin_s = stats.s_to_s_centroid - stats.s_to_b_centroid
    axes[1].hist(margin_b, bins=40, color="#70B0D8", alpha=0.6, density=True, label="B")
    axes[1].hist(
        margin_s,
        bins=min(12, max(4, len(margin_s))),
        color="#1F8A65",
        alpha=0.75,
        density=True,
        label="S",
    )
    axes[1].axvline(0, color="black", lw=0.8)
    axes[1].set_xlabel("cos(S_cent) − cos(B_cent)")
    axes[1].set_title("Preference for S vs B centroid")
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle(title, y=1.02)
    return _save(fig, output_path)


def plot_projection_s_vs_b(
    projection: ProjectionResult,
    output_path: PathLike,
    title: str | None = None,
    dims: tuple[int, int] = (0, 1),
) -> Path:
    """Scatter S vs B on a 2D projection."""
    output_path = Path(output_path)
    i, j = dims
    coords = projection.coords
    n_s = projection.n_s
    fig, ax = plt.subplots(figsize=(8, 6.5))
    ax.scatter(
        coords[n_s:, i],
        coords[n_s:, j],
        s=8,
        c="#70B0D8",
        alpha=0.35,
        label=f"B (n={coords.shape[0] - n_s})",
    )
    ax.scatter(
        coords[:n_s, i],
        coords[:n_s, j],
        s=55,
        c="#1F8A65",
        edgecolors="black",
        linewidths=0.5,
        label=f"S (n={n_s})",
        zorder=5,
    )
    xlab = f"{projection.method.upper()}-{i + 1}"
    ylab = f"{projection.method.upper()}-{j + 1}"
    if projection.explained_variance_pct and i < len(projection.explained_variance_pct):
        xlab = f"PC{i + 1} ({projection.explained_variance_pct[i]:.1f}%)"
        ylab = f"PC{j + 1} ({projection.explained_variance_pct[j]:.1f}%)"
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_title(title or f"{projection.method.upper()}: S vs B")
    ax.legend(frameon=False, fontsize=8)
    return _save(fig, output_path)


def plot_text_similarity(
    s_scores: np.ndarray,
    b_scores: np.ndarray,
    output_path: PathLike,
    xlabel: str = "cosine to text query",
    title: str = "Video ↔ text similarity",
) -> Path:
    output_path = Path(output_path)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(b_scores, bins=40, color="#C06028", alpha=0.55, density=True, label="B")
    ax.hist(
        s_scores,
        bins=min(12, max(4, len(s_scores))),
        color="#1F8A65",
        alpha=0.8,
        density=True,
        label="S",
    )
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8)
    return _save(fig, output_path)


def write_summary_json(
    path: PathLike,
    stats: DistanceStats,
    extra: dict | None = None,
) -> Path:
    import json

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = summarize_distance_stats(stats)
    if extra:
        payload.update(extra)
    path.write_text(json.dumps(payload, indent=2))
    return path
