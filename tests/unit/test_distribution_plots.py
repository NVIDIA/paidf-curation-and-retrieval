# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for distribution plot writers (matplotlib Agg)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from packages.analytics.distribution_plots import (
    plot_centroid_scatter,
    plot_distance_histograms,
    plot_projection_s_vs_b,
    plot_text_similarity,
    write_summary_json,
)
from packages.analytics.embedding_distribution import (
    DistanceStats,
    compute_distance_stats,
    project_pca,
)


def _stats() -> DistanceStats:
    rng = np.random.default_rng(0)
    xs = rng.normal(size=(8, 4))
    xb = rng.normal(size=(30, 4))
    return compute_distance_stats(xs, xb)


class TestDistributionPlots:
    def test_distance_and_centroid_plots(self, tmp_path: Path):
        stats = _stats()
        p1 = plot_distance_histograms(stats, tmp_path / "dist.png")
        p2 = plot_centroid_scatter(stats, tmp_path / "cent.png")
        assert p1.exists() and p1.stat().st_size > 0
        assert p2.exists() and p2.stat().st_size > 0

    def test_projection_and_text_plots(self, tmp_path: Path):
        rng = np.random.default_rng(1)
        xs = rng.normal(size=(6, 5))
        xb = rng.normal(size=(25, 5))
        proj = project_pca(xs, xb, n_components=2)
        p = plot_projection_s_vs_b(proj, tmp_path / "pca.png")
        assert p.exists()
        p2 = plot_text_similarity(
            rng.random(6),
            rng.random(25),
            tmp_path / "text.png",
        )
        assert p2.exists()

    def test_summary_json(self, tmp_path: Path):
        stats = _stats()
        path = write_summary_json(tmp_path / "summary.json", stats, extra={"n_S": 8})
        assert path.exists()
        text = path.read_text()
        assert "s_to_b_top1_mean" in text
        assert "n_S" in text
