# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for embedding distribution analytics."""

from __future__ import annotations

import numpy as np
import pytest

from packages.analytics.divknn_select import knn_unique_select
from packages.analytics.embedding_distribution import (
    compute_distance_stats,
    project_pca,
    summarize_distance_stats,
    text_similarity_to_gallery,
)
from packages.analytics.embeddings.vectors import l2_normalize, load_embeddings_parquet


def _unit(v):
    a = np.asarray(v, dtype=np.float64)
    return a / np.linalg.norm(a)


class TestVectors:
    def test_l2_normalize(self):
        x = np.array([[3.0, 4.0], [0.0, 0.0]])
        y = l2_normalize(x)
        assert y[0] == pytest.approx(np.array([0.6, 0.8]))
        assert np.linalg.norm(y[1]) < 1e-6

    def test_load_parquet(self, tmp_path):
        import pandas as pd

        df = pd.DataFrame(
            {
                "file_name": ["a.mp4", "b.mp4"],
                "embedding": [[1.0, 0.0], [0.0, 1.0]],
            }
        )
        path = tmp_path / "embeddings.parquet"
        df.to_parquet(path)
        names, mat, loaded = load_embeddings_parquet(path)
        assert names == ["a.mp4", "b.mp4"]
        assert mat.shape == (2, 2)
        assert pytest.approx(np.linalg.norm(mat[0]), abs=1e-6) == 1.0


class TestDistanceStats:
    def test_s_closer_to_matching_b(self):
        xs = np.stack([_unit([1, 0]), _unit([0, 1])])
        xb = np.stack([_unit([1, 0.01]), _unit([0.01, 1]), _unit([-1, 0])])
        stats = compute_distance_stats(xs, xb)
        assert stats.s_to_b_top1.shape == (2,)
        assert stats.s_to_b_top1.min() > 0.9
        summary = summarize_distance_stats(stats)
        assert "s_to_b_top1_mean" in summary

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            compute_distance_stats(np.zeros((0, 2)), np.eye(2))


class TestPCA:
    def test_project_pca_shapes(self):
        xs = np.random.default_rng(0).normal(size=(5, 8))
        xb = np.random.default_rng(1).normal(size=(20, 8))
        proj = project_pca(xs, xb, n_components=3)
        assert proj.coords.shape == (25, 3)
        assert proj.n_s == 5
        assert proj.explained_variance_pct is not None
        assert len(proj.explained_variance_pct) == 3


class TestTextSim:
    def test_text_similarity(self):
        gallery = np.stack([_unit([1, 0]), _unit([0, 1])])
        text = _unit([1, 0])
        sims = text_similarity_to_gallery(gallery, text)
        assert sims[0] == pytest.approx(1.0, abs=1e-6)
        assert sims[1] == pytest.approx(0.0, abs=1e-6)


class TestDivknnSelect:
    def test_selects_unique_neighbors(self):
        t_names = ["t0", "t1"]
        t_emb = np.stack([_unit([1, 0, 0]), _unit([0, 1, 0])])
        s_names = ["b0", "b1", "b2", "b3"]
        s_emb = np.stack(
            [
                _unit([0.99, 0.01, 0]),
                _unit([0.98, 0.02, 0]),
                _unit([0.01, 0.99, 0]),
                _unit([0, 0, 1]),
            ]
        )
        selected = knn_unique_select(t_names, t_emb, s_names, s_emb, top_n=1, backup=2)
        assert len(selected) == len(set(selected))
        assert "b0" in selected or "b1" in selected
        assert "b2" in selected
