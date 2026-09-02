# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for DivKNN adapter uniqueness behavior (CPU path)."""

from __future__ import annotations

import pytest

from adapters.data_mining.divknn_adapter import DataMiningDivKnnAdapter
from adapters.data_mining.tmm_parquet import TMM_METRICS
from packages.domain.types import EmbeddingRecord


class TestDataMiningDivKnnAdapter:
    def test_select_similar_returns_unique_sources(self):
        adapter = DataMiningDivKnnAdapter(use_gpu_knn=False)
        targets = [
            EmbeddingRecord("t0", [1.0, 0.0]),
            EmbeddingRecord("t1", [0.9, 0.1]),
        ]
        sources = [
            EmbeddingRecord("b0", [1.0, 0.0]),
            EmbeddingRecord("b1", [0.95, 0.05]),
            EmbeddingRecord("b2", [0.0, 1.0]),
        ]
        result = adapter.select_similar(
            targets, sources, top_n=1, backup_candidates=2, metric="cosine"
        )
        assert len(result.unique_source_ids) == len(set(result.unique_source_ids))
        assert all(m.rank == 1 for m in result.matches)

    def test_invalid_top_n_with_uniqueness(self):
        adapter = DataMiningDivKnnAdapter(use_gpu_knn=False)
        targets = [EmbeddingRecord("t0", [1.0, 0.0])]
        sources = [EmbeddingRecord("b0", [1.0, 0.0])]
        with pytest.raises(ValueError, match="top_n must be less than total_n"):
            adapter.select_similar(
                targets, sources, top_n=1, backup_candidates=0, apply_uniqueness=True
            )

    def test_empty_sources(self):
        adapter = DataMiningDivKnnAdapter(use_gpu_knn=False)
        targets = [EmbeddingRecord("t0", [1.0, 0.0])]
        result = adapter.select_similar(
            targets, [], top_n=1, backup_candidates=1, apply_uniqueness=False
        )
        assert result.unique_source_ids == []

    def test_euclidean_metric(self):
        adapter = DataMiningDivKnnAdapter(use_gpu_knn=False)
        targets = [EmbeddingRecord("t0", [1.0, 0.0])]
        sources = [
            EmbeddingRecord("b0", [1.0, 0.0]),
            EmbeddingRecord("b1", [0.0, 1.0]),
        ]
        result = adapter.select_similar(
            targets, sources, top_n=1, backup_candidates=1, metric="euclidean"
        )
        assert result.matches[0].source_id == "b0"

    def test_tao_tmm_metrics_do_not_change_in_process_divknn_contract(self):
        assert TMM_METRICS == {"cosine", "euclidean", "manhattan"}
        assert "l2" not in TMM_METRICS
        adapter = DataMiningDivKnnAdapter(use_gpu_knn=False)
        with pytest.raises(ValueError, match="Unsupported metric"):
            adapter.select_similar(
                [EmbeddingRecord("t0", [1.0])],
                [EmbeddingRecord("b0", [1.0])],
                top_n=1,
                backup_candidates=1,
                metric="manhattan",
                apply_uniqueness=False,
            )

    def test_total_candidates_must_be_positive(self):
        adapter = DataMiningDivKnnAdapter(use_gpu_knn=False)
        with pytest.raises(ValueError, match="must be >= 1"):
            adapter.select_similar([], [], top_n=0, backup_candidates=0)
