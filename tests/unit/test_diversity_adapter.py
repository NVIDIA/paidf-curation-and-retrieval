# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for farthest-point diversity adapter."""

from __future__ import annotations

import pytest

from adapters.data_mining.diversity_adapter import DataMiningDiversityAdapter
from packages.domain.types import EmbeddingRecord


class TestDataMiningDiversityAdapter:
    def test_selects_requested_count(self):
        adapter = DataMiningDiversityAdapter()
        records = [
            EmbeddingRecord("a", [0.0, 0.0]),
            EmbeddingRecord("b", [1.0, 0.0]),
            EmbeddingRecord("c", [0.0, 1.0]),
            EmbeddingRecord("d", [1.0, 1.0]),
        ]
        subset = adapter.select_diverse(records, n_samples=3)
        assert len(subset.record_ids) == 3
        assert subset.strategy == "farthest_point"

    def test_cluster_seed_length_mismatch(self):
        adapter = DataMiningDiversityAdapter()
        records = [EmbeddingRecord("a", [0.0, 0.0]), EmbeddingRecord("b", [1.0, 0.0])]
        with pytest.raises(ValueError, match="cluster_labels length"):
            adapter.select_diverse(records, n_samples=1, cluster_labels=[0])

    def test_n_samples_must_be_positive(self):
        adapter = DataMiningDiversityAdapter()
        with pytest.raises(ValueError, match="n_samples must be positive"):
            adapter.select_diverse([EmbeddingRecord("a", [1.0])], n_samples=0)

    def test_empty_records(self):
        adapter = DataMiningDiversityAdapter()
        subset = adapter.select_diverse([], n_samples=5)
        assert subset.record_ids == []
