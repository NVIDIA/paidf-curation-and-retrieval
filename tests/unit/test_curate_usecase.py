# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for curation use-case composition."""

from __future__ import annotations

import pytest

from apps.cli.curate import curate_domain_subset
from apps.composition import build_services
from packages.domain.types import EmbeddingRecord


class TestCurateDomainSubset:
    def test_happy_path_unique_ids(self):
        services = build_services("http://localhost:8000", use_gpu_knn=False)
        targets = [EmbeddingRecord("t0", [1.0, 0.0, 0.0])]
        sources = [
            EmbeddingRecord("b0", [1.0, 0.0, 0.0]),
            EmbeddingRecord("b1", [0.0, 1.0, 0.0]),
            EmbeddingRecord("b2", [0.0, 0.0, 1.0]),
        ]
        result = curate_domain_subset(services, targets, sources, top_n=1, backup_candidates=2)
        assert "b0" in result["unique_source_ids"]
        assert len(result["dataset_search_rows"]) >= 1
        assert result["dataset_search_rows"][0]["id"] in result["unique_source_ids"]

    def test_with_diversify(self):
        services = build_services("http://localhost:8000", use_gpu_knn=False)
        targets = [
            EmbeddingRecord("t0", [1.0, 0.0]),
            EmbeddingRecord("t1", [0.0, 1.0]),
        ]
        sources = [
            EmbeddingRecord("b0", [1.0, 0.0]),
            EmbeddingRecord("b1", [0.9, 0.1]),
            EmbeddingRecord("b2", [0.0, 1.0]),
            EmbeddingRecord("b3", [0.1, 0.9]),
        ]
        result = curate_domain_subset(
            services,
            targets,
            sources,
            top_n=1,
            backup_candidates=2,
            diversify=True,
            n_diverse=2,
            n_clusters=2,
        )
        assert len(result["unique_source_ids"]) <= 2
        assert result["subset"] is not None

    def test_cluster_empty_and_cpu_kmeans(self):
        from adapters.data_mining.cluster_adapter import DataMiningClusterAdapter

        adapter = DataMiningClusterAdapter()
        empty = adapter.fit_predict([])
        assert empty.assignments == []
        records = [
            EmbeddingRecord("a", [0.0, 0.0]),
            EmbeddingRecord("b", [1.0, 0.0]),
            EmbeddingRecord("c", [0.0, 1.0]),
        ]
        result = adapter.fit_predict(records, method="kmeans", n_clusters=2)
        assert len(result.assignments) == 3
        assert result.method == "kmeans"

    def test_non_kmeans_without_cuml_raises(self):
        from adapters.data_mining.cluster_adapter import DataMiningClusterAdapter

        adapter = DataMiningClusterAdapter()
        with pytest.raises(ValueError, match="requires cuML"):
            adapter.fit_predict(
                [EmbeddingRecord("a", [1.0, 0.0]), EmbeddingRecord("b", [0.0, 1.0])],
                method="hdbscan",
            )
