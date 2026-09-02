# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Composition root — wires ports to adapters for the integration layer.

Use-cases depend only on ports; this module is the single place that knows
about Dataset Search HTTP and Data Mining libraries.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from adapters.data_mining.cluster_adapter import DataMiningClusterAdapter
from adapters.data_mining.diversity_adapter import DataMiningDiversityAdapter
from adapters.data_mining.divknn_adapter import DataMiningDivKnnAdapter
from adapters.dataset_search.caption_adapter import CaptionAdapter
from adapters.dataset_search.retrieval_adapter import DatasetSearchAdapter, normalize_cds_base_url
from adapters.docker_jobs import CuratorDockerRunner, DataMiningDockerRunner
from packages.ports import (
    BatchSimilarityPort,
    ClusterAnalyticsPort,
    IngestPort,
    RetrievalPort,
    SubsetSelectionPort,
)

DEFAULT_CDS_URL = "http://localhost:8888/v1"


@dataclass
class ProductServices:
    """Injectable service bundle for CLI / jobs / API workers."""

    retrieval: RetrievalPort
    ingest: IngestPort
    batch_similarity: BatchSimilarityPort
    clustering: ClusterAnalyticsPort
    diversity: SubsetSelectionPort


def resolve_cds_base_url(cds_url: str | None = None) -> str:
    """Resolve the Dataset Search base URL from CLI input or environment."""
    return normalize_cds_base_url(cds_url or os.environ.get("CDS_URL") or DEFAULT_CDS_URL)


def build_dataset_search_adapter(
    cds_url: str | None = None,
    *,
    session=None,
) -> DatasetSearchAdapter:
    """Build the Dataset Search adapter used by CLI delivery code."""
    return DatasetSearchAdapter(base_url=resolve_cds_base_url(cds_url), session=session)


def build_caption_adapter(
    cds_url: str | None = None,
    *,
    session=None,
) -> CaptionAdapter:
    """Build the Dataset Search caption adapter used by CLI delivery code."""
    return CaptionAdapter(resolve_cds_base_url(cds_url), session=session)


def build_curator_runner(*, image: str) -> CuratorDockerRunner:
    """Build the Cosmos Curator Docker runner."""
    return CuratorDockerRunner(image=image)


def build_data_mining_runner(
    *,
    image: str,
    gpus: str,
    shm_size: str,
) -> DataMiningDockerRunner:
    """Build the TAO Data Mining Docker runner."""
    return DataMiningDockerRunner(image=image, gpus=gpus, shm_size=shm_size)


def build_services(
    cds_base_url: str,
    *,
    use_gpu_knn: bool = False,
    session=None,
) -> ProductServices:
    """Factory used by apps. Behavior of each adapter is unchanged from source tools."""
    dataset_search = build_dataset_search_adapter(cds_base_url, session=session)
    return ProductServices(
        retrieval=dataset_search,
        ingest=dataset_search,
        batch_similarity=DataMiningDivKnnAdapter(use_gpu_knn=use_gpu_knn),
        clustering=DataMiningClusterAdapter(),
        diversity=DataMiningDiversityAdapter(),
    )
