# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ports: interfaces between product use-cases and infrastructure.

Implementations live under ``adapters/``. Domain code depends only on these
protocols — never on Dataset Search FastAPI modules or Data Mining click CLIs directly.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from packages.domain.types import (
    BatchSimilarityResult,
    BulkInsertRequest,
    BulkJobStatus,
    ClusterResult,
    CollectionCreateSpec,
    CollectionInfo,
    CollectionPatchSpec,
    CollectionRef,
    DocumentInfo,
    DocumentSpec,
    EmbeddingRecord,
    MultiCollectionQuery,
    PipelineInfo,
    SearchHit,
    SearchQuery,
    SearchRefinementResult,
    SearchRefinementSpec,
    SubsetSelection,
)


class RetrievalPort(Protocol):
    """Online search (owned by Dataset Search / CVDS)."""

    def search(self, collection: CollectionRef, query: SearchQuery) -> Sequence[SearchHit]:
        """Run multimodal search; maps to POST /collections/{id}/search."""

    def multi_collection_search(self, request: MultiCollectionQuery) -> Sequence[SearchHit]:
        """Search across collections; maps to POST /retrieval."""

    def export_embeddings(self, collection: CollectionRef) -> Sequence[EmbeddingRecord]:
        """Export vectors + ids when the public API exposes them."""


class CollectionAdminPort(Protocol):
    """Collection CRUD and admin flush (CVDS collections + milvus_admin)."""

    def list_collections(self) -> Sequence[CollectionInfo]:
        """GET /collections."""

    def get_collection(self, collection_id: str) -> CollectionInfo:
        """GET /collections/{id}."""

    def create_collection(self, spec: CollectionCreateSpec) -> CollectionInfo:
        """POST /collections."""

    def update_collection(self, collection_id: str, patch: CollectionPatchSpec) -> CollectionInfo:
        """PATCH /collections/{id}."""

    def delete_collection(self, collection_id: str) -> None:
        """DELETE /collections/{id}."""

    def flush_collection(self, collection_id: str) -> Mapping[str, Any]:
        """POST /admin/collections/{id}/flush."""


class PipelinePort(Protocol):
    """Pipeline discovery (CVDS pipelines API)."""

    def list_pipelines(self) -> Sequence[PipelineInfo]:
        """GET /pipelines."""

    def draw_pipeline(self, name: str, mode: str = "index") -> bytes:
        """GET /pipelines/draw/{name} — returns pipeline diagram image bytes."""


class IngestPort(Protocol):
    """Document + bulk parquet ingest into Dataset Search."""

    def ingest_documents(
        self, collection: CollectionRef, documents: Sequence[DocumentSpec]
    ) -> Sequence[DocumentInfo]:
        """POST /collections/{id}/documents."""

    def delete_document(self, collection: CollectionRef, document_id: str) -> None:
        """DELETE /collections/{id}/documents/{document_id}."""

    def delete_documents_by_filter(
        self, collection: CollectionRef, filters: Mapping[str, Any]
    ) -> None:
        """DELETE /collections/{id}/documents with filter body."""

    def ingest_parquet(
        self,
        collection: CollectionRef,
        parquet_paths: Sequence[str],
        *,
        embedding_family: str,
        allow_document_fallback: bool = False,
    ) -> str:
        """POST /insert-data with an explicit vector family; returns job_id."""

    def bulk_insert(self, request: BulkInsertRequest) -> BulkJobStatus:
        """POST /insert-data with optional storage credentials."""

    def get_job_status(self, job_id: str) -> BulkJobStatus:
        """GET /job-status/{job_id}."""

    def list_jobs(self) -> Sequence[BulkJobStatus]:
        """GET /jobs."""


class SearchRefinementPort(Protocol):
    """Search refinement training (CVDS /search_refinement/train)."""

    def train_search_refinement(self, spec: SearchRefinementSpec) -> SearchRefinementResult:
        """POST /search_refinement/train."""


class HealthPort(Protocol):
    """Service liveness."""

    def health(self) -> str:
        """GET /health."""


class EmbeddingStorePort(Protocol):
    """Read/write embedding parquet (shared substrate)."""

    def load(self, path: str) -> Sequence[EmbeddingRecord]:
        """Load embedding records from a directory or file."""

    def save(self, path: str, records: Sequence[EmbeddingRecord]) -> None:
        """Persist embedding records for ingest or analytics."""


class BatchSimilarityPort(Protocol):
    """Target-guided batch kNN with uniqueness (Data Mining DivKNN behavior)."""

    def select_similar(
        self,
        targets: Sequence[EmbeddingRecord],
        sources: Sequence[EmbeddingRecord],
        top_n: int = 5,
        backup_candidates: int = 15,
        metric: str = "cosine",
        apply_uniqueness: bool = True,
    ) -> BatchSimilarityResult:
        """Return diverse/de-duplicated top-K neighbors from sources for each target."""


class ClusterAnalyticsPort(Protocol):
    """Offline clustering over exported embeddings (TAO clustering)."""

    def fit_predict(
        self,
        records: Sequence[EmbeddingRecord],
        method: str = "kmeans",
        n_clusters: int = 8,
    ) -> ClusterResult:
        """Assign cluster labels; same algorithms as tao_data_curation.clustering."""


class SubsetSelectionPort(Protocol):
    """Farthest-point / diversity sampling (TAO diversity_sampling)."""

    def select_diverse(
        self,
        records: Sequence[EmbeddingRecord],
        n_samples: int,
        cluster_labels: Sequence[int] | None = None,
    ) -> SubsetSelection:
        """Select a diverse subset; optional cluster seeds."""
