# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared domain types for the PAIDF integration layer.

These types are the contract between Dataset Search (online retrieval) and
Data Mining (offline analytics). Adapters map to/from each engine's native schemas.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class CollectionRef:
    """Reference to a searchable collection in the retrieval service."""

    collection_id: str
    name: str | None = None


@dataclass(frozen=True)
class CollectionInfo:
    """Collection metadata returned by Dataset Search."""

    collection_id: str
    name: str
    pipeline: str
    tags: Mapping[str, Any] = field(default_factory=dict)
    created_at: str | None = None
    total_documents_count: int | None = None


@dataclass(frozen=True)
class CollectionCreateSpec:
    """Payload to create a Dataset Search collection (CVDS CollectionCreate)."""

    name: str
    pipeline: str
    tags: Mapping[str, Any] = field(default_factory=dict)
    collection_config: Mapping[str, Any] = field(default_factory=dict)
    index_config: Mapping[str, Any] = field(default_factory=dict)
    metadata_config: Mapping[str, Any] = field(default_factory=dict)
    collection_id: str | None = None


@dataclass(frozen=True)
class CollectionPatchSpec:
    """Partial update for a collection (name and/or tags)."""

    name: str | None = None
    tags: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class PipelineInfo:
    """Pipeline descriptor from GET /pipelines."""

    pipeline_id: str
    name: str | None = None
    description: str | None = None
    raw: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EmbeddingRecord:
    """Canonical embedding row used across analytics and ingest.

    Dataset Search uses ``id`` + ``embedding`` + optional ``$meta``.
    Data Mining uses ``file_name`` + ``embedding`` (list). Adapters map both ways.
    """

    record_id: str
    embedding: Sequence[float]
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SearchHit:
    """One retrieval result from the online search path."""

    record_id: str
    score: float
    metadata: Mapping[str, Any] = field(default_factory=dict)
    collection_id: str | None = None
    asset_url: str | None = None
    embedding: Sequence[float] | None = None


@dataclass(frozen=True)
class SearchQuery:
    """Multimodal query against Dataset Search (CVDS SearchRequest).

    Provide exactly one modality: text, embedding, image/image_path, video,
    video_frames, or session_segment. ``image_path`` is kept for backward
    compatibility and maps to CVDS ``ImageQuery.image``.
    """

    text: str | None = None
    image_path: str | None = None
    image: str | None = None
    video: str | None = None
    video_frames: str | None = None
    embedding: Sequence[float] | None = None
    session_segment: Mapping[str, str] | None = None
    top_k: int = 10
    filters: Mapping[str, Any] | str = field(default_factory=dict)
    reconstruct: bool = False
    search_params: Mapping[str, Any] = field(default_factory=dict)
    # CVDS defaults True, but that requires collection storage-secrets tags.
    # Integration default False avoids local failures without S3 wiring.
    generate_asset_url: bool = False
    clf: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MultiCollectionQuery:
    """Multi-collection retrieval (CVDS POST /retrieval)."""

    collection_ids: Sequence[str]
    query: SearchQuery
    rerank: bool = True
    payload_keys: Sequence[str] | None = None
    generate_asset_url: bool = False


@dataclass(frozen=True)
class DocumentSpec:
    """Document for online indexing (JSON content, URL, or precomputed embedding)."""

    mime_type: str
    content: str | None = None
    url: str | None = None
    embedding: Sequence[float] | None = None
    document_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DocumentInfo:
    """Indexed document acknowledgment from Dataset Search."""

    document_id: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
    indexed_at: str | None = None


@dataclass(frozen=True)
class BulkInsertRequest:
    """Bulk parquet ingest (CVDS POST /insert-data)."""

    collection_name: str
    parquet_paths: Sequence[str]
    embedding_family: str
    access_key: str | None = None
    secret_key: str | None = None
    endpoint_url: str | None = None
    allow_insecure_endpoint: bool = False


@dataclass(frozen=True)
class BulkJobStatus:
    """Status of a Milvus bulk-insert job."""

    job_id: str
    status: str
    details: str = ""
    progress: int | None = None
    collection_name: str | None = None
    raw: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LabelledDocuments:
    """Binary labels for documents in one collection (search refinement)."""

    collection_name: str
    labelled_documents: Mapping[str, bool]


@dataclass(frozen=True)
class SearchRefinementSpec:
    """Train a search-refinement model (CVDS POST /search_refinement/train)."""

    grounding_queries: Sequence[SearchQuery]
    labels: Sequence[LabelledDocuments]
    model_type: str = "linear_probe"
    regularization_strength: float = 0.05


@dataclass(frozen=True)
class SearchRefinementResult:
    """Result of search-refinement training."""

    model_type: str
    queries: Sequence[Sequence[float]] = field(default_factory=tuple)
    coef: Sequence[Sequence[float]] | None = None
    intercept: Sequence[float] | None = None
    raw: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class NeighborMatch:
    """One target→source neighbor from batch similarity (DivKNN-style)."""

    target_id: str
    source_id: str
    distance: float
    rank: int


@dataclass(frozen=True)
class BatchSimilarityResult:
    """Result of target-guided batch similarity with uniqueness applied."""

    matches: Sequence[NeighborMatch]
    unique_source_ids: Sequence[str]


@dataclass(frozen=True)
class ClusterAssignment:
    """Cluster label for one embedding record."""

    record_id: str
    label: int


@dataclass(frozen=True)
class ClusterResult:
    """Offline clustering output."""

    assignments: Sequence[ClusterAssignment]
    method: str
    description: str = ""


@dataclass(frozen=True)
class SubsetSelection:
    """Diverse / curated subset of record ids."""

    record_ids: Sequence[str]
    strategy: str
    params: Mapping[str, Any] = field(default_factory=dict)
