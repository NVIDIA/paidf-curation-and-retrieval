# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Domain package: shared types only. No I/O, no framework deps."""

from packages.domain.types import (
    BatchSimilarityResult,
    BulkInsertRequest,
    BulkJobStatus,
    ClusterAssignment,
    ClusterResult,
    CollectionCreateSpec,
    CollectionInfo,
    CollectionPatchSpec,
    CollectionRef,
    DocumentInfo,
    DocumentSpec,
    EmbeddingRecord,
    LabelledDocuments,
    MultiCollectionQuery,
    NeighborMatch,
    PipelineInfo,
    SearchHit,
    SearchQuery,
    SearchRefinementResult,
    SearchRefinementSpec,
    SubsetSelection,
)

__all__ = [
    "BatchSimilarityResult",
    "BulkInsertRequest",
    "BulkJobStatus",
    "ClusterAssignment",
    "ClusterResult",
    "CollectionCreateSpec",
    "CollectionInfo",
    "CollectionPatchSpec",
    "CollectionRef",
    "DocumentInfo",
    "DocumentSpec",
    "EmbeddingRecord",
    "LabelledDocuments",
    "MultiCollectionQuery",
    "NeighborMatch",
    "PipelineInfo",
    "SearchHit",
    "SearchQuery",
    "SearchRefinementResult",
    "SearchRefinementSpec",
    "SubsetSelection",
]
