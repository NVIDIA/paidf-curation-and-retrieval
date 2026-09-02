# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cross-product artifact handoff workflows."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from packages.domain.types import CollectionRef


class CuratorExportConverter(Protocol):
    """Port for converting Curator exports to CDS-shaped parquet."""

    def to_cds_parquet(self, base_dir: str, output_path: str, backend: str = "auto") -> str:
        """Convert one Curator output directory to a parquet file."""


class DatasetSearchIngestClient(Protocol):
    """Port for ingesting parquet artifacts into Dataset Search."""

    def ingest_parquet(
        self,
        collection: CollectionRef,
        parquet_paths: Sequence[str],
        *,
        embedding_family: str,
        allow_document_fallback: bool = False,
    ) -> str:
        """Ingest parquet artifacts into one collection."""


class ObjectArtifactStager(Protocol):
    """Port for staging one local artifact to object storage."""

    def stage(self, source: str, destination: str) -> str:
        """Stage a local source path to an object-store destination."""


class TmmParquetPairPreparer(Protocol):
    """Port for preparing target/source parquet pairs for TMM."""

    def __call__(
        self,
        data_dir: str,
        *,
        target_subdir: str,
        source_subdir: str,
        prep_subdir: str,
    ) -> tuple[Path, Path]:
        """Prepare TMM target and source parquet files."""


class TmmParquetPairValidator(Protocol):
    """Port for validating a TMM parquet pair."""

    def __call__(self, target_path: Path, source_path: Path) -> int:
        """Return the validated embedding dimension."""


@dataclass(frozen=True)
class IngestCuratorExportRequest:
    """Inputs for converting and optionally ingesting Curator exports."""

    curator_dir: str
    output_parquet: str
    collection: str | None = None
    embedding_backend: str = "auto"
    convert_only: bool = True
    allow_lab_document_fallback: bool = False


@dataclass(frozen=True)
class IngestCuratorExportResult:
    """Stable result returned to CLI/API presenters."""

    cds_parquet: str
    embedding_backend: str
    ingested: bool = False
    collection: str | None = None
    job_id: str | None = None
    embedding_family: str | None = None

    @property
    def conversion_payload(self) -> dict[str, str]:
        """JSON-ready payload for the conversion step."""
        return {
            "cds_parquet": self.cds_parquet,
            "embedding_backend": self.embedding_backend,
        }

    @property
    def ingest_payload(self) -> dict[str, object] | None:
        """JSON-ready payload for the optional ingest step."""
        if not self.ingested:
            return None
        return {
            "ingested": True,
            "collection": self.collection,
            "job_id": self.job_id,
            "embedding_family": self.embedding_family,
        }


@dataclass(frozen=True)
class StageArtifactRequest:
    """Inputs for staging one artifact to object storage."""

    source: str
    destination: str


@dataclass(frozen=True)
class PrepareCdsCe1ForTdmRequest:
    """Inputs for preparing CDS-compatible CE1 parquet as TMM S/B inputs."""

    data_dir: str
    target_selection: str
    source_selection: str
    output_subdir: str
    embedding_family: str


def ingest_curator_export(
    request: IngestCuratorExportRequest,
    *,
    converter_factory: Callable[[], CuratorExportConverter],
    client_factory: Callable[[], DatasetSearchIngestClient] | None = None,
) -> IngestCuratorExportResult:
    """Convert a Curator export and optionally ingest it into Dataset Search."""
    selected_backend = request.embedding_backend.lower()
    if not request.convert_only:
        if not request.collection:
            raise ValueError("--collection is required with --ingest")
        if selected_backend == "iv2":
            raise ValueError("CDS-bound ingest requires the CE1 embedding backend")
        if selected_backend == "auto":
            selected_backend = "ce1"

    out = converter_factory().to_cds_parquet(
        request.curator_dir,
        request.output_parquet,
        backend=selected_backend,
    )

    if request.convert_only:
        return IngestCuratorExportResult(
            cds_parquet=out,
            embedding_backend=selected_backend,
        )

    if client_factory is None:
        raise ValueError("client_factory is required with --ingest")
    collection = str(request.collection)
    job_id = client_factory().ingest_parquet(
        CollectionRef(collection_id=collection, name=collection),
        [str(Path(out).resolve())],
        embedding_family="ce1",
        allow_document_fallback=request.allow_lab_document_fallback,
    )
    return IngestCuratorExportResult(
        cds_parquet=out,
        embedding_backend=selected_backend,
        ingested=True,
        collection=collection,
        job_id=job_id,
        embedding_family="ce1",
    )


def stage_artifact_handoff(
    request: StageArtifactRequest,
    *,
    stager_factory: Callable[[], ObjectArtifactStager],
) -> dict[str, str]:
    """Stage one artifact and return a stable handoff manifest."""
    uri = stager_factory().stage(request.source, request.destination)
    return {
        "status": "staged",
        "input": str(Path(request.source)),
        "output": uri,
    }


def prepare_cds_ce1_for_tdm_handoff(
    request: PrepareCdsCe1ForTdmRequest,
    *,
    validate_embedding_family: Callable[[str], str],
    prepare_pair: TmmParquetPairPreparer,
    validate_pair: TmmParquetPairValidator,
    container_path: Callable[[Path, str], str],
) -> dict[str, Any]:
    """Prepare selected CDS-compatible CE1 parquet for TMM."""
    family = validate_embedding_family(request.embedding_family)
    target_path, source_path = prepare_pair(
        request.data_dir,
        target_subdir=request.target_selection,
        source_subdir=request.source_selection,
        prep_subdir=request.output_subdir,
    )
    dimension = validate_pair(target_path, source_path)
    return {
        "status": "prepared",
        "embedding_family": family,
        "dimension": dimension,
        "target_selection": request.target_selection,
        "source_selection": request.source_selection,
        "target_parquet": str(target_path),
        "source_parquet": str(source_path),
        "target_container_path": container_path(target_path, request.data_dir),
        "source_container_path": container_path(source_path, request.data_dir),
    }
