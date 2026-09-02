# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""External caption application workflows."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


class CaptionCommandClient(Protocol):
    """Port for Dataset Search caption command operations."""

    def search(
        self,
        query: str,
        *,
        limit: int = 5000,
        data_sources: Sequence[str] | None = None,
    ) -> Mapping[str, Any]:
        """Search external captions."""

    def upload_parquet(
        self,
        parquet_path: str | Path,
        *,
        model_name: str = "default",
        data_source: str = "",
    ) -> Mapping[str, Any]:
        """Upload one local caption parquet."""

    def bulk_insert(
        self,
        parquet_paths: Sequence[str],
        *,
        access_key: str | None = None,
        secret_key: str | None = None,
        endpoint_url: str | None = None,
        allow_insecure_endpoint: bool = False,
        model_name_override: str | None = None,
        data_source_override: str | None = None,
    ) -> Mapping[str, Any]:
        """Submit caption parquet object-store references."""


class CaptionParquetBuilder(Protocol):
    """Port for building a caption parquet artifact."""

    def __call__(
        self,
        rows: Sequence[dict[str, Any]],
        output_path: str,
        *,
        indexed_clip_ids: set[str] | None = None,
    ) -> Path:
        """Build one caption parquet file."""


class CaptionParquetValidator(Protocol):
    """Port for validating a caption parquet artifact."""

    def __call__(
        self,
        parquet_path: str | Path,
        *,
        indexed_clip_ids: set[str] | None = None,
    ) -> dict[str, Any]:
        """Validate one caption parquet file."""


@dataclass(frozen=True)
class CaptionReadinessRequest:
    """Inputs for external-caption readiness checks."""

    cds_profile: str
    required_pipeline: str


@dataclass(frozen=True)
class CaptionSearchRequest:
    """Inputs for caption keyword search."""

    query: str
    limit: int
    data_sources: Sequence[str] | None = None


@dataclass(frozen=True)
class BuildCaptionParquetRequest:
    """Inputs for building the caption parquet contract."""

    input_json: str
    output_parquet: str
    indexed_clip_ids: set[str] | None = None


@dataclass(frozen=True)
class ValidateCaptionParquetRequest:
    """Inputs for validating the caption parquet contract."""

    parquet_path: str
    indexed_clip_ids: set[str] | None = None


@dataclass(frozen=True)
class UploadCaptionParquetRequest:
    """Inputs for validating and uploading one caption parquet."""

    parquet_path: str
    model_name: str
    data_source: str
    indexed_clip_ids: set[str] | None = None


@dataclass(frozen=True)
class BulkInsertCaptionParquetsRequest:
    """Inputs for submitting caption parquet object-store references."""

    parquet_paths: Sequence[str]
    access_key: str | None = None
    secret_key: str | None = None
    endpoint_url: str | None = None
    allow_lab_http_endpoint: bool = False
    model_name_override: str | None = None
    data_source_override: str | None = None


def run_caption_readiness(
    request: CaptionReadinessRequest,
    *,
    cds_client: object,
    caption_client: object,
    readiness_check: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    """Run the external-caption readiness check."""
    return readiness_check(
        cds_client,
        caption_client,
        cds_profile=request.cds_profile.lower(),
        required_pipeline=request.required_pipeline,
    )


def search_captions(
    request: CaptionSearchRequest,
    *,
    client_factory: Callable[[], CaptionCommandClient],
) -> dict[str, Any]:
    """Run finite caption keyword search."""
    result = client_factory().search(
        request.query,
        limit=request.limit,
        data_sources=request.data_sources,
    )
    return {"status": "ok", **result}


def build_caption_parquet_handoff(
    request: BuildCaptionParquetRequest,
    *,
    build_caption: CaptionParquetBuilder,
    validate_caption: CaptionParquetValidator,
) -> dict[str, Any]:
    """Build and validate a caption parquet artifact."""
    try:
        payload = json.loads(Path(request.input_json).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid caption JSON: {exc.msg}") from exc
    if not isinstance(payload, list) or any(not isinstance(row, dict) for row in payload):
        raise ValueError("--input-json must contain a JSON array of row objects")
    output = build_caption(
        payload,
        request.output_parquet,
        indexed_clip_ids=request.indexed_clip_ids,
    )
    summary = validate_caption(output, indexed_clip_ids=request.indexed_clip_ids)
    return {"status": "built", **summary}


def validate_caption_parquet_handoff(
    request: ValidateCaptionParquetRequest,
    *,
    validate_caption: CaptionParquetValidator,
) -> dict[str, Any]:
    """Validate a caption parquet artifact."""
    summary = validate_caption(
        request.parquet_path,
        indexed_clip_ids=request.indexed_clip_ids,
    )
    return {"status": "valid", **summary}


def upload_caption_parquet_handoff(
    request: UploadCaptionParquetRequest,
    *,
    validate_caption: CaptionParquetValidator,
    client_factory: Callable[[], CaptionCommandClient],
) -> dict[str, Any]:
    """Validate and upload one caption parquet artifact."""
    validate_caption(
        request.parquet_path,
        indexed_clip_ids=request.indexed_clip_ids,
    )
    result = client_factory().upload_parquet(
        request.parquet_path,
        model_name=request.model_name,
        data_source=request.data_source,
    )
    return {"status": "submitted", "result": result}


def bulk_insert_caption_parquets(
    request: BulkInsertCaptionParquetsRequest,
    *,
    client_factory: Callable[[], CaptionCommandClient],
) -> dict[str, Any]:
    """Submit caption parquet object-store references."""
    result = client_factory().bulk_insert(
        request.parquet_paths,
        access_key=request.access_key,
        secret_key=request.secret_key,
        endpoint_url=request.endpoint_url,
        allow_insecure_endpoint=request.allow_lab_http_endpoint,
        model_name_override=request.model_name_override,
        data_source_override=request.data_source_override,
    )
    return {"status": "submitted", "result": result}
