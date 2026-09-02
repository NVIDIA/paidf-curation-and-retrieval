# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Application workflows for TAO image and text embedding handoffs.

Request/result DTOs and orchestration functions used by the Make-backed CLI.
These workflows call adapter helpers to build or check parquet contracts, then
delegate live Docker execution to a
:class:`~adapters.docker_jobs.DataMiningDockerRunner` (or test double).
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


class ImageEmbeddingRunResult(Protocol):
    """Minimum result contract needed by the image-embedding run workflow."""

    @property
    def command(self) -> Sequence[str]:
        """Command used by the underlying runner."""
        ...

    @property
    def evidence(self) -> dict[str, Any]:
        """Validation evidence returned by the underlying runner."""
        ...


class ImageEmbeddingRunner(Protocol):
    """Port for running TAO Data Services image embeddings."""

    def image_embeddings_config(
        self,
        *,
        config_file: str,
        data_dir: str,
        dry_run: bool = False,
    ) -> ImageEmbeddingRunResult:
        """Run image embeddings from an engine-native YAML."""

    def image_embeddings(
        self,
        *,
        data_dir: str,
        input_parquet: str,
        output_parquet: str,
        model_type: str,
        model_name_or_path: str,
        model_config_path: str | None = None,
        batch_size: int = 64,
        dry_run: bool = False,
    ) -> ImageEmbeddingRunResult:
        """Run image embeddings from direct options."""


class ImageEmbeddingInputBuilder(Protocol):
    """Port for building a TAO DS image-embedding input parquet."""

    def __call__(
        self,
        rows: Sequence[dict[str, Any]],
        output_path: str,
        *,
        data_dir: str,
    ) -> Path:
        """Build one image-embedding input parquet."""


class ImageEmbeddingInputValidator(Protocol):
    """Port for validating a TAO DS image-embedding input parquet."""

    def __call__(
        self,
        parquet_path: str | Path,
        *,
        data_dir: str,
    ) -> dict[str, Any]:
        """Validate one image-embedding input parquet."""


class ImageEmbeddingOutputValidator(Protocol):
    """Port for validating a TAO DS image-embedding output parquet."""

    def __call__(
        self,
        output_path: str | Path,
        *,
        data_dir: str,
        input_path: str | Path,
    ) -> dict[str, Any]:
        """Validate one image-embedding output parquet."""


@dataclass(frozen=True)
class BuildImageEmbeddingInputRequest:
    """Inputs for building a TAO DS image-embedding input parquet."""

    input_json: str
    data_dir: str
    output_parquet: str


@dataclass(frozen=True)
class ValidateImageEmbeddingInputRequest:
    """Inputs for validating a TAO DS image-embedding input parquet."""

    parquet_path: str
    data_dir: str


@dataclass(frozen=True)
class RunImageEmbeddingsRequest:
    """Inputs for running TAO DS image embeddings."""

    data_dir: str
    input_parquet: str | None = None
    output_parquet: str | None = None
    model_type: str | None = None
    model_name_or_path: str | None = None
    model_config_path: str | None = None
    config_file: str | None = None
    batch_size: int = 64
    dry_run: bool = False


@dataclass(frozen=True)
class ValidateImageEmbeddingOutputRequest:
    """Inputs for validating TAO DS image-embedding output parquet."""

    input_parquet: str
    output_parquet: str
    data_dir: str


def build_image_embedding_input_handoff(
    request: BuildImageEmbeddingInputRequest,
    *,
    build_input: ImageEmbeddingInputBuilder,
    validate_input: ImageEmbeddingInputValidator,
) -> dict[str, Any]:
    """Build and validate a TAO DS image-embedding input parquet."""
    payload = json.loads(Path(request.input_json).read_text(encoding="utf-8"))
    if not isinstance(payload, list) or any(not isinstance(row, dict) for row in payload):
        raise ValueError("--input-json must contain a JSON array of row objects")
    output = build_input(payload, request.output_parquet, data_dir=request.data_dir)
    summary = validate_input(output, data_dir=request.data_dir)
    return {"status": "built", **summary}


def validate_image_embedding_input_handoff(
    request: ValidateImageEmbeddingInputRequest,
    *,
    validate_input: ImageEmbeddingInputValidator,
) -> dict[str, Any]:
    """Validate a TAO DS image-embedding input parquet."""
    summary = validate_input(request.parquet_path, data_dir=request.data_dir)
    return {"status": "valid", **summary}


def run_image_embeddings(
    request: RunImageEmbeddingsRequest,
    *,
    runner_factory: Callable[[], ImageEmbeddingRunner],
) -> dict[str, Any]:
    """Run TAO DS image embeddings through the configured runner."""
    direct_values = (
        request.input_parquet,
        request.output_parquet,
        request.model_type,
        request.model_name_or_path,
    )
    if request.config_file:
        if any(value is not None for value in direct_values) or request.model_config_path:
            raise ValueError(
                "--config-file cannot be combined with direct model/input/output options"
            )
        result = runner_factory().image_embeddings_config(
            config_file=request.config_file,
            data_dir=request.data_dir,
            dry_run=request.dry_run,
        )
    else:
        if (
            request.input_parquet is None
            or request.output_parquet is None
            or request.model_type is None
            or request.model_name_or_path is None
        ):
            raise ValueError(
                "direct mode requires --input-parquet, --output-parquet, "
                "--model-type, and --model-name-or-path"
            )
        result = runner_factory().image_embeddings(
            data_dir=request.data_dir,
            input_parquet=request.input_parquet,
            output_parquet=request.output_parquet,
            model_type=request.model_type,
            model_name_or_path=request.model_name_or_path,
            model_config_path=request.model_config_path,
            batch_size=request.batch_size,
            dry_run=request.dry_run,
        )
    return {
        "status": "dry-run" if request.dry_run else "completed",
        "dry_run": request.dry_run,
        "command": list(result.command),
        "evidence": result.evidence,
    }


def validate_image_embedding_output_handoff(
    request: ValidateImageEmbeddingOutputRequest,
    *,
    validate_output: ImageEmbeddingOutputValidator,
) -> dict[str, Any]:
    """Validate TAO DS image-embedding output parquet."""
    summary = validate_output(
        request.output_parquet,
        data_dir=request.data_dir,
        input_path=request.input_parquet,
    )
    return {"status": "valid", **summary}


class TextEmbeddingRunner(Protocol):
    """Port for running TAO Data Services text embeddings."""

    def text_embeddings_config(
        self,
        *,
        config_file: str,
        data_dir: str,
        dry_run: bool = False,
    ) -> ImageEmbeddingRunResult:
        """Run text embeddings from an engine-native YAML."""

    def text_embeddings(
        self,
        *,
        data_dir: str,
        input_parquet: str,
        output_parquet: str,
        model: str,
        model_path: str,
        batch_size: int = 64,
        dry_run: bool = False,
    ) -> ImageEmbeddingRunResult:
        """Run text embeddings from direct options."""


@dataclass(frozen=True)
class RunTextEmbeddingsRequest:
    """Inputs for running TAO DS text embeddings."""

    data_dir: str
    input_parquet: str | None = None
    output_parquet: str | None = None
    model: str | None = None
    model_path: str | None = None
    config_file: str | None = None
    batch_size: int = 64
    dry_run: bool = False


@dataclass(frozen=True)
class ValidateTextEmbeddingOutputRequest:
    """Inputs for validating TAO DS text-embedding output parquet."""

    input_parquet: str
    output_parquet: str
    data_dir: str


def run_text_embeddings(
    request: RunTextEmbeddingsRequest,
    *,
    runner_factory: Callable[[], TextEmbeddingRunner],
) -> dict[str, Any]:
    """Run TAO DS text embeddings through the configured runner."""
    direct_values = (
        request.input_parquet,
        request.output_parquet,
        request.model,
        request.model_path,
    )
    if request.config_file:
        if any(value is not None for value in direct_values):
            raise ValueError(
                "--config-file cannot be combined with direct model/input/output options"
            )
        result = runner_factory().text_embeddings_config(
            config_file=request.config_file,
            data_dir=request.data_dir,
            dry_run=request.dry_run,
        )
    else:
        if any(value is None for value in direct_values):
            raise ValueError(
                "direct mode requires --input-parquet, --output-parquet, --model, and --model-path"
            )
        result = runner_factory().text_embeddings(
            data_dir=request.data_dir,
            input_parquet=str(request.input_parquet),
            output_parquet=str(request.output_parquet),
            model=str(request.model),
            model_path=str(request.model_path),
            batch_size=request.batch_size,
            dry_run=request.dry_run,
        )
    return {
        "status": "dry-run" if request.dry_run else "completed",
        "dry_run": request.dry_run,
        "command": list(result.command),
        "evidence": result.evidence,
    }


def validate_text_embedding_output_handoff(
    request: ValidateTextEmbeddingOutputRequest,
    *,
    validate_output: ImageEmbeddingOutputValidator,
) -> dict[str, Any]:
    """Validate TAO DS text-embedding output parquet."""
    summary = validate_output(
        request.output_parquet,
        data_dir=request.data_dir,
        input_path=request.input_parquet,
    )
    return {"status": "valid", **summary}
