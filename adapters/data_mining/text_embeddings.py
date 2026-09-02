# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host-side handoff helpers for TAO DS ``embedding text_embeddings``.

This module owns the parquet/path contract used by Make and the integration
CLI before and after the container runs ``embedding text_embeddings``:

* resolve host or ``/data`` paths and keep them under ``DATA_DIR``
* require a non-empty ``text`` column and forbid a reserved ``embedding``
  column on input
* map model type / model path (HF id or ``DATA_DIR``-local artifact)
* verify output vectors (finite, uniform dimension) and metadata preservation
* preflight engine-native YAML experiment specs

Live model execution stays inside the TAO image; these helpers only build a
safe handoff and evidence payload for the runner.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from adapters.data_mining.image_embeddings import (
    ImageEmbeddingParquetError,
    container_data_path,
    resolve_data_path,
)
from adapters.data_mining.tmm_parquet import validate_embedding_dimension

TEXT_EMBEDDING_MODELS = {"clip": "CLIP", "siglip": "SigLIP", "siglip2": "SigLIP2"}
_REQUIRED_OUTPUT_COLUMNS = ("text", "embedding")


class TextEmbeddingParquetError(ValueError):
    """Text-embedding data does not satisfy the TAO DS handoff contract."""


def resolve_text_data_path(
    value: str | Path,
    data_dir: str | Path,
    *,
    role: str,
    must_exist: bool,
    file_only: bool = True,
) -> Path:
    """Resolve a host or ``/data`` path and require containment in ``DATA_DIR``."""
    try:
        return resolve_data_path(
            value,
            data_dir,
            role=role,
            must_exist=must_exist,
            file_only=file_only,
        )
    except ImageEmbeddingParquetError as exc:
        raise TextEmbeddingParquetError(str(exc)) from exc


def text_container_data_path(path: str | Path, data_dir: str | Path, *, role: str) -> str:
    """Map a contained host path to the mounted ``/data`` namespace."""
    try:
        return container_data_path(path, data_dir, role=role)
    except ImageEmbeddingParquetError as exc:
        raise TextEmbeddingParquetError(str(exc)) from exc


def validate_text_embedding_input_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Require a usable ``text`` column and no reserved embedding column."""
    if "text" not in frame.columns:
        raise TextEmbeddingParquetError("Input parquet missing required columns: ['text']")
    if "embedding" in frame.columns:
        raise TextEmbeddingParquetError(
            "Input parquet must not contain a reserved embedding column"
        )
    if frame.empty:
        raise TextEmbeddingParquetError("Input parquet must contain at least one row")
    for row_number, value in enumerate(frame["text"], start=1):
        if not isinstance(value, str) or not value.strip():
            raise TextEmbeddingParquetError(f"Text row {row_number} must be a non-empty string")
    return frame


def validate_text_embedding_input(
    parquet_path: str | Path,
    *,
    data_dir: str | Path,
) -> dict[str, Any]:
    """Preflight a text-embedding input parquet under ``DATA_DIR``.

    Requires a non-empty ``text`` column and rejects an already-populated
    ``embedding`` column. Returns row/column evidence for the CLI/Make path.
    """
    path = resolve_text_data_path(parquet_path, data_dir, role="Input parquet", must_exist=True)
    frame = validate_text_embedding_input_frame(pd.read_parquet(path))
    return {"path": str(path), "rows": len(frame), "columns": list(frame.columns)}


def validate_text_embedding_model(
    model: str,
    model_path: str,
    *,
    data_dir: str | Path,
) -> dict[str, str]:
    """Map operator model knobs onto vendor ``model`` / ``model_path`` fields.

    Accepts ``clip``, ``siglip``, or ``siglip2``. Hugging Face ids pass through;
    local artifacts must resolve under ``DATA_DIR`` and are rewritten to
    ``/data/...`` for the container experiment spec.
    """
    canonical = TEXT_EMBEDDING_MODELS.get(str(model).strip().lower().replace("-", ""))
    if canonical is None:
        raise TextEmbeddingParquetError(
            f"model must be one of: {', '.join(sorted(TEXT_EMBEDDING_MODELS))}"
        )
    if not isinstance(model_path, str) or not model_path.strip():
        raise TextEmbeddingParquetError("model path must be a non-empty string")
    resolved_path = model_path.strip()
    supplied = Path(resolved_path)
    is_container_path = resolved_path == "/data" or resolved_path.startswith("/data/")
    is_local = is_container_path or supplied.is_absolute()
    if not is_local and (Path(data_dir) / supplied).exists():
        is_local = True
    if is_local:
        host_model = resolve_text_data_path(
            resolved_path,
            data_dir,
            role="Model",
            must_exist=True,
            file_only=False,
        )
        resolved_path = text_container_data_path(host_model, data_dir, role="Model")
    return {"model": canonical, "model_path": resolved_path}


def validate_text_embedding_output(
    output_path: str | Path,
    *,
    data_dir: str | Path,
    input_path: str | Path | None = None,
) -> dict[str, Any]:
    """Post-check text-embedding output vectors and optional input fidelity.

    Requires ``text`` + ``embedding``, finite uniform dimensions, and—when
    ``input_path`` is supplied—matching row count, text values, and preserved
    metadata columns.
    """
    output = resolve_text_data_path(output_path, data_dir, role="Output parquet", must_exist=True)
    frame = pd.read_parquet(output)
    missing = [column for column in _REQUIRED_OUTPUT_COLUMNS if column not in frame.columns]
    if missing:
        raise TextEmbeddingParquetError(f"Output parquet missing required columns: {missing}")
    if frame.empty:
        raise TextEmbeddingParquetError("Output parquet must contain at least one row")
    dimension = validate_embedding_dimension(frame, error_type=TextEmbeddingParquetError)

    metadata_columns: list[str] = []
    if input_path is not None:
        source = resolve_text_data_path(input_path, data_dir, role="Input parquet", must_exist=True)
        input_frame = validate_text_embedding_input_frame(pd.read_parquet(source))
        if len(input_frame) != len(frame):
            raise TextEmbeddingParquetError(
                f"Output parquet row count {len(frame)} does not match input {len(input_frame)}"
            )
        # text_embeddings joins metadata positionally because captions can repeat.
        if list(input_frame["text"]) != list(frame["text"]):
            raise TextEmbeddingParquetError("Output text values do not match the input parquet")
        metadata_columns = [
            column for column in input_frame.columns if column not in _REQUIRED_OUTPUT_COLUMNS
        ]
        missing_metadata = [column for column in metadata_columns if column not in frame.columns]
        if missing_metadata:
            raise TextEmbeddingParquetError(
                f"Output parquet missing preserved metadata columns: {missing_metadata}"
            )

    return {
        "path": str(output),
        "rows": len(frame),
        "columns": list(frame.columns),
        "dimension": dimension,
        "metadata_columns": metadata_columns,
    }


def validate_text_embedding_config(
    config_file: str | Path,
    *,
    data_dir: str | Path,
) -> dict[str, Any]:
    """Preflight an engine-native text-embedding YAML before Docker execution.

    Confirms required string fields, resolves parquet/model paths under
    ``DATA_DIR``, and reuses the input/model checks above.
    """
    config_path = Path(config_file)
    if not config_path.is_file():
        raise FileNotFoundError(f"Text embedding config not found: {config_path}")
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise TextEmbeddingParquetError(
            f"Unable to read text embedding config: {config_path}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise TextEmbeddingParquetError("Text embedding config must contain a mapping")

    required = ("input_parquet", "output_parquet", "model", "model_path")
    missing = [key for key in required if not isinstance(payload.get(key), str)]
    if missing:
        raise TextEmbeddingParquetError(
            f"Text embedding config missing required string fields: {missing}"
        )
    input_path = resolve_text_data_path(
        payload["input_parquet"], data_dir, role="Input parquet", must_exist=True
    )
    output_path = resolve_text_data_path(
        payload["output_parquet"],
        data_dir,
        role="Output parquet",
        must_exist=False,
    )
    validate_text_embedding_input(input_path, data_dir=data_dir)
    model = validate_text_embedding_model(
        payload["model"],
        payload["model_path"],
        data_dir=data_dir,
    )
    batch_size = payload.get("batch_size", 64)
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise TextEmbeddingParquetError("batch_size must be a positive integer")
    return {
        "input_path": input_path,
        "output_path": output_path,
        "model": model,
        "batch_size": batch_size,
    }
