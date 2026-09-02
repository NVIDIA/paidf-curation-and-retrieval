# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host-side handoff helpers for TAO DS ``embedding image_embeddings``.

Builds the operator input parquet from JSON rows, then checks the
input/output/model contract used by Make and the integration CLI:

* image ``filepath`` rows must resolve to non-empty files under ``DATA_DIR``
* input must not already carry a reserved ``embedding`` column
* model type is ``clip`` or ``siglip`` (HF id or local TAO checkpoint)
* output must preserve identity/metadata and emit finite uniform vectors

Live CLIP/SigLIP inference runs in the TAO container; this module only
prepares and verifies the parquet handoff around that call.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from numbers import Real
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from adapters.data_mining.tmm_parquet import validate_embedding_dimension

_REQUIRED_INPUT_COLUMNS = ("filepath",)
_REQUIRED_OUTPUT_COLUMNS = ("filepath", "embedding")
_TAO_CHECKPOINT_SUFFIXES = {".ckpt", ".pth"}
_SUPPORTED_MODEL_TYPES = {"clip": "CLIP", "siglip": "SigLIP"}


class ImageEmbeddingParquetError(ValueError):
    """Image-embedding data does not satisfy the TAO DS handoff contract."""


def _data_root(data_dir: str | Path) -> Path:
    root = Path(data_dir).resolve()
    if not root.is_dir():
        raise ImageEmbeddingParquetError(f"DATA_DIR not found: {root}")
    return root


def _clean_path(value: Any, role: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ImageEmbeddingParquetError(f"{role} path must be a non-empty string")
    cleaned = value.strip()
    if any(ord(character) < 32 for character in cleaned):
        raise ImageEmbeddingParquetError(f"{role} path contains control characters")
    return cleaned


def resolve_data_path(
    value: str | Path,
    data_dir: str | Path,
    *,
    role: str,
    must_exist: bool,
    file_only: bool = True,
) -> Path:
    """Resolve a host or ``/data`` path and require containment in ``DATA_DIR``."""
    root = _data_root(data_dir)
    cleaned = _clean_path(str(value), role)
    if cleaned == "/data" or cleaned.startswith("/data/"):
        relative = cleaned.removeprefix("/data").lstrip("/")
        candidate = root / relative
    else:
        supplied = Path(cleaned)
        candidate = supplied if supplied.is_absolute() else root / supplied
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ImageEmbeddingParquetError(
            f"{role} path must be contained in DATA_DIR: {cleaned}"
        ) from exc
    if must_exist and not resolved.exists():
        raise ImageEmbeddingParquetError(f"{role} path not found: {cleaned}")
    if must_exist and file_only and not resolved.is_file():
        raise ImageEmbeddingParquetError(f"{role} path must be a file: {cleaned}")
    return resolved


def container_data_path(path: str | Path, data_dir: str | Path, *, role: str) -> str:
    """Map a contained host path to the mounted ``/data`` namespace."""
    root = _data_root(data_dir)
    resolved = resolve_data_path(
        path,
        root,
        role=role,
        must_exist=False,
        file_only=False,
    )
    return f"/data/{resolved.relative_to(root).as_posix()}"


def _validate_columns(frame: pd.DataFrame, required: Sequence[str], role: str) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ImageEmbeddingParquetError(f"{role} parquet missing required columns: {missing}")
    if frame.empty:
        raise ImageEmbeddingParquetError(f"{role} parquet must contain at least one row")
    if any(not isinstance(column, str) or not column.strip() for column in frame.columns):
        raise ImageEmbeddingParquetError(f"{role} parquet column names must be non-empty strings")


def _metadata_equal(left: Any, right: Any) -> bool:
    if isinstance(left, Real) and isinstance(right, Real):
        if math.isnan(float(left)) and math.isnan(float(right)):
            return True
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return left.keys() == right.keys() and all(
            _metadata_equal(left[key], right[key]) for key in left
        )
    if hasattr(left, "tolist"):
        left = left.tolist()
    if hasattr(right, "tolist"):
        right = right.tolist()
    if (
        isinstance(left, Sequence)
        and not isinstance(left, (str, bytes))
        and isinstance(right, Sequence)
        and not isinstance(right, (str, bytes))
    ):
        return len(left) == len(right) and all(
            _metadata_equal(left_value, right_value)
            for left_value, right_value in zip(left, right, strict=True)
        )
    try:
        return bool(left == right)
    except (TypeError, ValueError):
        return False


def validate_image_embedding_input_frame(
    frame: pd.DataFrame,
    data_dir: str | Path,
    *,
    require_local_images: bool = True,
) -> pd.DataFrame:
    """Normalize image identity rows for the TAO ``image_embeddings`` input.

    Requires ``filepath``, rejects a reserved ``embedding`` column, checks
    unique non-empty files under ``DATA_DIR`` when requested, and rewrites
    paths to the mounted ``/data/...`` namespace.
    """
    _validate_columns(frame, _REQUIRED_INPUT_COLUMNS, "Input")
    if "embedding" in frame.columns:
        raise ImageEmbeddingParquetError(
            "Input parquet must not contain a reserved embedding column"
        )

    clean = frame.copy()
    normalized: list[str] = []
    for row_number, value in enumerate(clean["filepath"], start=1):
        role = f"Image row {row_number}"
        if require_local_images:
            host_path = resolve_data_path(value, data_dir, role=role, must_exist=True)
            if host_path.stat().st_size == 0:
                raise ImageEmbeddingParquetError(f"{role} path must be a non-empty file")
            normalized.append(container_data_path(host_path, data_dir, role=role))
        else:
            normalized.append(_clean_path(value, role))
    clean["filepath"] = normalized
    if clean["filepath"].duplicated().any():
        raise ImageEmbeddingParquetError("Input filepath values must be unique")
    return clean


def build_image_embedding_input(
    rows: Sequence[Mapping[str, Any]],
    output_path: str | Path,
    *,
    data_dir: str | Path,
) -> Path:
    """Write a TAO-ready input parquet from JSON-like image rows.

    Preserves extra metadata columns and runs the input-frame checks above
    before writing under ``DATA_DIR``.
    """
    frame = validate_image_embedding_input_frame(pd.DataFrame(list(rows)), data_dir)
    output = resolve_data_path(
        output_path,
        data_dir,
        role="Input parquet output",
        must_exist=False,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output, index=False)
    return output


def validate_image_embedding_input(
    parquet_path: str | Path,
    *,
    data_dir: str | Path,
    require_local_images: bool = True,
) -> dict[str, Any]:
    """Preflight an existing image-embedding input parquet under ``DATA_DIR``."""
    path = resolve_data_path(parquet_path, data_dir, role="Input parquet", must_exist=True)
    clean = validate_image_embedding_input_frame(
        pd.read_parquet(path),
        data_dir,
        require_local_images=require_local_images,
    )
    return {"path": str(path), "rows": len(clean), "columns": list(clean.columns)}


def validate_image_embedding_output(
    output_path: str | Path,
    *,
    data_dir: str | Path,
    input_path: str | Path | None = None,
) -> dict[str, Any]:
    """Post-check image-embedding vectors and optional input fidelity.

    Requires ``filepath`` + ``embedding``, unique identities, finite uniform
    dimensions, and—when ``input_path`` is supplied—matching rows and
    preserved metadata values.
    """
    output = resolve_data_path(output_path, data_dir, role="Output parquet", must_exist=True)
    output_frame = pd.read_parquet(output)
    _validate_columns(output_frame, _REQUIRED_OUTPUT_COLUMNS, "Output")
    output_frame = output_frame.copy()
    output_frame["filepath"] = output_frame["filepath"].map(
        lambda value: _clean_path(value, "Output filepath")
    )
    if output_frame["filepath"].duplicated().any():
        raise ImageEmbeddingParquetError("Output filepath values must be unique")
    dimension = validate_embedding_dimension(
        output_frame,
        error_type=ImageEmbeddingParquetError,
    )

    metadata_columns: list[str] = []
    if input_path is not None:
        source = resolve_data_path(input_path, data_dir, role="Input parquet", must_exist=True)
        input_frame = validate_image_embedding_input_frame(pd.read_parquet(source), data_dir)
        metadata_columns = [
            column for column in input_frame.columns if column not in _REQUIRED_OUTPUT_COLUMNS
        ]
        missing_metadata = [
            column for column in metadata_columns if column not in output_frame.columns
        ]
        if missing_metadata:
            raise ImageEmbeddingParquetError(
                f"Output parquet missing preserved metadata columns: {missing_metadata}"
            )
        if set(input_frame["filepath"]) != set(output_frame["filepath"]):
            raise ImageEmbeddingParquetError(
                "Output filepath values do not match the input parquet"
            )
        input_rows = {row["filepath"]: row for row in input_frame.to_dict(orient="records")}
        output_rows = {row["filepath"]: row for row in output_frame.to_dict(orient="records")}
        metadata_matches = all(
            _metadata_equal(input_rows[filepath][column], output_rows[filepath][column])
            for filepath in input_rows
            for column in metadata_columns
        )
        if not metadata_matches:
            raise ImageEmbeddingParquetError(
                "Output metadata values do not match the input parquet"
            )

    return {
        "path": str(output),
        "rows": len(output_frame),
        "columns": list(output_frame.columns),
        "dimension": dimension,
        "metadata_columns": metadata_columns,
    }


def validate_image_embedding_model(
    model_type: str,
    model_name_or_path: str,
    *,
    data_dir: str | Path,
    model_config_path: str | None = None,
) -> dict[str, str]:
    """Map operator model knobs onto vendor ``model`` / ``model_path`` fields.

    Accepts ``clip`` or ``siglip``. Hugging Face ids pass through; local TAO
    checkpoints must resolve under ``DATA_DIR`` (CLIP requires a sibling
    config path) and are rewritten to ``/data/...``.
    """
    canonical_model = _SUPPORTED_MODEL_TYPES.get(model_type.strip().lower())
    if canonical_model is None:
        raise ImageEmbeddingParquetError("model type must be one of: clip, siglip")
    model_path = _clean_path(model_name_or_path, "Model")
    suffix = Path(model_path).suffix.lower()
    is_checkpoint = suffix in _TAO_CHECKPOINT_SUFFIXES
    is_container_path = model_path == "/data" or model_path.startswith("/data/")
    supplied_path = Path(model_path)
    is_local = is_checkpoint or is_container_path or supplied_path.is_absolute()
    if not is_local and (Path(data_dir) / supplied_path).exists():
        is_local = True

    if is_local:
        host_model = resolve_data_path(model_path, data_dir, role="Model", must_exist=True)
        model_path = container_data_path(host_model, data_dir, role="Model")
    if is_checkpoint and canonical_model != "CLIP":
        raise ImageEmbeddingParquetError("TAO checkpoints are supported only for model type clip")
    if is_checkpoint and not model_config_path:
        raise ImageEmbeddingParquetError("model config path is required for a TAO checkpoint")
    if not is_checkpoint and model_config_path:
        raise ImageEmbeddingParquetError("model config path is valid only for a TAO checkpoint")

    result = {"model": canonical_model, "model_path": model_path}
    if model_config_path:
        host_config = resolve_data_path(
            model_config_path,
            data_dir,
            role="Model config",
            must_exist=True,
        )
        result["model_config_path"] = container_data_path(
            host_config,
            data_dir,
            role="Model config",
        )
    return result


def validate_image_embedding_config(
    config_file: str | Path,
    *,
    data_dir: str | Path,
) -> dict[str, Any]:
    """Preflight an engine-native image-embedding YAML before Docker execution.

    Confirms required fields, resolves parquet/model paths under ``DATA_DIR``,
    and reuses the input/model checks above.
    """
    config_path = Path(config_file)
    if not config_path.is_file():
        raise FileNotFoundError(f"Image embedding config not found: {config_path}")
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ImageEmbeddingParquetError(
            f"Unable to read image embedding config: {config_path}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise ImageEmbeddingParquetError("Image embedding config must contain a mapping")

    required = ("input_parquet", "output_parquet", "model", "model_path")
    missing = [key for key in required if not isinstance(payload.get(key), str)]
    if missing:
        raise ImageEmbeddingParquetError(
            f"Image embedding config missing required string fields: {missing}"
        )
    input_path = resolve_data_path(
        payload["input_parquet"],
        data_dir,
        role="Input parquet",
        must_exist=True,
    )
    output_path = resolve_data_path(
        payload["output_parquet"],
        data_dir,
        role="Output parquet",
        must_exist=False,
    )
    validate_image_embedding_input(input_path, data_dir=data_dir)
    model = validate_image_embedding_model(
        payload["model"],
        payload["model_path"],
        data_dir=data_dir,
        model_config_path=payload.get("model_config_path") or None,
    )
    batch_size = payload.get("batch_size", 64)
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ImageEmbeddingParquetError("batch_size must be a positive integer")
    return {
        "input_path": input_path,
        "output_path": output_path,
        "model": model,
        "batch_size": batch_size,
    }
