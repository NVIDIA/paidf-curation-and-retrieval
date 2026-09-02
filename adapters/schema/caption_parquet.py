# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build and validate the CDS EA external-caption parquet contract."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

_REQUIRED_COLUMNS = ("clip_id", "summary")
_OPTIONAL_COLUMNS = ("start_time", "end_time", "model_name", "data_source")


class CaptionParquetError(ValueError):
    """Caption rows do not satisfy the EA upload contract."""


def _clean_identity(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def load_indexed_clip_ids(path: str | Path) -> set[str]:
    """Load clip identities from a JSON manifest/list or newline-delimited file."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Indexed-ID file not found: {source}")
    text = source.read_text(encoding="utf-8").strip()
    if not text:
        raise CaptionParquetError("Indexed-ID file must not be empty")

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        values: Iterable[Any] = text.splitlines()
    else:
        if isinstance(payload, Mapping):
            values = payload.get("clip_ids", payload.get("ids", payload.get("items", [])))
        else:
            values = payload
        if not isinstance(values, list):
            raise CaptionParquetError(
                "Indexed-ID JSON must be a list or an object containing clip_ids, ids, or items"
            )

    identities: list[str] = []
    for item in values:
        if isinstance(item, Mapping):
            item = item.get("clip_id", item.get("id"))
        identities.append(_clean_identity(item))
    if not identities or any(not identity for identity in identities):
        raise CaptionParquetError("Indexed clip identities must be non-empty")
    if len(set(identities)) != len(identities):
        raise CaptionParquetError("Indexed clip identities must be unique")
    return set(identities)


def validate_caption_frame(
    frame: pd.DataFrame,
    *,
    indexed_clip_ids: set[str] | None = None,
) -> pd.DataFrame:
    """Validate required fields, unique identities, timing, and optional ID alignment."""
    missing = [column for column in _REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise CaptionParquetError(f"Caption parquet missing required columns: {missing}")
    if frame.empty:
        raise CaptionParquetError("Caption parquet must contain at least one row")

    clean = frame.copy()
    clean["clip_id"] = clean["clip_id"].map(_clean_identity)
    clean["summary"] = clean["summary"].map(_clean_identity)
    if (clean["clip_id"] == "").any():
        raise CaptionParquetError("clip_id values must be non-empty")
    if clean["clip_id"].duplicated().any():
        raise CaptionParquetError("clip_id values must be unique")
    if (clean["summary"] == "").any():
        raise CaptionParquetError("summary values must be non-empty")

    for column in ("start_time", "end_time"):
        if column in clean.columns:
            try:
                clean[column] = pd.to_numeric(clean[column], errors="raise")
            except (TypeError, ValueError) as exc:
                raise CaptionParquetError(f"{column} values must be numeric") from exc
    if {"start_time", "end_time"}.issubset(clean.columns):
        invalid = (clean["start_time"] >= 0) & (
            (clean["end_time"] < 0) | (clean["end_time"] < clean["start_time"])
        )
        if invalid.any():
            raise CaptionParquetError(
                "end_time must be greater than or equal to start_time when timing is provided"
            )

    if indexed_clip_ids is not None:
        caption_ids = set(clean["clip_id"])
        missing_captions = indexed_clip_ids - caption_ids
        unknown_captions = caption_ids - indexed_clip_ids
        if missing_captions or unknown_captions:
            raise CaptionParquetError(
                "Caption clip identities do not exactly align with the supplied indexed-ID set "
                f"(missing={len(missing_captions)}, unknown={len(unknown_captions)})"
            )
    return clean


def build_caption_parquet(
    rows: Sequence[Mapping[str, Any]],
    output_path: str | Path,
    *,
    indexed_clip_ids: set[str] | None = None,
) -> Path:
    """Write validated `clip_id`/`summary` rows for CDS EA caption upload."""
    frame = validate_caption_frame(pd.DataFrame(list(rows)), indexed_clip_ids=indexed_clip_ids)
    columns = [column for column in (*_REQUIRED_COLUMNS, *_OPTIONAL_COLUMNS) if column in frame]
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.loc[:, columns].to_parquet(output, index=False)
    return output


def validate_caption_parquet(
    parquet_path: str | Path,
    *,
    indexed_clip_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Validate an existing parquet and return a minimal machine-readable summary."""
    path = Path(parquet_path)
    if not path.is_file():
        raise FileNotFoundError(f"Caption parquet not found: {path}")
    clean = validate_caption_frame(
        pd.read_parquet(path),
        indexed_clip_ids=indexed_clip_ids,
    )
    return {
        "path": str(path),
        "rows": len(clean),
        "columns": list(clean.columns),
        "identity_aligned": indexed_clip_ids is not None,
    }
