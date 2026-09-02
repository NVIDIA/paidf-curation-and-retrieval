# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convert Cosmos Curator export layout to Dataset Search ingest parquet.

Contract (same as Dataset Search curator_parquet_converter):
  {base_dir}/iv2_embd_parquet/*.parquet              InternVideo2
  {base_dir}/ce1_embd_parquet/*.parquet              Cosmos-Embed1
  {base_dir}/ce1_embd_<variant>_parquet/*.parquet    Cosmos-Embed1 variant dirs
  columns: id, embedding
  {base_dir}/metas/v0/{uuid}.json                    span_uuid + metadata
Output parquet columns: id, embedding, $meta
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from packages.domain.types import EmbeddingRecord

_IV2_DIR_NAME = "iv2_embd_parquet"
_CE1_DIR_NAME = "ce1_embd_parquet"
_CE1_VARIANT_GLOB = "ce1_embd_*_parquet"


class CuratorExportError(ValueError):
    """Invalid Curator export layout or UUID mismatch."""


def _list_parquet_files(parquet_dir: Path) -> list[Path]:
    if not parquet_dir.is_dir():
        raise CuratorExportError(f"Missing Curator parquet dir: {parquet_dir}")
    files = sorted(parquet_dir.glob("**/*.parquet"))
    if not files:
        files = sorted(parquet_dir.glob("*.parquet"))
    if not files:
        raise CuratorExportError(f"No parquet files under {parquet_dir}")
    return files


def _dir_has_parquets(path: Path) -> bool:
    if not path.is_dir():
        return False
    return bool(list(path.glob("**/*.parquet")) or list(path.glob("*.parquet")))


def find_embedding_parquet_dir(
    base_dir: str | Path,
    backend: str = "auto",
) -> Path:
    """Locate Curator embedding parquet dir (IV2 or Cosmos-Embed1).

    ``backend``: ``auto`` | ``iv2`` | ``ce1`` / ``cosmos-embed1``.
    """
    root = Path(base_dir)
    if not root.is_dir():
        raise CuratorExportError(f"Curator output dir missing: {root}")

    normalized = backend.strip().lower()
    if normalized in {"ce1", "cosmos-embed1", "cosmos_embed1"}:
        normalized = "ce1"
    if normalized not in {"auto", "iv2", "ce1"}:
        raise CuratorExportError(f"Unknown embedding backend {backend!r}; use auto, iv2, or ce1")

    candidates: list[Path] = []
    if normalized in {"auto", "iv2"}:
        candidates.append(root / _IV2_DIR_NAME)
        candidates.extend(sorted(root.rglob(_IV2_DIR_NAME)))
    if normalized in {"auto", "ce1"}:
        candidates.append(root / _CE1_DIR_NAME)
        candidates.extend(sorted(root.glob(_CE1_VARIANT_GLOB)))
        candidates.extend(sorted(root.rglob(_CE1_DIR_NAME)))
        candidates.extend(sorted(root.rglob(_CE1_VARIANT_GLOB)))

    seen: set[Path] = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if _dir_has_parquets(path):
            return path

    if normalized == "iv2":
        raise CuratorExportError(
            f"No {_IV2_DIR_NAME}/ with parquet under {root} (required for InternVideo2 ingest)"
        )
    if normalized == "ce1":
        raise CuratorExportError(
            f"No {_CE1_DIR_NAME}/ or ce1_embd_*_parquet/ with parquet under {root} "
            "(required for Cosmos-Embed1 ingest)"
        )
    raise CuratorExportError(
        f"No iv2_embd_parquet/ or ce1_embd*_parquet/ with parquet under {root}"
    )


def _record_from_row(row: dict[str, Any], base_dir: Path) -> EmbeddingRecord:
    uuid = str(row["id"])
    embedding = list(row["embedding"])
    meta_path = base_dir / "metas" / "v0" / f"{uuid}.json"
    if not meta_path.is_file():
        # Allow missing meta with empty dict (some exports omit sidecars)
        return EmbeddingRecord(record_id=uuid, embedding=embedding, metadata={})

    with meta_path.open("r", encoding="utf-8") as fh:
        metadata = json.load(fh)
    span = metadata.get("span_uuid")
    if span is not None and str(span) != uuid:
        raise CuratorExportError(f"UUID mismatch: parquet={uuid} meta.span_uuid={span}")
    return EmbeddingRecord(record_id=uuid, embedding=embedding, metadata=metadata)


def process_single_file(parquet_path: Path, base_dir: Path) -> Sequence[EmbeddingRecord]:
    """Map every Curator parquet row and its sidecar JSON to embedding records."""
    df = pd.read_parquet(parquet_path)
    if "id" not in df.columns or "embedding" not in df.columns:
        raise CuratorExportError(f"{parquet_path} must contain 'id' and 'embedding' columns")
    if df.empty:
        raise CuratorExportError(f"{parquet_path} must contain at least one row")
    rows: list[dict[str, Any]] = df.loc[:, ["id", "embedding"]].to_dict(orient="records")
    return [_record_from_row(row, base_dir) for row in rows]


def load_curator_export(
    base_dir: str | Path,
    backend: str = "auto",
) -> Sequence[EmbeddingRecord]:
    """Load all Curator embedding rows from a split/dedup output directory."""
    root = Path(base_dir)
    parquet_dir = find_embedding_parquet_dir(root, backend=backend)
    records: list[EmbeddingRecord] = []
    for path in _list_parquet_files(parquet_dir):
        records.extend(process_single_file(path, root))
    return records


def convert_curator_dir_to_cds_parquet(
    base_dir: str | Path,
    output_path: str | Path,
    backend: str = "auto",
) -> Path:
    """Write CDS-shaped parquet (id, embedding, $meta) for /insert-data."""
    from adapters.schema.mapper import records_to_dataset_search_frame_dicts

    records = load_curator_export(base_dir, backend=backend)
    rows = records_to_dataset_search_frame_dicts(records)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out, index=False)
    return out


class CuratorExportAdapter:
    """Implements Curator export → domain records / CDS parquet files."""

    def load(self, base_dir: str, backend: str = "auto") -> Sequence[EmbeddingRecord]:
        return load_curator_export(base_dir, backend=backend)

    def to_cds_parquet(self, base_dir: str, output_path: str, backend: str = "auto") -> str:
        return str(convert_curator_dir_to_cds_parquet(base_dir, output_path, backend=backend))
