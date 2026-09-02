# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Schema mapping between Dataset Search and Data Mining parquet contracts.

Dataset Search Curator/bulk ingest: ``id``, ``embedding``, ``$meta`` (JSON string or dict)
Data Mining / in-process adapters: ``file_name``, ``embedding`` (list)
TAO Toolkit DS ``tmm nearest_neighbors``: ``filepath``, ``embedding`` (list)

Behavior of each system is unchanged; this adapter only translates at the boundary.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from packages.domain.types import EmbeddingRecord

_DM_ID_KEYS = ("file_name", "filepath", "id")


def data_mining_row_to_record(row: Mapping[str, Any]) -> EmbeddingRecord:
    """Map a Data Mining / TMM row to the domain record."""
    record_id = ""
    for key in _DM_ID_KEYS:
        value = row.get(key)
        if value is not None and str(value) != "":
            record_id = str(value)
            break
    embedding = list(row["embedding"])
    meta = {k: v for k, v in row.items() if k not in (*_DM_ID_KEYS, "embedding")}
    return EmbeddingRecord(record_id=record_id, embedding=embedding, metadata=meta)


def dataset_search_row_to_record(row: Mapping[str, Any]) -> EmbeddingRecord:
    """Map a Dataset Search row (id + embedding + $meta) to the domain record."""
    record_id = str(row["id"])
    embedding = list(row["embedding"])
    raw_meta = row.get("$meta", row.get("meta", {}))
    if isinstance(raw_meta, str):
        meta: dict[str, Any] = json.loads(raw_meta) if raw_meta else {}
    elif isinstance(raw_meta, Mapping):
        meta = dict(raw_meta)
    else:
        meta = {}
    extras = {k: v for k, v in row.items() if k not in ("id", "embedding", "$meta", "meta")}
    meta.update(extras)
    return EmbeddingRecord(record_id=record_id, embedding=embedding, metadata=meta)


def record_to_data_mining_row(record: EmbeddingRecord) -> dict[str, Any]:
    """Domain record → in-process Data Mining parquet columns (``file_name``)."""
    row: dict[str, Any] = {
        "file_name": record.record_id,
        "embedding": list(record.embedding),
    }
    row.update(record.metadata)
    return row


def record_to_tmm_row(record: EmbeddingRecord) -> dict[str, Any]:
    """Domain record → TAO Toolkit DS TMM parquet columns (``filepath``)."""
    row: dict[str, Any] = {
        "filepath": record.record_id,
        "embedding": list(record.embedding),
    }
    row.update(record.metadata)
    return row


def record_to_dataset_search_row(record: EmbeddingRecord) -> dict[str, Any]:
    """Domain record → Dataset Search bulk-ingest columns (id, embedding, $meta)."""
    return {
        "id": record.record_id,
        "embedding": list(record.embedding),
        "$meta": json.dumps(dict(record.metadata)),
    }


def enrich_dataset_search_meta(
    record: EmbeddingRecord,
    *,
    cluster_id: int | None = None,
    diversity_rank: int | None = None,
    selected: bool | None = None,
) -> EmbeddingRecord:
    """Attach analytics fields into metadata for Dataset Search ingest/filter."""
    meta = dict(record.metadata)
    if cluster_id is not None:
        meta["cluster_id"] = cluster_id
    if diversity_rank is not None:
        meta["diversity_rank"] = diversity_rank
    if selected is not None:
        meta["selected"] = selected
    return EmbeddingRecord(
        record_id=record.record_id,
        embedding=record.embedding,
        metadata=meta,
    )


def records_to_data_mining_frame_dicts(records: Sequence[EmbeddingRecord]) -> list[dict[str, Any]]:
    return [record_to_data_mining_row(r) for r in records]


def records_to_dataset_search_frame_dicts(
    records: Sequence[EmbeddingRecord],
) -> list[dict[str, Any]]:
    return [record_to_dataset_search_row(r) for r in records]
