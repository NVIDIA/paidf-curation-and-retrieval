# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dataset Search request and response mappers."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from packages.domain.types import (
    CollectionInfo,
    DocumentSpec,
    SearchHit,
    SearchQuery,
)


def query_payload(query: SearchQuery) -> dict[str, Any]:
    """Map SearchQuery modalities to a CVDS QueryType object."""
    modalities = [
        query.text is not None,
        query.embedding is not None,
        query.image is not None or query.image_path is not None,
        query.video is not None,
        query.video_frames is not None,
        query.session_segment is not None,
    ]
    if sum(bool(m) for m in modalities) != 1:
        raise ValueError(
            "SearchQuery requires exactly one of: text, embedding, image/image_path, "
            "video, video_frames, or session_segment"
        )
    if query.text is not None:
        if not str(query.text).strip():
            raise ValueError("SearchQuery.text must be non-empty")
        return {"text": query.text}
    if query.embedding is not None:
        if len(query.embedding) == 0:
            raise ValueError("SearchQuery.embedding must be non-empty")
        return {"embedding": list(query.embedding)}
    if query.image is not None or query.image_path is not None:
        image_val = query.image if query.image is not None else query.image_path
        if not image_val:
            raise ValueError("SearchQuery.image/image_path must be non-empty")
        return {"image": image_val}
    if query.video is not None:
        if not query.video:
            raise ValueError("SearchQuery.video must be non-empty")
        return {"video": query.video}
    if query.video_frames is not None:
        if not query.video_frames:
            raise ValueError("SearchQuery.video_frames must be non-empty")
        return {"video_frames": query.video_frames}

    segment = dict(query.session_segment or {})
    required = ("session_id", "start_timestamp", "end_timestamp", "camera")
    missing = [key for key in required if key not in segment or segment[key] is None]
    if missing:
        raise ValueError(f"session_segment missing required keys: {missing}")
    return {"session_segment": segment}


def search_request_body(query: SearchQuery) -> dict[str, Any]:
    """Map a domain search query to a Dataset Search request body."""
    body: dict[str, Any] = {
        "query": [query_payload(query)],
        "top_k": query.top_k,
        "reconstruct": query.reconstruct,
        "generate_asset_url": query.generate_asset_url,
    }
    if query.filters:
        body["filters"] = (
            dict(query.filters) if isinstance(query.filters, Mapping) else query.filters
        )
    if query.search_params:
        body["search_params"] = dict(query.search_params)
    if query.clf:
        body["clf"] = dict(query.clf)
    return body


def parse_hits(body: Any) -> list[SearchHit]:
    """Map a Dataset Search hit response into domain hits."""
    items: Any
    if isinstance(body, list):
        items = body
    elif isinstance(body, dict):
        if "retrievals" in body:
            items = body["retrievals"]
        elif "results" in body:
            items = body["results"]
        else:
            raise ValueError("Unexpected search response: missing retrievals or results")
    else:
        raise ValueError("Unexpected search response: expected an object or list")
    if not isinstance(items, list):
        raise ValueError("Unexpected search response: results must be a list")

    hits: list[SearchHit] = []
    for item_number, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Unexpected search response item {item_number}: expected an object")
        record_id = item.get("id")
        if record_id is None:
            record_id = item.get("document_id")
        if record_id is None or not str(record_id):
            raise ValueError(f"Unexpected search response item {item_number}: missing id")
        score = item.get("score")
        if score is None:
            score = item.get("distance")
        if score is None:
            raise ValueError(f"Unexpected search response item {item_number}: missing score")
        raw_metadata = item.get("meta", item.get("metadata"))
        if raw_metadata is None:
            raw_metadata = {}
        if not isinstance(raw_metadata, Mapping):
            raise ValueError(
                f"Unexpected search response item {item_number}: metadata must be an object"
            )
        emb = item.get("embedding")
        hits.append(
            SearchHit(
                record_id=str(record_id),
                score=float(score),
                metadata=dict(raw_metadata),
                collection_id=(str(item["collection_id"]) if item.get("collection_id") else None),
                asset_url=item.get("asset_url"),
                embedding=list(emb) if emb is not None else None,
            )
        )
    return hits


def parse_collection(raw: Mapping[str, Any], total: int | None = None) -> CollectionInfo:
    """Map a Dataset Search collection response into a domain collection."""
    collection_id = raw.get("id")
    if collection_id is None:
        collection_id = raw.get("collection_id", "")
    return CollectionInfo(
        collection_id=str(collection_id),
        name=str(raw.get("name", "")),
        pipeline=str(raw.get("pipeline", "")),
        tags=dict(raw.get("tags") or {}),
        created_at=str(raw["created_at"]) if raw.get("created_at") is not None else None,
        total_documents_count=(total if total is not None else raw.get("total_documents_count")),
    )


def sanitize_document_metadata(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    """Keep only JSON-scalar metadata fields accepted by GA DocumentUpload schemas."""
    if not metadata:
        return {}
    clean: dict[str, Any] = {}
    for key, value in metadata.items():
        if value is None or isinstance(value, (str, int, float, bool)):
            clean[str(key)] = value
    return clean


def document_specs_from_cds_parquet(parquet_path: str) -> list[DocumentSpec]:
    """Load CDS-shaped parquet rows as embedding DocumentSpecs."""
    path = Path(parquet_path)
    if not path.is_file():
        raise FileNotFoundError(f"Parquet not found: {parquet_path}")
    docs: list[DocumentSpec] = []
    row_number = 0
    for batch in pq.ParquetFile(path).iter_batches():
        for row in batch.to_pylist():
            row_number += 1
            if not isinstance(row, dict):
                continue
            embedding = row.get("embedding")
            if embedding is None:
                raise ValueError(f"{parquet_path} row {row_number} is missing embedding")
            meta_raw = row.get("$meta", row.get("meta", {}))
            if isinstance(meta_raw, str):
                try:
                    meta_raw = json.loads(meta_raw)
                except json.JSONDecodeError:
                    meta_raw = {"raw_meta": meta_raw}
            if not isinstance(meta_raw, Mapping):
                meta_raw = {}
            doc_id = row.get("id")
            docs.append(
                DocumentSpec(
                    mime_type="application/octet-stream",
                    embedding=[float(x) for x in embedding],
                    document_id=str(doc_id) if doc_id is not None else None,
                    metadata=sanitize_document_metadata(meta_raw),
                )
            )
    return docs


def document_payload(doc: DocumentSpec) -> dict[str, Any]:
    """Map a document spec into the Dataset Search upload payload."""
    modes = [
        doc.content is not None,
        doc.url is not None,
        doc.embedding is not None,
    ]
    if sum(bool(m) for m in modes) != 1:
        raise ValueError("DocumentSpec requires exactly one of: content, url, or embedding")
    if not doc.mime_type:
        raise ValueError("DocumentSpec.mime_type is required")
    payload: dict[str, Any] = {"mime_type": doc.mime_type}
    if doc.document_id is not None:
        payload["id"] = doc.document_id
    if doc.metadata:
        payload["metadata"] = sanitize_document_metadata(doc.metadata)
    if doc.content is not None:
        if not doc.content:
            raise ValueError("DocumentSpec.content must be non-empty")
        payload["content"] = doc.content
    elif doc.url is not None:
        if not doc.url:
            raise ValueError("DocumentSpec.url must be non-empty")
        payload["url"] = doc.url
    else:
        if not doc.embedding:
            raise ValueError("DocumentSpec.embedding must be non-empty")
        payload["embedding"] = list(doc.embedding)
    return payload
