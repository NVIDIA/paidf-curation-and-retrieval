# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dataset Search application workflows."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from packages.domain.types import (
    BulkInsertRequest,
    BulkJobStatus,
    CollectionRef,
    DocumentSpec,
    LabelledDocuments,
    MultiCollectionQuery,
    SearchHit,
    SearchQuery,
    SearchRefinementSpec,
)
from packages.ports import IngestPort, RetrievalPort, SearchRefinementPort

_FAILED_JOB_STATES = frozenset({"cancelled", "canceled", "failed", "error"})


class BulkJobWaitPort(Protocol):
    """Port extension for polling Dataset Search bulk jobs."""

    def wait_for_job(
        self,
        job_id: str,
        *,
        timeout_seconds: float = 900.0,
        poll_interval_seconds: float = 5.0,
    ) -> BulkJobStatus:
        """Wait for one job status."""


class DatasetSearchCommandClient(
    RetrievalPort,
    IngestPort,
    SearchRefinementPort,
    BulkJobWaitPort,
    Protocol,
):
    """Dataset Search ports needed by non-trivial command workflows."""


@dataclass(frozen=True)
class SearchQueryInput:
    """Raw search query inputs from a delivery mechanism."""

    text: str | None = None
    embedding: str | None = None
    image: str | None = None
    video: str | None = None
    video_frames: str | None = None
    top_k: int = 10
    filters_json: str | None = None
    generate_asset_url: bool = False


@dataclass(frozen=True)
class SubmitBulkInsertRequest:
    """Inputs for submitting a Dataset Search bulk-insert job."""

    collection: str
    embedding_family: str
    parquet_paths: Sequence[str]
    access_key: str | None = None
    secret_key: str | None = None
    endpoint_url: str | None = None
    allow_lab_http_endpoint: bool = False


@dataclass(frozen=True)
class JobStatusRequest:
    """Inputs for reading or waiting for a Dataset Search job."""

    job_id: str
    wait: bool = False
    timeout_seconds: float = 900.0
    poll_interval_seconds: float = 5.0


def _search_hit_payload(hit: SearchHit) -> dict[str, Any]:
    return {
        "id": hit.record_id,
        "score": hit.score,
        "meta": dict(hit.metadata),
        "asset_url": hit.asset_url,
        "collection_id": hit.collection_id,
    }


def job_status_payload(status: BulkJobStatus) -> dict[str, Any]:
    """Return the stable JSON payload for one Dataset Search bulk job."""
    return {
        "job_id": status.job_id,
        "status": status.status,
        "details": status.details,
        "progress": status.progress,
        "collection_name": status.collection_name,
    }


def is_failed_job_status(status: str) -> bool:
    """Return whether a normalized Dataset Search job state is failed."""
    return status.strip().lower() in _FAILED_JOB_STATES


def _parse_filters(raw_filters: str | None) -> Mapping[str, Any] | str:
    if not raw_filters:
        return {}
    try:
        filters = json.loads(raw_filters)
    except json.JSONDecodeError:
        return raw_filters
    if isinstance(filters, dict) or isinstance(filters, str):
        return filters
    raise ValueError("--filters-json must be a JSON object or Milvus expr string")


def build_search_query(request: SearchQueryInput) -> SearchQuery:
    """Build a domain search query from raw command inputs."""
    embedding = None
    if request.embedding:
        raw_embedding = json.loads(request.embedding)
        if not isinstance(raw_embedding, list):
            raise ValueError("--embedding must be a JSON array of floats")
        embedding = []
        for value in raw_embedding:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("--embedding must be a JSON array of floats")
            embedding.append(float(value))
    filters = _parse_filters(request.filters_json)
    return SearchQuery(
        text=request.text,
        embedding=embedding,
        image=request.image,
        video=request.video,
        video_frames=request.video_frames,
        top_k=request.top_k,
        filters=filters,
        generate_asset_url=request.generate_asset_url,
    )


def search_collection_handoff(
    *,
    collection: str,
    query: SearchQueryInput,
    client: DatasetSearchCommandClient,
) -> list[dict[str, Any]]:
    """Search within one collection and return JSON-ready hits."""
    hits = client.search(CollectionRef(collection_id=collection), build_search_query(query))
    return [_search_hit_payload(hit) for hit in hits]


def retrieve_collections_handoff(
    *,
    collections: Sequence[str],
    query: SearchQueryInput,
    rerank: bool,
    generate_asset_url: bool,
    client: DatasetSearchCommandClient,
) -> list[dict[str, Any]]:
    """Search across collections and return JSON-ready hits."""
    hits = client.multi_collection_search(
        MultiCollectionQuery(
            collection_ids=list(collections),
            query=build_search_query(query),
            rerank=rerank,
            generate_asset_url=generate_asset_url,
        )
    )
    return [_search_hit_payload(hit) for hit in hits]


def ingest_documents_handoff(
    *,
    collection: str,
    document_jsons: Sequence[str],
    client: DatasetSearchCommandClient,
) -> list[dict[str, str | None]]:
    """Index document specs from JSON command inputs."""
    docs: list[DocumentSpec] = []
    for raw in document_jsons:
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("--document-json must contain a JSON object")
        if "mime_type" not in data:
            raise ValueError("--document-json must include mime_type")
        raw_metadata = data.get("metadata")
        if raw_metadata is not None and not isinstance(raw_metadata, dict):
            raise ValueError("--document-json metadata must be a JSON object")
        metadata = raw_metadata or {}
        docs.append(
            DocumentSpec(
                mime_type=data["mime_type"],
                content=data.get("content"),
                url=data.get("url"),
                embedding=data.get("embedding"),
                document_id=data.get("id"),
                metadata=metadata,
            )
        )
    infos = client.ingest_documents(CollectionRef(collection_id=collection), docs)
    return [{"id": doc.document_id, "indexed_at": doc.indexed_at} for doc in infos]


def delete_document_handoff(
    *,
    collection: str,
    document_id: str,
    client: DatasetSearchCommandClient,
) -> dict[str, str]:
    """Delete one document from one collection."""
    client.delete_document(CollectionRef(collection_id=collection), document_id)
    return {"deleted": document_id, "collection": collection}


def submit_bulk_insert_handoff(
    request: SubmitBulkInsertRequest,
    *,
    client: DatasetSearchCommandClient,
) -> dict[str, Any]:
    """Submit one bulk-insert job."""
    status = client.bulk_insert(
        BulkInsertRequest(
            collection_name=request.collection,
            parquet_paths=list(request.parquet_paths),
            embedding_family=request.embedding_family.lower(),
            access_key=request.access_key,
            secret_key=request.secret_key,
            endpoint_url=request.endpoint_url,
            allow_insecure_endpoint=request.allow_lab_http_endpoint,
        )
    )
    payload = job_status_payload(status)
    payload["embedding_family"] = request.embedding_family.lower()
    return payload


def read_job_status_handoff(
    request: JobStatusRequest,
    *,
    client: DatasetSearchCommandClient,
) -> dict[str, Any]:
    """Read or wait for one Dataset Search bulk job."""
    status = (
        client.wait_for_job(
            request.job_id,
            timeout_seconds=request.timeout_seconds,
            poll_interval_seconds=request.poll_interval_seconds,
        )
        if request.wait
        else client.get_job_status(request.job_id)
    )
    return job_status_payload(status)


def list_jobs_handoff(client: DatasetSearchCommandClient) -> dict[str, Any]:
    """List recent Dataset Search bulk jobs."""
    return {"jobs": [job_status_payload(job) for job in client.list_jobs()]}


def _load_refinement_spec(spec_file: str) -> Mapping[str, Any]:
    raw = json.loads(Path(spec_file).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("--spec-file must contain a JSON object")

    grounding_queries = raw.get("grounding_queries", [])
    if not isinstance(grounding_queries, list):
        raise ValueError("grounding_queries must be a JSON array")
    if any(not isinstance(item, dict) for item in grounding_queries):
        raise ValueError("grounding_queries entries must be JSON objects")

    labels = raw.get("labels", [])
    if not isinstance(labels, list):
        raise ValueError("labels must be a JSON array")
    for label in labels:
        if not isinstance(label, dict):
            raise ValueError("labels entries must be JSON objects")
        if label.get("collection_name") is None:
            raise ValueError("labels entries must include collection_name")
        if not isinstance(label.get("labelled_documents"), dict):
            raise ValueError("labels entries must include labelled_documents object")

    return raw


def train_refinement_handoff(
    *,
    spec_file: str,
    client: DatasetSearchCommandClient,
) -> dict[str, Any]:
    """Train search refinement from a JSON spec file."""
    raw = _load_refinement_spec(spec_file)
    grounding = [
        SearchQuery(
            text=item.get("text"),
            video=item.get("video"),
            embedding=item.get("embedding"),
        )
        for item in raw.get("grounding_queries", [])
    ]
    labels = [
        LabelledDocuments(
            collection_name=label["collection_name"],
            labelled_documents=label["labelled_documents"],
        )
        for label in raw.get("labels", [])
    ]
    result = client.train_search_refinement(
        SearchRefinementSpec(
            grounding_queries=grounding,
            labels=labels,
            model_type=raw.get("model_type", "linear_probe"),
            regularization_strength=float(raw.get("regularization_strength", 0.05)),
        )
    )
    return {
        "model_type": result.model_type,
        "queries": [list(query) for query in result.queries],
        "coef": result.coef,
        "intercept": result.intercept,
        "raw": dict(result.raw),
    }
