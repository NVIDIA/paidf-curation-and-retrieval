# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dataset Search (CVDS) CLI commands — full standalone retrieval surface."""

from __future__ import annotations

import json
from typing import Any

import click

from adapters.dataset_search.retrieval_adapter import (
    BulkJobPollingTimeout,
    DatasetSearchAdapter,
)
from adapters.object_store import ObjectStoreStagingError, resolve_credential_pair
from apps.composition import build_dataset_search_adapter
from apps.workflows.dataset_search import (
    JobStatusRequest,
    SearchQueryInput,
    SubmitBulkInsertRequest,
    delete_document_handoff,
    ingest_documents_handoff,
    is_failed_job_status,
    list_jobs_handoff,
    read_job_status_handoff,
    retrieve_collections_handoff,
    search_collection_handoff,
    submit_bulk_insert_handoff,
    train_refinement_handoff,
)
from packages.domain.types import (
    CollectionCreateSpec,
    CollectionPatchSpec,
)


def _client(cds_url: str | None = None) -> DatasetSearchAdapter:
    return build_dataset_search_adapter(cds_url)


def _echo_json(payload: Any) -> None:
    click.echo(json.dumps(payload, indent=2, default=str))


def _credential_values(access_key_env: str, secret_key_env: str) -> tuple[str | None, str | None]:
    """Resolve a validated credential pair without accepting values as CLI arguments."""
    try:
        return resolve_credential_pair(access_key_env, secret_key_env)
    except (ObjectStoreStagingError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc


@click.group("ds")
def dataset_search_group() -> None:
    """Dataset Search (CVDS) — collections, search, ingest, jobs, refinement."""


@dataset_search_group.command("health")
@click.option("--cds-url", default=None)
def health(cds_url: str | None) -> None:
    """GET /health."""
    _echo_json({"status": _client(cds_url).health()})


@dataset_search_group.command("pipelines")
@click.option("--cds-url", default=None)
def pipelines(cds_url: str | None) -> None:
    """List available embedding pipelines."""
    items = [
        {
            "id": p.pipeline_id,
            "name": p.name,
            "description": p.description,
        }
        for p in _client(cds_url).list_pipelines()
    ]
    _echo_json({"pipelines": items})


@dataset_search_group.command("collections")
@click.option("--cds-url", default=None)
def collections(cds_url: str | None) -> None:
    """List collections."""
    items = [
        {
            "id": c.collection_id,
            "name": c.name,
            "pipeline": c.pipeline,
            "tags": dict(c.tags),
            "total_documents_count": c.total_documents_count,
        }
        for c in _client(cds_url).list_collections()
    ]
    _echo_json({"collections": items})


@dataset_search_group.command("get-collection")
@click.argument("collection_id")
@click.option("--cds-url", default=None)
def get_collection(collection_id: str, cds_url: str | None) -> None:
    """Get one collection (with document count when available)."""
    c = _client(cds_url).get_collection(collection_id)
    _echo_json(
        {
            "id": c.collection_id,
            "name": c.name,
            "pipeline": c.pipeline,
            "tags": dict(c.tags),
            "total_documents_count": c.total_documents_count,
            "created_at": c.created_at,
        }
    )


@dataset_search_group.command("create-collection")
@click.option("--name", required=True)
@click.option("--pipeline", required=True, help="e.g. cosmos_embed1_embedding_milvus")
@click.option("--id", "collection_id", default=None, help="Optional UUID for the collection")
@click.option("--tags-json", default="{}", help="JSON object of collection tags")
@click.option("--cds-url", default=None)
def create_collection(
    name: str,
    pipeline: str,
    collection_id: str | None,
    tags_json: str,
    cds_url: str | None,
) -> None:
    """Create a collection."""
    tags = json.loads(tags_json)
    created = _client(cds_url).create_collection(
        CollectionCreateSpec(
            name=name,
            pipeline=pipeline,
            tags=tags,
            collection_id=collection_id,
        )
    )
    _echo_json(
        {
            "id": created.collection_id,
            "name": created.name,
            "pipeline": created.pipeline,
            "tags": dict(created.tags),
        }
    )


@dataset_search_group.command("update-collection")
@click.argument("collection_id")
@click.option("--name", default=None)
@click.option("--tags-json", default=None)
@click.option("--cds-url", default=None)
def update_collection(
    collection_id: str,
    name: str | None,
    tags_json: str | None,
    cds_url: str | None,
) -> None:
    """Patch collection name and/or tags."""
    tags = json.loads(tags_json) if tags_json is not None else None
    updated = _client(cds_url).update_collection(
        collection_id, CollectionPatchSpec(name=name, tags=tags)
    )
    _echo_json({"id": updated.collection_id, "name": updated.name, "tags": dict(updated.tags)})


@dataset_search_group.command("delete-collection")
@click.argument("collection_id")
@click.option("--cds-url", default=None)
@click.confirmation_option(prompt="Delete this collection?")
def delete_collection(collection_id: str, cds_url: str | None) -> None:
    """Delete a collection."""
    _client(cds_url).delete_collection(collection_id)
    _echo_json({"deleted": collection_id})


@dataset_search_group.command("flush")
@click.argument("collection_id")
@click.option("--cds-url", default=None)
def flush(collection_id: str, cds_url: str | None) -> None:
    """Force-flush a Milvus-backed collection."""
    _echo_json(dict(_client(cds_url).flush_collection(collection_id)))


@dataset_search_group.command("search")
@click.option("--collection", required=True)
@click.option("--text", default=None, help="Text query")
@click.option("--embedding", default=None, help='JSON array, e.g. "[0.1, 0.2]"')
@click.option("--image", default=None, help="Image data URI or URL (C-Radio)")
@click.option("--video", default=None, help="Video data URI or URL (Cosmos-Embed)")
@click.option("--video-frames", default=None, help="Exactly 8 frames data URI")
@click.option("--top-k", default=10, type=int)
@click.option("--filters-json", default=None, help="JSON filters object or Milvus expr string")
@click.option("--generate-asset-url/--no-asset-url", default=False)
@click.option("--cds-url", default=None)
def search_cmd(
    collection: str,
    text: str | None,
    embedding: str | None,
    image: str | None,
    video: str | None,
    video_frames: str | None,
    top_k: int,
    filters_json: str | None,
    generate_asset_url: bool,
    cds_url: str | None,
) -> None:
    """Multimodal search within one collection."""
    try:
        payload = search_collection_handoff(
            collection=collection,
            query=SearchQueryInput(
                text=text,
                embedding=embedding,
                image=image,
                video=video,
                video_frames=video_frames,
                top_k=top_k,
                filters_json=filters_json,
                generate_asset_url=generate_asset_url,
            ),
            client=_client(cds_url),
        )
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    _echo_json(payload)


@dataset_search_group.command("retrieve")
@click.option("--collection", "collections", multiple=True, required=True)
@click.option("--text", default=None)
@click.option("--embedding", default=None)
@click.option("--image", default=None)
@click.option("--video", default=None)
@click.option("--top-k", default=10, type=int)
@click.option("--filters-json", default=None)
@click.option("--rerank/--no-rerank", default=True)
@click.option("--generate-asset-url/--no-asset-url", default=False)
@click.option("--cds-url", default=None)
def retrieve_cmd(
    collections: tuple[str, ...],
    text: str | None,
    embedding: str | None,
    image: str | None,
    video: str | None,
    top_k: int,
    filters_json: str | None,
    rerank: bool,
    generate_asset_url: bool,
    cds_url: str | None,
) -> None:
    """Multi-collection retrieval (POST /retrieval)."""
    try:
        payload = retrieve_collections_handoff(
            collections=collections,
            query=SearchQueryInput(
                text=text,
                embedding=embedding,
                image=image,
                video=video,
                top_k=top_k,
                filters_json=filters_json,
                generate_asset_url=generate_asset_url,
            ),
            rerank=rerank,
            generate_asset_url=generate_asset_url,
            client=_client(cds_url),
        )
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    _echo_json(payload)


@dataset_search_group.command("ingest-documents")
@click.option("--collection", required=True)
@click.option(
    "--document-json",
    "document_jsons",
    multiple=True,
    required=True,
    help='JSON DocumentSpec, e.g. \'{"mime_type":"text/plain","content":"hi"}\'',
)
@click.option("--cds-url", default=None)
def ingest_documents(collection: str, document_jsons: tuple[str, ...], cds_url: str | None) -> None:
    """Index documents (content / url / embedding)."""
    try:
        payload = ingest_documents_handoff(
            collection=collection,
            document_jsons=document_jsons,
            client=_client(cds_url),
        )
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    _echo_json(payload)


@dataset_search_group.command("delete-document")
@click.option("--collection", required=True)
@click.option("--document-id", required=True)
@click.option("--cds-url", default=None)
def delete_document(collection: str, document_id: str, cds_url: str | None) -> None:
    """Delete one document by id."""
    _echo_json(
        delete_document_handoff(
            collection=collection,
            document_id=document_id,
            client=_client(cds_url),
        )
    )


@dataset_search_group.command("bulk-insert")
@click.option("--collection", required=True, help="Collection name or id")
@click.option(
    "--embedding-family",
    required=True,
    type=click.Choice(["ce1"], case_sensitive=False),
    help="Required artifact declaration; vectors do not identify their model family",
)
@click.option(
    "--parquet",
    "parquet_paths",
    multiple=True,
    required=True,
    help="s3:// path (repeatable)",
)
@click.option("--access-key-env", default="AWS_ACCESS_KEY_ID", show_default=True)
@click.option("--secret-key-env", default="AWS_SECRET_ACCESS_KEY", show_default=True)
@click.option("--endpoint-url", default=None)
@click.option(
    "--allow-lab-http-endpoint",
    is_flag=True,
    help="Lab only: permit an HTTP object-store endpoint",
)
@click.option("--cds-url", default=None)
def bulk_insert(
    collection: str,
    embedding_family: str,
    parquet_paths: tuple[str, ...],
    access_key_env: str,
    secret_key_env: str,
    endpoint_url: str | None,
    allow_lab_http_endpoint: bool,
    cds_url: str | None,
) -> None:
    """Submit one bulk parquet job; POST is never retried automatically."""
    access_key, secret_key = _credential_values(access_key_env, secret_key_env)
    _echo_json(
        submit_bulk_insert_handoff(
            SubmitBulkInsertRequest(
                collection=collection,
                parquet_paths=parquet_paths,
                embedding_family=embedding_family,
                access_key=access_key,
                secret_key=secret_key,
                endpoint_url=endpoint_url,
                allow_lab_http_endpoint=allow_lab_http_endpoint,
            ),
            client=_client(cds_url),
        )
    )


@dataset_search_group.command("job-status")
@click.argument("job_id")
@click.option("--wait/--no-wait", default=False, help="Poll until CDS reports a terminal state")
@click.option("--timeout-seconds", default=900.0, type=click.FloatRange(min=0))
@click.option("--poll-interval-seconds", default=5.0, type=click.FloatRange(min=0, min_open=True))
@click.option("--cds-url", default=None)
def job_status(
    job_id: str,
    wait: bool,
    timeout_seconds: float,
    poll_interval_seconds: float,
    cds_url: str | None,
) -> None:
    """Get or poll externally owned CDS bulk-job status."""
    client = _client(cds_url)
    try:
        payload = read_job_status_handoff(
            JobStatusRequest(
                job_id=job_id,
                wait=wait,
                timeout_seconds=timeout_seconds,
                poll_interval_seconds=poll_interval_seconds,
            ),
            client=client,
        )
    except BulkJobPollingTimeout as exc:
        _echo_json({"job_id": job_id, "status": "timeout", "error": str(exc)})
        raise click.exceptions.Exit(2) from exc

    _echo_json(payload)
    if wait and is_failed_job_status(str(payload["status"])):
        raise click.exceptions.Exit(2)


@dataset_search_group.command("jobs")
@click.option("--cds-url", default=None)
def jobs(cds_url: str | None) -> None:
    """List recent bulk-insert jobs."""
    _echo_json(list_jobs_handoff(_client(cds_url)))


@dataset_search_group.command("train-refinement")
@click.option(
    "--spec-file",
    required=True,
    type=click.Path(exists=True),
    help="JSON SearchRefinementSpec file",
)
@click.option("--cds-url", default=None)
def train_refinement(spec_file: str, cds_url: str | None) -> None:
    """Train search refinement from a JSON spec file."""
    try:
        payload = train_refinement_handoff(spec_file=spec_file, client=_client(cds_url))
    except (OSError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    _echo_json(payload)
