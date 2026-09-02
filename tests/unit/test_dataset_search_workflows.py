# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dataset Search workflow tests."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from apps.workflows import (
    JobStatusRequest,
    SearchQueryInput,
    SubmitBulkInsertRequest,
    build_search_query,
    ingest_documents_handoff,
    read_job_status_handoff,
    retrieve_collections_handoff,
    search_collection_handoff,
    submit_bulk_insert_handoff,
    train_refinement_handoff,
)
from packages.domain.types import (
    BulkJobStatus,
    DocumentInfo,
    SearchHit,
    SearchRefinementResult,
)


def test_build_search_query_rejects_non_array_embedding() -> None:
    with pytest.raises(ValueError, match="JSON array"):
        build_search_query(SearchQueryInput(embedding='{"not": "array"}'))


def test_build_search_query_rejects_non_numeric_embedding_items() -> None:
    with pytest.raises(ValueError, match="JSON array of floats"):
        build_search_query(SearchQueryInput(embedding='[1.0, "bad"]'))


def test_build_search_query_accepts_json_object_filters() -> None:
    query = build_search_query(SearchQueryInput(filters_json='{"camera_id": "cam-1"}'))

    assert query.filters == {"camera_id": "cam-1"}


def test_build_search_query_accepts_milvus_expr_filters() -> None:
    query = build_search_query(SearchQueryInput(filters_json='camera_id == "cam-1"'))

    assert query.filters == 'camera_id == "cam-1"'


def test_build_search_query_rejects_non_object_json_filters() -> None:
    with pytest.raises(ValueError, match="JSON object or Milvus expr string"):
        build_search_query(SearchQueryInput(filters_json='["camera_id"]'))


def test_search_collection_handoff_maps_hits_to_cli_payload() -> None:
    client = MagicMock()
    client.search.return_value = [
        SearchHit(
            record_id="clip-1",
            score=0.9,
            metadata={"camera": "front"},
            collection_id="collection-1",
            asset_url="s3://bucket/clip-1.mp4",
        )
    ]

    payload = search_collection_handoff(
        collection="collection-1",
        query=SearchQueryInput(text="pedestrian", top_k=5),
        client=client,
    )

    collection_arg, query_arg = client.search.call_args.args
    assert collection_arg.collection_id == "collection-1"
    assert query_arg.text == "pedestrian"
    assert query_arg.top_k == 5
    assert payload == [
        {
            "id": "clip-1",
            "score": 0.9,
            "meta": {"camera": "front"},
            "asset_url": "s3://bucket/clip-1.mp4",
            "collection_id": "collection-1",
        }
    ]


def test_retrieve_collections_handoff_maps_request_and_hits() -> None:
    client = MagicMock()
    client.multi_collection_search.return_value = [
        SearchHit(record_id="clip-2", score=0.7, metadata={}, collection_id="b")
    ]

    payload = retrieve_collections_handoff(
        collections=("a", "b"),
        query=SearchQueryInput(text="forklift", top_k=3),
        rerank=False,
        generate_asset_url=True,
        client=client,
    )

    request = client.multi_collection_search.call_args.args[0]
    assert request.collection_ids == ["a", "b"]
    assert request.query.text == "forklift"
    assert request.query.top_k == 3
    assert request.rerank is False
    assert request.generate_asset_url is True
    assert payload[0]["id"] == "clip-2"
    assert payload[0]["collection_id"] == "b"


def test_ingest_documents_handoff_maps_document_specs() -> None:
    client = MagicMock()
    client.ingest_documents.return_value = [
        DocumentInfo(document_id="doc-1", indexed_at="2026-08-05T00:00:00Z")
    ]

    payload = ingest_documents_handoff(
        collection="collection-1",
        document_jsons=(
            json.dumps(
                {
                    "mime_type": "text/plain",
                    "content": "delivery cart",
                    "metadata": {"site": "warehouse"},
                    "id": "source-1",
                }
            ),
        ),
        client=client,
    )

    collection_arg, docs = client.ingest_documents.call_args.args
    assert collection_arg.collection_id == "collection-1"
    assert docs[0].mime_type == "text/plain"
    assert docs[0].content == "delivery cart"
    assert docs[0].metadata == {"site": "warehouse"}
    assert docs[0].document_id == "source-1"
    assert payload == [{"id": "doc-1", "indexed_at": "2026-08-05T00:00:00Z"}]


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ('["not-object"]', "JSON object"),
        ('{"content": "missing mime"}', "mime_type"),
        ('{"mime_type": "text/plain", "metadata": []}', "metadata"),
    ],
)
def test_ingest_documents_handoff_rejects_invalid_specs(raw: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        ingest_documents_handoff(
            collection="collection-1",
            document_jsons=(raw,),
            client=MagicMock(),
        )


def test_submit_bulk_insert_handoff_passes_credentials_without_cli_options() -> None:
    client = MagicMock()
    client.bulk_insert.return_value = BulkJobStatus(job_id="job-1", status="accepted")

    payload = submit_bulk_insert_handoff(
        SubmitBulkInsertRequest(
            collection="demo",
            embedding_family="CE1",
            parquet_paths=("s3://bucket/data.parquet",),
            access_key="access",
            secret_key="secret",
            endpoint_url="https://s3.example",
            allow_lab_http_endpoint=True,
        ),
        client=client,
    )

    request = client.bulk_insert.call_args.args[0]
    assert request.collection_name == "demo"
    assert request.embedding_family == "ce1"
    assert request.parquet_paths == ["s3://bucket/data.parquet"]
    assert request.access_key == "access"
    assert request.secret_key == "secret"
    assert request.endpoint_url == "https://s3.example"
    assert request.allow_insecure_endpoint is True
    assert payload["job_id"] == "job-1"
    assert payload["embedding_family"] == "ce1"


def test_read_job_status_handoff_uses_wait_options() -> None:
    client = MagicMock()
    client.wait_for_job.return_value = BulkJobStatus(
        job_id="job-2",
        status="failed",
        details="bad parquet",
        progress=20,
        collection_name="demo",
    )

    payload = read_job_status_handoff(
        JobStatusRequest(
            job_id="job-2",
            wait=True,
            timeout_seconds=12.0,
            poll_interval_seconds=0.5,
        ),
        client=client,
    )

    client.wait_for_job.assert_called_once_with(
        "job-2",
        timeout_seconds=12.0,
        poll_interval_seconds=0.5,
    )
    assert payload == {
        "job_id": "job-2",
        "status": "failed",
        "details": "bad parquet",
        "progress": 20,
        "collection_name": "demo",
    }


def test_train_refinement_handoff_maps_spec_file(tmp_path: Path) -> None:
    spec_file = tmp_path / "refinement.json"
    spec_file.write_text(
        json.dumps(
            {
                "grounding_queries": [{"text": "person loading"}, {"video": "s3://v.mp4"}],
                "labels": [
                    {
                        "collection_name": "demo",
                        "labelled_documents": {"doc-1": True, "doc-2": False},
                    }
                ],
                "model_type": "linear_probe",
                "regularization_strength": 0.2,
            }
        ),
        encoding="utf-8",
    )
    client = MagicMock()
    client.train_search_refinement.return_value = SearchRefinementResult(
        model_type="linear_probe",
        queries=([0.1, 0.2],),
        coef=([1.0, 2.0],),
        intercept=[0.5],
        raw={"ok": True},
    )

    payload = train_refinement_handoff(spec_file=str(spec_file), client=client)

    spec = client.train_search_refinement.call_args.args[0]
    assert [query.text for query in spec.grounding_queries] == ["person loading", None]
    assert [query.video for query in spec.grounding_queries] == [None, "s3://v.mp4"]
    assert spec.labels[0].collection_name == "demo"
    assert spec.labels[0].labelled_documents == {"doc-1": True, "doc-2": False}
    assert spec.regularization_strength == 0.2
    assert payload["queries"] == [[0.1, 0.2]]
    assert payload["raw"] == {"ok": True}


@pytest.mark.parametrize(
    ("spec", "message"),
    [
        ([], "JSON object"),
        ({"grounding_queries": {}}, "grounding_queries must be a JSON array"),
        ({"grounding_queries": ["text"]}, "grounding_queries entries"),
        ({"labels": {}}, "labels must be a JSON array"),
        ({"labels": ["label"]}, "labels entries"),
        ({"labels": [{"labelled_documents": {"doc": True}}]}, "collection_name"),
        ({"labels": [{"collection_name": "demo", "labelled_documents": []}]}, "labelled_documents"),
    ],
)
def test_train_refinement_handoff_rejects_malformed_specs(
    tmp_path: Path,
    spec: object,
    message: str,
) -> None:
    spec_file = tmp_path / "refinement.json"
    spec_file.write_text(json.dumps(spec), encoding="utf-8")
    client = MagicMock()

    with pytest.raises(ValueError, match=message):
        train_refinement_handoff(spec_file=str(spec_file), client=client)

    client.train_search_refinement.assert_not_called()
