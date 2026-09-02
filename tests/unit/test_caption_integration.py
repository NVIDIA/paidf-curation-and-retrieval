# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the CDS EA external-caption handoff."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from adapters.dataset_search.caption_adapter import CaptionAdapter, CaptionCapabilityError
from adapters.schema.caption_parquet import (
    CaptionParquetError,
    build_caption_parquet,
    load_indexed_clip_ids,
    validate_caption_frame,
    validate_caption_parquet,
)


def _response(payload: object, status_code: int = 200) -> MagicMock:
    response = MagicMock()
    response.status_code = status_code
    response.json.return_value = payload
    return response


def test_build_and_validate_caption_parquet(tmp_path: Path) -> None:
    output = build_caption_parquet(
        [
            {
                "clip_id": "clip-1",
                "summary": "A pedestrian crosses.",
                "start_time": 0.0,
                "end_time": 2.5,
                "model_name": "reason",
                "data_source": "traffic",
            }
        ],
        tmp_path / "captions.parquet",
        indexed_clip_ids={"clip-1"},
    )

    summary = validate_caption_parquet(output, indexed_clip_ids={"clip-1"})
    frame = pd.read_parquet(output)

    assert summary["rows"] == 1
    assert summary["identity_aligned"] is True
    assert list(frame.columns) == [
        "clip_id",
        "summary",
        "start_time",
        "end_time",
        "model_name",
        "data_source",
    ]


@pytest.mark.parametrize(
    ("frame", "message"),
    [
        (pd.DataFrame([{"clip_id": "a"}]), "required columns"),
        (pd.DataFrame([{"clip_id": "", "summary": "caption"}]), "non-empty"),
        (
            pd.DataFrame(
                [
                    {"clip_id": "a", "summary": "one"},
                    {"clip_id": "a", "summary": "two"},
                ]
            ),
            "unique",
        ),
        (pd.DataFrame([{"clip_id": "a", "summary": " "}]), "summary"),
        (
            pd.DataFrame([{"clip_id": "a", "summary": "one", "start_time": 2, "end_time": 1}]),
            "end_time",
        ),
    ],
)
def test_caption_validation_boundaries(frame: pd.DataFrame, message: str) -> None:
    with pytest.raises(CaptionParquetError, match=message):
        validate_caption_frame(frame)


def test_caption_identity_alignment_is_exact() -> None:
    frame = pd.DataFrame([{"clip_id": "clip-1", "summary": "caption"}])
    with pytest.raises(CaptionParquetError, match="exactly align"):
        validate_caption_frame(frame, indexed_clip_ids={"clip-1", "clip-2"})
    with pytest.raises(CaptionParquetError, match="exactly align"):
        validate_caption_frame(frame, indexed_clip_ids={"different"})


def test_load_indexed_clip_ids_from_manifest_and_lines(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"clip_ids": ["a", "b"]}), encoding="utf-8")
    lines = tmp_path / "ids.txt"
    lines.write_text("a\nb\n", encoding="utf-8")

    assert load_indexed_clip_ids(manifest) == {"a", "b"}
    assert load_indexed_clip_ids(lines) == {"a", "b"}


def test_caption_upload_calls_actual_ea_route(tmp_path: Path) -> None:
    parquet = tmp_path / "captions.parquet"
    parquet.write_bytes(b"parquet")
    session = MagicMock()
    session.request.return_value = _response({"status": "ok", "rows_inserted": 1})

    result = CaptionAdapter("https://cds.example", session=session).upload_parquet(
        parquet,
        model_name="reason",
        data_source="traffic",
    )

    assert result["rows_inserted"] == 1
    call = session.request.call_args
    assert call.args[:2] == ("POST", "https://cds.example/v1/captions/upload")
    assert call.kwargs["params"] == {"model_name": "reason", "data_source": "traffic"}
    assert call.kwargs["files"]["file"][0] == "captions.parquet"


def test_caption_stats_proves_endpoint_and_metadata_capability() -> None:
    session = MagicMock()
    session.request.return_value = _response({"captions": 3, "models": 1})

    result = CaptionAdapter("https://cds.example", session=session).stats()

    assert result == {"captions": 3, "models": 1}
    session.request.assert_called_once_with(
        "GET",
        "https://cds.example/v1/captions/stats",
        timeout=30,
    )


def test_caption_stats_rejects_non_object_json() -> None:
    session = MagicMock()
    session.request.return_value = _response(["unexpected"])

    with pytest.raises(CaptionCapabilityError, match="invalid JSON"):
        CaptionAdapter("https://cds.example", session=session).stats()


def test_caption_bulk_insert_calls_actual_ea_route_without_retry() -> None:
    session = MagicMock()
    session.request.return_value = _response(
        {"status": "success", "total_rows_inserted": 2, "files_processed": 1}
    )

    result = CaptionAdapter("https://cds.example/v1", session=session).bulk_insert(
        ["s3://bucket/captions.parquet"],
        access_key="access",
        secret_key="secret",
        endpoint_url="https://objects.example",
    )

    assert result["files_processed"] == 1
    assert session.request.call_count == 1
    call = session.request.call_args
    assert call.args[:2] == ("POST", "https://cds.example/v1/captions/bulk-insert")
    assert call.kwargs["json"]["parquet_paths"] == ["s3://bucket/captions.parquet"]


@pytest.mark.parametrize("status_code", [404, 503])
def test_caption_adapter_maps_unavailable_capability(
    tmp_path: Path,
    status_code: int,
) -> None:
    parquet = tmp_path / "captions.parquet"
    parquet.write_bytes(b"parquet")
    session = MagicMock()
    session.request.return_value = _response({"detail": "unavailable"}, status_code)

    with pytest.raises(CaptionCapabilityError, match="unavailable"):
        CaptionAdapter("https://cds.example", session=session).upload_parquet(parquet)


def test_caption_bulk_insert_validates_s3_and_https() -> None:
    adapter = CaptionAdapter("https://cds.example", session=MagicMock())
    with pytest.raises(ValueError, match="s3://"):
        adapter.bulk_insert(["/tmp/captions.parquet"])
    with pytest.raises(ValueError, match="lab-only"):
        adapter.bulk_insert(
            ["s3://bucket/captions.parquet"],
            endpoint_url="http://minio:9000",
        )


def test_caption_search_calls_verified_ea_contract() -> None:
    session = MagicMock()
    session.request.return_value = _response({"clip_ids": ["clip-1", "clip-2"], "count": 2})

    result = CaptionAdapter("https://cds.example", session=session).search(
        "pedestrian crossing",
        limit=25,
        data_sources=["traffic"],
    )

    assert result == {"clip_ids": ["clip-1", "clip-2"], "count": 2}
    session.request.assert_called_once_with(
        "POST",
        "https://cds.example/v1/captions/search",
        json={
            "query": "pedestrian crossing",
            "limit": 25,
            "data_sources": ["traffic"],
        },
        timeout=60,
    )


@pytest.mark.parametrize(
    ("query", "limit", "message"),
    [
        ("", 1, "query"),
        ("   ", 1, "query"),
        ("valid", 0, "limit"),
        ("valid", 50001, "limit"),
    ],
)
def test_caption_search_rejects_invalid_input_without_http(
    query: str,
    limit: int,
    message: str,
) -> None:
    session = MagicMock()

    with pytest.raises(ValueError, match=message):
        CaptionAdapter("https://cds.example", session=session).search(query, limit=limit)

    session.request.assert_not_called()


@pytest.mark.parametrize("data_sources", [[""], ["traffic", "   "], "traffic"])
def test_caption_search_rejects_invalid_sources_without_http(
    data_sources: object,
) -> None:
    session = MagicMock()

    with pytest.raises(ValueError, match="data_sources"):
        CaptionAdapter("https://cds.example", session=session).search(
            "pedestrian",
            data_sources=data_sources,  # type: ignore[arg-type]
        )

    session.request.assert_not_called()


@pytest.mark.parametrize("status_code", [404, 503])
def test_caption_search_maps_unavailable_capability(status_code: int) -> None:
    session = MagicMock()
    session.request.return_value = _response({"detail": "unavailable"}, status_code)

    with pytest.raises(CaptionCapabilityError, match="caption search is unavailable"):
        CaptionAdapter("https://cds.example", session=session).search("pedestrian")


def test_caption_search_does_not_log_query(caplog: pytest.LogCaptureFixture) -> None:
    sensitive_query = "customer-secret-scene"
    session = MagicMock()
    session.request.return_value = _response({"clip_ids": [], "count": 0})

    CaptionAdapter("https://cds.example", session=session).search(sensitive_query)

    assert sensitive_query not in caplog.text
