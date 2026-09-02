# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for selected-flow CDS EA external-caption readiness."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import requests
from click.testing import CliRunner

from adapters.dataset_search.caption_adapter import CaptionCapabilityError
from adapters.dataset_search.readiness import external_caption_readiness
from apps.cli.__main__ import main
from packages.domain.types import PipelineInfo


def _clients() -> tuple[MagicMock, MagicMock]:
    cds = MagicMock()
    cds.health.return_value = "ok"
    cds.list_pipelines.return_value = [
        PipelineInfo(
            pipeline_id="cosmos_video_search_milvus",
            name="cosmos_video_search_milvus",
        )
    ]
    captions = MagicMock()
    captions.stats.return_value = {"captions": 4, "models": 1}
    return cds, captions


def test_external_caption_readiness_reports_selected_dependencies() -> None:
    cds, captions = _clients()

    result = external_caption_readiness(
        cds,
        captions,
        cds_profile="internal",
        required_pipeline="cosmos_video_search_milvus",
    )

    assert result["ready"] is True
    assert result["checks"]["cds_health"]["ready"] is True
    assert result["checks"]["pipeline"]["ready"] is True
    assert result["checks"]["caption_endpoint"]["ready"] is True
    assert result["checks"]["metadata_store"]["ready"] is True
    assert "cradio" not in json.dumps(result).lower()
    assert "multicam" not in json.dumps(result).lower()


def test_external_caption_readiness_rejects_public_profile_without_calls() -> None:
    cds, captions = _clients()

    result = external_caption_readiness(
        cds,
        captions,
        cds_profile="public",
        required_pipeline="cosmos_video_search_milvus",
    )

    assert result["ready"] is False
    assert "CDS_PROFILE=internal" in result["error"]
    cds.health.assert_not_called()
    captions.stats.assert_not_called()


def test_external_caption_readiness_reports_missing_pipeline() -> None:
    cds, captions = _clients()

    result = external_caption_readiness(
        cds,
        captions,
        cds_profile="internal",
        required_pipeline="missing",
    )

    assert result["ready"] is False
    assert "pipeline is unavailable" in result["error"]
    captions.stats.assert_not_called()


def test_external_caption_readiness_reports_unavailable_metadata_store() -> None:
    cds, captions = _clients()
    captions.stats.side_effect = CaptionCapabilityError("caption store is not configured")

    result = external_caption_readiness(
        cds,
        captions,
        cds_profile="internal",
        required_pipeline="cosmos_video_search_milvus",
    )

    assert result["ready"] is False
    assert result["checks"]["caption_endpoint"]["ready"] is False
    assert "not configured" in result["error"]


def test_external_caption_readiness_reports_cds_connection_failure() -> None:
    cds, captions = _clients()
    cds.health.side_effect = requests.ConnectionError("connection refused")

    result = external_caption_readiness(
        cds,
        captions,
        cds_profile="internal",
        required_pipeline="cosmos_video_search_milvus",
    )

    assert result["ready"] is False
    assert result["error"].startswith("CDS is unavailable")
    captions.stats.assert_not_called()


def test_caption_readiness_cli_emits_json_for_ready_and_blocked(monkeypatch) -> None:
    ready = {
        "flow": "external_caption",
        "profile": "internal",
        "ready": True,
        "checks": {},
    }
    monkeypatch.setattr(
        "apps.cli.integration_cmds.external_caption_readiness",
        lambda *args, **kwargs: ready,
    )
    runner = CliRunner()

    success = runner.invoke(
        main,
        [
            "integration",
            "captions",
            "readiness",
            "--cds-url",
            "https://cds.example",
            "--cds-profile",
            "internal",
            "--pipeline",
            "cosmos_video_search_milvus",
        ],
    )
    assert success.exit_code == 0
    assert json.loads(success.output)["ready"] is True

    ready["ready"] = False
    ready["error"] = "metadata unavailable"
    blocked = runner.invoke(
        main,
        [
            "integration",
            "captions",
            "readiness",
            "--cds-url",
            "https://cds.example",
            "--cds-profile",
            "internal",
            "--pipeline",
            "cosmos_video_search_milvus",
        ],
    )
    assert blocked.exit_code == 2
    assert json.loads(blocked.output)["error"] == "metadata unavailable"
