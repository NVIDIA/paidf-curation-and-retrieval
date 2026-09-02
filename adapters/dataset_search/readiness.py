# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Capability-based readiness checks for selected CDS EA integration flows."""

from __future__ import annotations

from typing import Any

import requests

from adapters.dataset_search.caption_adapter import CaptionAdapter, CaptionCapabilityError
from adapters.dataset_search.retrieval_adapter import DatasetSearchAdapter


def external_caption_readiness(
    cds: DatasetSearchAdapter,
    captions: CaptionAdapter,
    *,
    cds_profile: str,
    required_pipeline: str,
) -> dict[str, Any]:
    """Check only dependencies required by the EA external-caption handoff."""
    result: dict[str, Any] = {
        "flow": "external_caption",
        "profile": cds_profile,
        "ready": False,
        "checks": {
            "cds_health": {"ready": False},
            "pipeline": {"ready": False, "required": required_pipeline},
            "caption_endpoint": {"ready": False, "path": "/v1/captions/stats"},
            "metadata_store": {
                "ready": False,
                "evidence": "successful /v1/captions/stats response",
            },
        },
    }
    if cds_profile != "internal":
        result["error"] = "External caption integration requires CDS_PROFILE=internal (EA)"
        return result
    if not required_pipeline.strip():
        result["error"] = "required_pipeline must be non-empty"
        return result

    try:
        health = cds.health()
        result["checks"]["cds_health"] = {"ready": True, "status": health}
        pipelines = cds.list_pipelines()
    except requests.RequestException as exc:
        result["error"] = f"CDS is unavailable: {exc}"
        return result

    available = sorted(
        {
            value
            for pipeline in pipelines
            for value in (pipeline.pipeline_id, pipeline.name)
            if value
        }
    )
    result["checks"]["pipeline"]["available"] = available
    if required_pipeline not in available:
        result["error"] = f"Required CDS pipeline is unavailable: {required_pipeline}"
        return result
    result["checks"]["pipeline"]["ready"] = True

    try:
        stats = captions.stats()
    except (CaptionCapabilityError, requests.RequestException, ValueError) as exc:
        result["error"] = str(exc)
        return result

    result["checks"]["caption_endpoint"]["ready"] = True
    result["checks"]["metadata_store"]["ready"] = True
    result["checks"]["metadata_store"]["stats_fields"] = sorted(str(key) for key in stats)
    result["ready"] = True
    return result
