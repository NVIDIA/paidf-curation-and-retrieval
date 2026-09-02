# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the experimental Dataset Search compose asset."""

from __future__ import annotations

from pathlib import Path

import yaml

COMPOSE = Path(__file__).resolve().parents[2] / "deploy/compose/docker-compose.yml"


def _compose() -> dict:
    data = yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


class TestDatasetSearchCompose:
    def test_compose_references_pulled_image_only(self) -> None:
        service = _compose()["services"]["dataset-search"]

        assert service["image"].startswith("${DATASET_SEARCH_IMAGE")
        assert "DATASET_SEARCH_TAG" in service["image"]
        assert "build" not in service

    def test_compose_has_ga_pipeline_init_env(self) -> None:
        service = _compose()["services"]["dataset-search"]

        assert service["environment"]["MILVUS_DB"] == "${MILVUS_DB:-default}"
        assert "host.docker.internal:host-gateway" in service["extra_hosts"]
