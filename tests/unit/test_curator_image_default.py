# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Curator image default resolution."""

from __future__ import annotations

from adapters.docker_runtime import default_curator_image


class TestDefaultCuratorImage:
    def test_composes_name_and_tag(self, monkeypatch) -> None:
        monkeypatch.delenv("COSMOS_CURATOR_DOCKER_IMAGE", raising=False)
        monkeypatch.setenv("COSMOS_CURATOR_IMAGE", "cosmos-curator")
        monkeypatch.setenv("COSMOS_CURATOR_TAG", "cli-tag")
        assert default_curator_image() == "cosmos-curator:cli-tag"

    def test_explicit_docker_image_wins(self, monkeypatch) -> None:
        monkeypatch.setenv(
            "COSMOS_CURATOR_DOCKER_IMAGE", "nvcr.io/nvidia/cosmos/cosmos-curator:explicit"
        )
        assert default_curator_image() == "nvcr.io/nvidia/cosmos/cosmos-curator:explicit"

    def test_fallback_defaults(self, monkeypatch) -> None:
        monkeypatch.delenv("COSMOS_CURATOR_DOCKER_IMAGE", raising=False)
        monkeypatch.delenv("COSMOS_CURATOR_TAG", raising=False)
        monkeypatch.delenv("COSMOS_CURATOR_IMAGE", raising=False)
        assert default_curator_image() == "cosmos-curator:2.3.0"

    def test_image_with_tag_passthrough(self, monkeypatch) -> None:
        monkeypatch.delenv("COSMOS_CURATOR_DOCKER_IMAGE", raising=False)
        monkeypatch.setenv("COSMOS_CURATOR_IMAGE", "cosmos-curator:custom")
        assert default_curator_image() == "cosmos-curator:custom"

    def test_registry_port_without_tag_appends_tag(self, monkeypatch) -> None:
        monkeypatch.delenv("COSMOS_CURATOR_DOCKER_IMAGE", raising=False)
        monkeypatch.setenv("COSMOS_CURATOR_IMAGE", "registry:5000/cosmos-curator")
        monkeypatch.setenv("COSMOS_CURATOR_TAG", "2.3.0")
        assert default_curator_image() == "registry:5000/cosmos-curator:2.3.0"

    def test_registry_port_with_explicit_tag_passthrough(self, monkeypatch) -> None:
        monkeypatch.delenv("COSMOS_CURATOR_DOCKER_IMAGE", raising=False)
        monkeypatch.setenv("COSMOS_CURATOR_IMAGE", "registry:5000/cosmos-curator:custom")
        assert default_curator_image() == "registry:5000/cosmos-curator:custom"

    def test_registry_path_without_tag_appends_tag(self, monkeypatch) -> None:
        monkeypatch.delenv("COSMOS_CURATOR_DOCKER_IMAGE", raising=False)
        monkeypatch.setenv("COSMOS_CURATOR_IMAGE", "nvcr.io/nvidia/cosmos/cosmos-curator")
        monkeypatch.setenv("COSMOS_CURATOR_TAG", "2.3.0")
        assert default_curator_image() == "nvcr.io/nvidia/cosmos/cosmos-curator:2.3.0"
