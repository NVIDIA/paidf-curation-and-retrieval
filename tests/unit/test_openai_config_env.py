# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for env-driven Cosmos Curator OpenAI config generation."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from adapters.cosmos_curator.openai_config_env import (
    OpenAIConfigEnvError,
    build_openai_section,
    main,
    prepare_config_dir,
)


class TestBuildOpenAISection:
    def test_none_when_unset(self) -> None:
        assert build_openai_section({}) is None

    def test_sqa_aliases(self) -> None:
        section = build_openai_section(
            {
                "SQA_VLM_BASE_URL": "http://host.docker.internal:8000/v1",
                "SQA_LLM_BASE_URL": "http://host.docker.internal:8002/v1",
            }
        )
        assert section is not None
        assert section["caption"]["base_url"].endswith(":8000/v1")
        assert section["enhance"]["base_url"].endswith(":8002/v1")
        assert section["filter"]["base_url"] == section["caption"]["base_url"]
        assert section["caption"]["api_key"] == "EMPTY"

    def test_explicit_overrides_alias(self) -> None:
        section = build_openai_section(
            {
                "SQA_VLM_BASE_URL": "http://alias:8000/v1",
                "COSMOS_CURATOR_OPENAI_CAPTION_BASE_URL": "http://explicit:9000/v1",
                "COSMOS_CURATOR_OPENAI_CAPTION_API_KEY": "cap-key",
                "SQA_LLM_BASE_URL": "http://host.docker.internal:8002/v1",
            }
        )
        assert section is not None
        assert section["caption"]["base_url"] == "http://explicit:9000/v1"
        assert section["caption"]["api_key"] == "cap-key"

    def test_per_stage_urls(self) -> None:
        section = build_openai_section(
            {
                "COSMOS_CURATOR_OPENAI_CAPTION_BASE_URL": "http://caption:8000/v1",
                "COSMOS_CURATOR_OPENAI_FILTER_BASE_URL": "http://filter:8001/v1",
                "COSMOS_CURATOR_OPENAI_CLASSIFIER_BASE_URL": "http://clf:8002/v1",
                "COSMOS_CURATOR_OPENAI_ENHANCE_BASE_URL": "http://enhance:8003/v1",
                "COSMOS_CURATOR_OPENAI_EMBEDDING_BASE_URL": "http://embed:8004/v1",
            }
        )
        assert section is not None
        assert section["caption"]["base_url"] == "http://caption:8000/v1"
        assert section["filter"]["base_url"] == "http://filter:8001/v1"
        assert section["classifier"]["base_url"] == "http://clf:8002/v1"
        assert section["enhance"]["base_url"] == "http://enhance:8003/v1"
        assert section["embedding"]["base_url"] == "http://embed:8004/v1"

    def test_invalid_url_raises(self) -> None:
        with pytest.raises(OpenAIConfigEnvError, match="http"):
            build_openai_section({"SQA_VLM_BASE_URL": "not-a-url"})


class TestPrepare:
    def test_none_without_env(self, tmp_path: Path) -> None:
        assert prepare_config_dir(tmp_path / "out", environ={}) is None

    def test_writes_and_merges_hf(self, tmp_path: Path) -> None:
        host = tmp_path / "host"
        host.mkdir()
        (host / "config.yaml").write_text(
            "huggingface:\n  api_key: hf_from_host\n",
            encoding="utf-8",
        )
        out = prepare_config_dir(
            tmp_path / "gen",
            merge_from=host,
            environ={
                "SQA_VLM_BASE_URL": "http://v:8000/v1",
                "SQA_LLM_BASE_URL": "http://v:8002/v1",
            },
        )
        assert out is not None
        data = yaml.safe_load((out / "cosmos_curator.yaml").read_text(encoding="utf-8"))
        assert (out / "config.yaml").is_file()
        assert data["openai"]["enhance"]["base_url"] == "http://v:8002/v1"
        assert data["huggingface"]["api_key"] == "hf_from_host"

    def test_cli_prints_mount(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SQA_VLM_BASE_URL", "http://host.docker.internal:8000/v1")
        monkeypatch.setenv("SQA_LLM_BASE_URL", "http://host.docker.internal:8002/v1")
        rc = main(["prepare", "--output-dir", str(tmp_path / "cfg"), "--print-docker-mount"])
        assert rc == 0
        assert (tmp_path / "cfg" / "cosmos_curator.yaml").is_file()
