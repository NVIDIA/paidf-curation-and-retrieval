# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for SeedVR2 checkpoint preflight helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from adapters.cosmos_curator.seedvr_ckpts import (
    EMA_3B,
    EMA_VAE,
    NEG_EMB,
    POS_EMB,
    SEEDVR_CONTAINER_CKPTS,
    SEEDVR_CONTAINER_HF_3B,
    SeedVRCkptError,
    config_requires_seedvr,
    missing_seedvr_ckpts,
    normalize_seedvr_variant,
    preflight_seedvr_for_config,
    required_seedvr_filenames,
    seedvr_ckpts_dir,
    seedvr_container_hf_dir,
    seedvr_docker_mount_args,
)


class TestSeedVRVariant:
    def test_normalize_3b_and_7b(self) -> None:
        assert normalize_seedvr_variant("seedvr2_3b") == "3b"
        assert normalize_seedvr_variant("seedvr2_7b") == "7b"
        assert normalize_seedvr_variant("SEEDVR2_7B_sharp") == "7b"

    def test_normalize_invalid_raises(self) -> None:
        with pytest.raises(SeedVRCkptError, match="Unsupported"):
            normalize_seedvr_variant("not-a-variant")

    def test_required_filenames_include_text_embeds(self) -> None:
        names = required_seedvr_filenames("seedvr2_3b")
        assert names == (EMA_VAE, EMA_3B, POS_EMB, NEG_EMB)
        assert POS_EMB in required_seedvr_filenames("seedvr2_7b")
        assert NEG_EMB in required_seedvr_filenames("seedvr2_7b")

    def test_container_hf_dir(self) -> None:
        assert seedvr_container_hf_dir("seedvr2_3b") == SEEDVR_CONTAINER_HF_3B
        assert "SeedVR2-7B" in seedvr_container_hf_dir("seedvr2_7b")


class TestSeedVRLayout:
    def test_ckpts_dir_under_models(self, tmp_path: Path) -> None:
        assert seedvr_ckpts_dir(tmp_path) == (tmp_path / "seedvr2").resolve()

    def test_missing_when_empty(self, tmp_path: Path) -> None:
        missing = missing_seedvr_ckpts(tmp_path, "seedvr2_3b")
        assert EMA_VAE in missing
        assert EMA_3B in missing
        assert POS_EMB in missing
        assert NEG_EMB in missing

    def _write_required(self, tmp_path: Path) -> Path:
        root = seedvr_ckpts_dir(tmp_path)
        root.mkdir(parents=True)
        for name in (EMA_VAE, EMA_3B, POS_EMB, NEG_EMB):
            (root / name).write_bytes(b"x")
        return root

    def test_missing_empty_when_present(self, tmp_path: Path) -> None:
        self._write_required(tmp_path)
        assert missing_seedvr_ckpts(tmp_path, "seedvr2_3b") == []

    def test_docker_mount_args_include_text_embeds(self, tmp_path: Path) -> None:
        self._write_required(tmp_path)
        args = seedvr_docker_mount_args(tmp_path, variant="seedvr2_3b")
        joined = " ".join(args)
        assert args[0] == "-v"
        assert str((tmp_path / "seedvr2").resolve()) in joined
        assert SEEDVR_CONTAINER_CKPTS in joined
        assert f"{SEEDVR_CONTAINER_HF_3B}/{POS_EMB}" in joined
        assert f"{SEEDVR_CONTAINER_HF_3B}/{NEG_EMB}" in joined
        assert args.count("-v") == 3

    def test_text_embeds_fallback_dir_when_seedvr2_not_writable(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from adapters.cosmos_curator import seedvr_ckpts as mod

        monkeypatch.setattr(mod, "_dir_is_writable", lambda _p: False)
        embeds = mod.seedvr_text_embeds_dir(tmp_path)
        assert embeds == (tmp_path / "seedvr2_embeds").resolve()
        # DiT still looked up under seedvr2/; embeds under fallback.
        ckpts = seedvr_ckpts_dir(tmp_path)
        ckpts.mkdir(parents=True)
        (ckpts / EMA_VAE).write_bytes(b"vae")
        (ckpts / EMA_3B).write_bytes(b"dit")
        assert POS_EMB in missing_seedvr_ckpts(tmp_path, "seedvr2_3b")
        embeds.mkdir(parents=True)
        (embeds / POS_EMB).write_bytes(b"pos")
        (embeds / NEG_EMB).write_bytes(b"neg")
        assert missing_seedvr_ckpts(tmp_path, "seedvr2_3b") == []
        joined = " ".join(seedvr_docker_mount_args(tmp_path, variant="seedvr2_3b"))
        assert "seedvr2_embeds" in joined
        assert POS_EMB in joined


class TestConfigRequiresSeedVR:
    def test_true_when_enabled(self, tmp_path: Path) -> None:
        cfg = tmp_path / "pipe.yaml"
        cfg.write_text("pipeline: split\nsuper_resolution: true\n", encoding="utf-8")
        assert config_requires_seedvr(cfg) is True

    def test_false_when_disabled_or_missing(self, tmp_path: Path) -> None:
        cfg = tmp_path / "pipe.yaml"
        cfg.write_text("pipeline: split\nsuper_resolution: false\n", encoding="utf-8")
        assert config_requires_seedvr(cfg) is False
        assert config_requires_seedvr(None) is False

    def test_true_with_inline_comment(self, tmp_path: Path) -> None:
        cfg = tmp_path / "pipe.yaml"
        cfg.write_text("super_resolution: true  # kitchen sink\n", encoding="utf-8")
        assert config_requires_seedvr(cfg) is True


class TestPreflight:
    def test_skip_returns_empty_when_sr_off(self, tmp_path: Path) -> None:
        cfg = tmp_path / "pipe.yaml"
        cfg.write_text("super_resolution: false\n", encoding="utf-8")
        assert preflight_seedvr_for_config(tmp_path, cfg, ensure="skip") == []

    def test_skip_mounts_when_present(self, tmp_path: Path) -> None:
        cfg = tmp_path / "pipe.yaml"
        cfg.write_text("super_resolution: true\nsr_variant: seedvr2_3b\n", encoding="utf-8")
        root = seedvr_ckpts_dir(tmp_path)
        root.mkdir(parents=True)
        for name in (EMA_VAE, EMA_3B, POS_EMB, NEG_EMB):
            (root / name).write_bytes(b"x")
        args = preflight_seedvr_for_config(tmp_path, cfg, ensure="skip")
        assert args and args[0] == "-v"
        assert POS_EMB in " ".join(args)

    def test_check_raises_when_missing(self, tmp_path: Path) -> None:
        cfg = tmp_path / "pipe.yaml"
        cfg.write_text("super_resolution: true\n", encoding="utf-8")
        with pytest.raises(SeedVRCkptError, match="Missing"):
            preflight_seedvr_for_config(tmp_path, cfg, ensure="check")
