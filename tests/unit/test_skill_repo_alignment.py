# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Skill pack stays aligned with the current Make-only, cookbook-first repo."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SKILL_ROOT = REPO_ROOT / "skills" / "paidf-curation-and-retrieval"
SKILL_MD = SKILL_ROOT / "SKILL.md"
SPLIT_MINIMAL = REPO_ROOT / "cookbook" / "traffic-video-analytics" / "split-minimal.yaml"

# Retired local harness language. Negations like "no in-repo E2E" are allowed.
RETIRED_TOKENS = (
    "E2E-test/",
    "E2E L1",
    "L0–L2",
    "L0-L2",
    "L3a+",
    "test L10",
    "E2E harness follow",
    "Verified by E2E",
    "E2E verification",
)


def _skill_markdown_files() -> list[Path]:
    return sorted(SKILL_ROOT.rglob("*.md"))


def test_skill_yaml_name_matches_the_product_name() -> None:
    """Harbor expected_skill and SKILL.md name must match the public product."""
    text = SKILL_MD.read_text(encoding="utf-8")
    assert text.startswith("---\nname: paidf-curation-and-retrieval\n")
    assert SKILL_ROOT.name == "paidf-curation-and-retrieval"
    """The gitignored E2E/L1 harness is gone; skills must not send operators there."""
    offenders: list[str] = []
    for path in _skill_markdown_files():
        text = path.read_text(encoding="utf-8")
        for token in RETIRED_TOKENS:
            if token in text:
                rel = path.relative_to(REPO_ROOT)
                offenders.append(f"{rel}: {token}")
    assert not offenders, "retired E2E language still in skill pack:\n" + "\n".join(offenders)


def test_skill_index_points_at_current_operator_surface() -> None:
    """SKILL.md first-run and validation match README / Make / unit tests."""
    text = SKILL_MD.read_text(encoding="utf-8")
    assert "split-minimal" in text
    assert "cookbook" in text
    assert "uv run pytest" in text
    assert "make check-image" in text
    assert "in-repo E2E or L1 harness" in text
    assert "cookbook/traffic-video-analytics/" not in text
    assert "configs/image.yaml" not in text
    assert "tests/unit" not in text
    assert ".env.example" not in text
    assert "docs/user-guide" not in text
    assert SPLIT_MINIMAL.is_file()
    recipe = SPLIT_MINIMAL.read_text(encoding="utf-8")
    assert 'embedding_algorithm: "cosmos-embed1-224p"' in recipe


def test_ffmpeg_sidecar_acceptance_uses_make_preflight() -> None:
    """Sidecar proof is Make + container ffmpeg, not a missing E2E level."""
    text = (SKILL_ROOT / "references" / "ffmpeg-sidecar.md").read_text(encoding="utf-8")
    assert "make check-setup" in text
    assert "make check-image" in text
    assert "make shell" in text
    assert "E2E-test/" not in text
    assert "Acceptance check (E2E L1 pattern)" not in text
