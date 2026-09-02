# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Agent harness entry files must route every tool to the same instructions."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
AGENTS_MD = REPO_ROOT / "AGENTS.md"
CLAUDE_MD = REPO_ROOT / "CLAUDE.md"


def test_agents_md_points_at_claude_md_and_codex_skills() -> None:
    """Codex and generic harnesses start at AGENTS.md; do not fork the rules."""
    text = AGENTS_MD.read_text(encoding="utf-8")
    assert CLAUDE_MD.is_file()
    assert "CLAUDE.md" in text
    assert ".codex/skills" in text
    assert "paidf-curation-and-retrieval" in text


def test_codex_skill_path_is_a_symlink_to_the_shared_pack() -> None:
    """`.codex/skills` must mirror `skills/`, not a second copy of the pack."""
    codex_skills = REPO_ROOT / ".codex" / "skills"
    assert codex_skills.is_symlink()
    assert (REPO_ROOT / "skills").is_dir()
    assert (codex_skills / "paidf-curation-and-retrieval" / "SKILL.md").is_file()
