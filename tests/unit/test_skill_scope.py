# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Skill index stays Curator-scoped; TAO CLI is not the skill surface."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SKILL = REPO_ROOT / "skills" / "curation-and-retrieval" / "SKILL.md"
DATA_MINING = REPO_ROOT / "skills" / "curation-and-retrieval" / "references" / "data-mining.md"
TAO_CLI_COMMANDS = (
    "tmm nearest_neighbors",
    "tmm unique_neighbor_matching",
    "embedding image_embeddings",
    "embedding text_embeddings",
)


def test_skill_index_does_not_name_tao_cli_commands() -> None:
    """SKILL.md is a Curator index; it must not list TAO CLI subcommands."""
    text = SKILL.read_text(encoding="utf-8")
    for command in TAO_CLI_COMMANDS:
        assert command not in text, f"SKILL.md still names {command}"
    assert "references/data-mining.md" in text
    assert "references/curation-retrieval-workflow.md" in text


def test_context_understanding_example_orders_critical_first() -> None:
    """Taxonomy example lists critical events before the baseline catch-all."""
    text = (
        REPO_ROOT / "skills" / "curation-and-retrieval" / "references" / "context-understanding.md"
    ).read_text(encoding="utf-8")
    start = text.index("Example -- traffic video analytics:")
    yaml_start = text.index("```yaml", start)
    yaml_end = text.index("```", yaml_start + 7)
    example = text[yaml_start:yaml_end]
    events = [
        line.split("- ", 1)[1].strip()
        for line in example.splitlines()
        if line.strip().startswith("- ")
    ]
    assert events[0] == "vehicle_to_vehicle_collision"
    assert events[-1] == "normal_traffic_flow"
    assert "signal_violation_or_wrong_way" in events
    assert events.index("vehicle_to_vehicle_collision") < events.index(
        "signal_violation_or_wrong_way"
    )


def test_data_mining_opening_is_handoff_not_tao_cli() -> None:
    """Opening of data-mining.md describes Curator handoff, not TAO CLI."""
    lines = DATA_MINING.read_text(encoding="utf-8").splitlines()
    opening = "\n".join(lines[:20])
    assert "Handoff" in opening or "handoff" in opening
    assert "make help" in opening
    for command in TAO_CLI_COMMANDS:
        assert f"- `{command}`" not in opening
