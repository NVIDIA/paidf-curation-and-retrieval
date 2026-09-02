# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Harbor-format checks for the curation skill eval dataset."""

from __future__ import annotations

import json
from pathlib import Path

EVALS = (
    Path(__file__).resolve().parents[2]
    / "skills"
    / "curation-and-retrieval"
    / "evals"
    / "evals.json"
)
REQUIRED_KEYS = (
    "id",
    "question",
    "expected_skill",
    "ground_truth",
    "expected_behavior",
)


def test_evals_json_is_harbor_task_array() -> None:
    """External-profile Tier 3 requires a JSON array of Harbor tasks."""
    payload = json.loads(EVALS.read_text(encoding="utf-8"))
    assert isinstance(payload, list)
    assert payload, "evals.json must contain at least one task"
    ids: list[str] = []
    for task in payload:
        assert isinstance(task, dict)
        missing = [key for key in REQUIRED_KEYS if key not in task]
        assert not missing, f"task {task.get('id')!r} missing {missing}"
        assert task["expected_skill"] == "curation-and-retrieval"
        assert isinstance(task["question"], str) and task["question"].strip()
        assert isinstance(task["ground_truth"], str) and task["ground_truth"].strip()
        assert isinstance(task["expected_behavior"], list)
        assert task["expected_behavior"]
        ids.append(str(task["id"]))
    assert len(ids) == len(set(ids))
    assert "skill_name" not in EVALS.read_text(encoding="utf-8")
