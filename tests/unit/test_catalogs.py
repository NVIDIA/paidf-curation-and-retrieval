# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contract tests for maintained visual-analytics catalogs."""

from __future__ import annotations

import re
from pathlib import Path

import yaml

CATALOG_DIR = Path(__file__).parents[2] / ".agents" / "references" / "catalogs"
REQUIRED_KEYS = {
    "domain",
    "baseline_event",
    "keywords",
    "events",
    "remap",
    "excluded",
}
EVENT_TIERS = {"critical", "serious", "moderate", "low"}
REMOVED_TERM = "surveil" + "lance"
PROHIBITED_LANGUAGE = re.compile(
    rf"\b(?:{REMOVED_TERM}|suspicious|criminal(?:ity)?|crime|theft|shoplifting|"
    r"fraud|casing|protected trait|identity recognition|emotion|personality|"
    r"intent inference|harassment|homeless|vagrant)\b",
    re.IGNORECASE,
)

EXPECTED_DOMAINS = {
    "construction.yaml": "construction site safety",
    "employee_conduct_monitoring.yaml": "employee conduct monitoring",
    "incident_video_analytics.yaml": "facility incident analytics",
    "logistics.yaml": "logistics and distribution operations",
    "parking_lot.yaml": "parking facility video analytics",
    "retail.yaml": "retail store video analytics",
    "traffic_safety.yaml": "traffic safety",
    "warehouse.yaml": "warehouse operations",
}


def _catalog_paths() -> list[Path]:
    return sorted(CATALOG_DIR.glob("*.yaml"))


def test_all_catalogs_have_stable_schema_and_resolvable_remaps() -> None:
    """Every catalog keeps the documented schema and valid alias targets."""
    paths = _catalog_paths()
    assert paths

    for path in paths:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert set(document) == REQUIRED_KEYS, path.name
        assert set(document["events"]) == EVENT_TIERS, path.name

        labels = [label for tier_labels in document["events"].values() for label in tier_labels]
        assert all(isinstance(label, str) and label.strip() for label in labels)
        assert len(labels) == len(set(labels)), path.name
        assert set(document["remap"].values()) <= set(labels), path.name


def test_catalog_filenames_avoid_prohibited_language() -> None:
    """Catalog filenames stay grounded in visual safety analytics."""
    for path in _catalog_paths():
        assert not PROHIBITED_LANGUAGE.search(path.name), path.name


def test_catalog_contents_avoid_prohibited_language() -> None:
    """Catalog YAML contents avoid prohibited investigative/surveillance language."""
    for path in _catalog_paths():
        text = path.read_text(encoding="utf-8")
        match = PROHIBITED_LANGUAGE.search(text)
        assert match is None, f"{path.name} contains prohibited term {match.group(0)!r}"


def test_retail_remaps_do_not_reuse_nearby_events_for_robbery_aliases() -> None:
    """Robbery phrases must not remap onto fight or break-in labels."""
    document = yaml.safe_load((CATALOG_DIR / "retail.yaml").read_text(encoding="utf-8"))
    remaps = document["remap"]
    assert "holdup" not in remaps
    assert "smash and grab" not in remaps
    assert remaps["fighting"] == "physical altercation involving employee or customer"
    assert remaps["break-in"] == "forced entry through entrance or window"


def test_parking_lot_fall_remap_is_not_used_for_violence_aliases() -> None:
    """Fighting and assault must not map to the accidental pedestrian-fall label."""
    document = yaml.safe_load((CATALOG_DIR / "parking_lot.yaml").read_text(encoding="utf-8"))
    remaps = document["remap"]
    fall_label = "pedestrian fall or injury"
    assert remaps.get("fighting") != fall_label
    assert remaps.get("assault") != fall_label
    assert "fighting" not in remaps
    assert "assault" not in remaps
    assert remaps["slip and fall"] == fall_label
    assert remaps["customer fall"] == fall_label


def test_incident_remaps_do_not_narrow_or_escalate_aliases() -> None:
    """Broad incident phrases must not remap to a narrower or more severe label."""
    document = yaml.safe_load(
        (CATALOG_DIR / "incident_video_analytics.yaml").read_text(encoding="utf-8")
    )
    remaps = document["remap"]
    forbidden_sources = {
        "stealing",
        "car taken",
        "vehicle taken",
        "auto taken",
        "gun",
        "knife",
        "firearm",
        "attack",
        "weapon",
        "bomb",
        "ied",
        "fire",
        "arson",
        "drug activity",
        "property damage",
        "active intruder",
    }
    assert forbidden_sources.isdisjoint(remaps)
    assert remaps.get("shooting") == "shooting or firearm discharge"
    assert remaps.get("bike taken") == "bicycle or scooter taken"


def test_catalog_domains_match_skill_table() -> None:
    """Catalog domain strings stay aligned with the skill catalog table."""
    paths = _catalog_paths()
    assert {path.name for path in paths} == set(EXPECTED_DOMAINS)

    for path in paths:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert document["domain"] == EXPECTED_DOMAINS[path.name], path.name
        assert isinstance(document["baseline_event"], str) and document["baseline_event"].strip()
        keywords = document["keywords"]
        assert isinstance(keywords, list), path.name
        assert keywords, path.name
        assert all(isinstance(item, str) and item.strip() for item in keywords), path.name
