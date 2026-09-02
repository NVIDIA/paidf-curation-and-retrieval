# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validation for external Cosmos Curator pipeline configurations."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import click
import yaml

DEPRECATED_PIPELINE_KEYS = {
    "enable_sam3": "sam3",
    "enable_event_captioning": "event_captioning",
}


def validate_curator_pipeline_config(config_file: str | Path) -> None:
    """Reject deprecated pipeline keys before Curator execution."""
    path = Path(config_file)
    try:
        config: object = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise click.ClickException(
            f"Curator pipeline configuration could not be read: {path}"
        ) from exc

    if not isinstance(config, Mapping):
        raise click.ClickException("Curator pipeline configuration must be a YAML mapping")

    parameter_mappings = [config]
    args = config.get("args")
    if isinstance(args, Mapping):
        parameter_mappings.append(args)

    deprecated_keys = [
        key
        for key in DEPRECATED_PIPELINE_KEYS
        if any(key in parameters for parameters in parameter_mappings)
    ]
    if deprecated_keys:
        replacements = ", ".join(
            f"'{key}' with '{DEPRECATED_PIPELINE_KEYS[key]}'" for key in deprecated_keys
        )
        raise click.ClickException(
            f"Curator pipeline configuration uses deprecated keys; replace {replacements}"
        )
