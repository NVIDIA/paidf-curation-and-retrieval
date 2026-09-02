# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Curator pipeline application workflows."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


class CuratorPipelineRunResult(Protocol):
    """Minimum result contract needed by the Curator pipeline workflow."""

    @property
    def command(self) -> Sequence[str]:
        """Command used by the underlying runner."""
        ...


class CuratorPipelineRunner(Protocol):
    """Port for infrastructure that can run a Curator pipeline."""

    def run_pipeline(
        self,
        config_file: str,
        *,
        data_dir: str | None = None,
        models_dir: str | None = None,
        ffmpeg_dir: str | None = None,
        dry_run: bool = False,
        extra_args: Sequence[str] | None = None,
    ) -> CuratorPipelineRunResult:
        """Run or dry-run the external Curator pipeline command."""


@dataclass(frozen=True)
class RunCuratorPipelineRequest:
    """Inputs for running a Curator pipeline from any delivery mechanism."""

    config_file: str
    data_dir: str | None = None
    models_dir: str | None = None
    ffmpeg_dir: str | None = None
    dry_run: bool = False


@dataclass(frozen=True)
class RunCuratorPipelineResult:
    """Stable result returned to CLI/API presenters."""

    dry_run: bool
    command: list[str]


def run_curator_pipeline(
    request: RunCuratorPipelineRequest,
    *,
    runner_factory: Callable[[], CuratorPipelineRunner],
    validate_config: Callable[[str | Path], None],
) -> RunCuratorPipelineResult:
    """Validate and run a Curator pipeline through the configured runner."""
    validate_config(request.config_file)
    runner = runner_factory()
    result = runner.run_pipeline(
        request.config_file,
        data_dir=request.data_dir,
        models_dir=request.models_dir,
        ffmpeg_dir=request.ffmpeg_dir,
        dry_run=request.dry_run,
    )
    return RunCuratorPipelineResult(
        dry_run=request.dry_run,
        command=list(result.command),
    )
