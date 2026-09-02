# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Application workflows for TAO TMM nearest neighbors and unique matching.

Orchestrates Make/CLI requests into
:class:`~adapters.docker_jobs.DataMiningDockerRunner` calls for
``tmm nearest_neighbors`` and ``tmm unique_neighbor_matching``, including
config-file vs direct-override modes and embedding-backend evidence.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol


class DataMiningRunResult(Protocol):
    """Minimum result contract needed by the Data Mining workflow."""

    @property
    def command(self) -> Sequence[str]:
        """Command used by the underlying runner."""
        ...

    @property
    def evidence(self) -> dict[str, Any]:
        """Generated experiment spec and post-run output validation."""
        ...


class DataMiningSelectionRunner(Protocol):
    """Port for infrastructure that can run TAO Data Mining selection."""

    def tmm_nearest_neighbors_config(
        self,
        *,
        config_file: str,
        data_dir: str,
        dry_run: bool = False,
    ) -> DataMiningRunResult:
        """Run TMM nearest-neighbor mining from a vendor YAML."""

    def tmm_nearest_neighbors(
        self,
        *,
        data_dir: str,
        target_subdir: str = "S",
        source_subdir: str = "B",
        output_subdir: str = "divknn_out",
        prep_subdir: str = "_tmm_prep",
        topn: int = 5,
        metric: str = "cosine",
        filter_by_label: bool = False,
        distance_threshold: float = -1.0,
        source_embed_column_name: str = "embedding",
        target_embed_column_name: str = "embedding",
        dry_run: bool = False,
    ) -> DataMiningRunResult:
        """Run TMM nearest-neighbor mining from direct CLI options."""

    def tmm_unique_neighbor_matching_config(
        self,
        *,
        config_file: str,
        data_dir: str,
        dry_run: bool = False,
    ) -> DataMiningRunResult:
        """Run unique_neighbor_matching from a vendor YAML."""

    def tmm_unique_neighbor_matching(
        self,
        *,
        data_dir: str,
        target_subdir: str = "S",
        source_subdir: str = "B",
        output_subdir: str = "unm_out",
        prep_subdir: str = "_tmm_prep",
        desired_unique_count: int = 100,
        allocation_policy: str = "global",
        metric: str = "euclidean",
        candidate_expansion_factor: int = 5,
        source_embedding_column: str = "embedding",
        target_embedding_column: str = "embedding",
        source_filepath_column: str = "filepath",
        target_filepath_column: str = "filepath",
        exclude_path: str | None = None,
        source_detection_file: str | None = None,
        target_detection_file: str | None = None,
        detection_format: str | None = None,
        rare_class_list: str = "",
        save_embeddings: bool = False,
        visualize: bool = False,
        dry_run: bool = False,
    ) -> DataMiningRunResult:
        """Run unique_neighbor_matching from direct CLI options."""


@dataclass(frozen=True)
class RunDataMiningSelectionRequest:
    """Inputs for TAO Data Mining nearest-neighbor selection."""

    data_dir: str
    config_file: str | None = None
    target_subdir: str = "S"
    source_subdir: str = "B"
    output_subdir: str = "divknn_out"
    topn: int = 5
    metric: str = "cosine"
    embedding_backend: str = "ce1"
    dry_run: bool = False
    filter_by_label: bool = False
    distance_threshold: float = -1.0
    source_embed_column_name: str = "embedding"
    target_embed_column_name: str = "embedding"
    in_process: bool = False


@dataclass(frozen=True)
class RunDataMiningSelectionResult:
    """Stable result returned to CLI/API presenters."""

    status: str
    dry_run: bool
    embedding_backend: str
    command: list[str]
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RunUniqueNeighborMatchingRequest:
    """Inputs for TAO unique_neighbor_matching selection."""

    data_dir: str
    config_file: str | None = None
    target_subdir: str = "S"
    source_subdir: str = "B"
    output_subdir: str = "unm_out"
    desired_unique_count: int = 100
    allocation_policy: str = "global"
    metric: str = "euclidean"
    candidate_expansion_factor: int = 5
    source_embedding_column: str = "embedding"
    target_embedding_column: str = "embedding"
    source_filepath_column: str = "filepath"
    target_filepath_column: str = "filepath"
    exclude_path: str | None = None
    source_detection_file: str | None = None
    target_detection_file: str | None = None
    detection_format: str | None = None
    rare_class_list: str = ""
    save_embeddings: bool = False
    visualize: bool = False
    embedding_backend: str = "ce1"
    dry_run: bool = False


@dataclass(frozen=True)
class RunUniqueNeighborMatchingResult:
    """Stable result returned for unique_neighbor_matching runs."""

    status: str
    dry_run: bool
    embedding_backend: str
    command: list[str]
    evidence: dict[str, Any] = field(default_factory=dict)


def _run_evidence(result: DataMiningRunResult) -> dict[str, Any]:
    """Return runner evidence, tolerating runners that do not publish any."""
    evidence = getattr(result, "evidence", None)
    return dict(evidence) if isinstance(evidence, dict) else {}


def run_data_mining_selection(
    request: RunDataMiningSelectionRequest,
    *,
    runner_factory: Callable[[], DataMiningSelectionRunner],
) -> RunDataMiningSelectionResult:
    """Run TAO Data Mining selection through the configured runner."""
    if request.in_process:
        raise ValueError(
            "in-process mode needs EmbeddingRecords; use apps.cli.curate or --docker with image"
        )

    runner = runner_factory()
    if request.config_file:
        result = runner.tmm_nearest_neighbors_config(
            config_file=request.config_file,
            data_dir=request.data_dir,
            dry_run=request.dry_run,
        )
    else:
        result = runner.tmm_nearest_neighbors(
            data_dir=request.data_dir,
            target_subdir=request.target_subdir,
            source_subdir=request.source_subdir,
            output_subdir=request.output_subdir,
            topn=request.topn,
            metric=request.metric,
            filter_by_label=request.filter_by_label,
            distance_threshold=request.distance_threshold,
            source_embed_column_name=request.source_embed_column_name,
            target_embed_column_name=request.target_embed_column_name,
            dry_run=request.dry_run,
        )

    return RunDataMiningSelectionResult(
        status="dry-run" if request.dry_run else "completed",
        dry_run=request.dry_run,
        embedding_backend=request.embedding_backend.lower(),
        command=list(result.command),
        evidence=_run_evidence(result),
    )


def run_unique_neighbor_matching(
    request: RunUniqueNeighborMatchingRequest,
    *,
    runner_factory: Callable[[], DataMiningSelectionRunner],
) -> RunUniqueNeighborMatchingResult:
    """Run TAO unique_neighbor_matching through the configured runner."""
    runner = runner_factory()
    if request.config_file:
        result = runner.tmm_unique_neighbor_matching_config(
            config_file=request.config_file,
            data_dir=request.data_dir,
            dry_run=request.dry_run,
        )
    else:
        result = runner.tmm_unique_neighbor_matching(
            data_dir=request.data_dir,
            target_subdir=request.target_subdir,
            source_subdir=request.source_subdir,
            output_subdir=request.output_subdir,
            desired_unique_count=request.desired_unique_count,
            allocation_policy=request.allocation_policy,
            metric=request.metric,
            candidate_expansion_factor=request.candidate_expansion_factor,
            source_embedding_column=request.source_embedding_column,
            target_embedding_column=request.target_embedding_column,
            source_filepath_column=request.source_filepath_column,
            target_filepath_column=request.target_filepath_column,
            exclude_path=request.exclude_path,
            source_detection_file=request.source_detection_file,
            target_detection_file=request.target_detection_file,
            detection_format=request.detection_format,
            rare_class_list=request.rare_class_list,
            save_embeddings=request.save_embeddings,
            visualize=request.visualize,
            dry_run=request.dry_run,
        )
    return RunUniqueNeighborMatchingResult(
        status="dry-run" if request.dry_run else "completed",
        dry_run=request.dry_run,
        embedding_backend=request.embedding_backend.lower(),
        command=list(result.command),
        evidence=_run_evidence(result),
    )
