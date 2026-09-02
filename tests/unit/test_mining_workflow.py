# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Data Mining workflow request wiring."""

from __future__ import annotations

from typing import Any

from apps.workflows.mining import (
    RunDataMiningSelectionRequest,
    RunUniqueNeighborMatchingRequest,
    run_data_mining_selection,
    run_unique_neighbor_matching,
)


class _FakeResult:
    def __init__(self, command: list[str]) -> None:
        self.command = command
        self.evidence: dict[str, Any] = {}


class _FakeRunner:
    def __init__(self) -> None:
        self.nn_kwargs: dict[str, Any] = {}
        self.unm_kwargs: dict[str, Any] = {}

    def tmm_nearest_neighbors_config(
        self,
        *,
        config_file: str,
        data_dir: str,
        dry_run: bool = False,
    ) -> _FakeResult:
        return _FakeResult(["config-nn"])

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
    ) -> _FakeResult:
        self.nn_kwargs = {
            "data_dir": data_dir,
            "target_subdir": target_subdir,
            "source_subdir": source_subdir,
            "output_subdir": output_subdir,
            "prep_subdir": prep_subdir,
            "topn": topn,
            "metric": metric,
            "filter_by_label": filter_by_label,
            "distance_threshold": distance_threshold,
            "source_embed_column_name": source_embed_column_name,
            "target_embed_column_name": target_embed_column_name,
            "dry_run": dry_run,
        }
        return _FakeResult(["direct-nn"])

    def tmm_unique_neighbor_matching_config(
        self,
        *,
        config_file: str,
        data_dir: str,
        dry_run: bool = False,
    ) -> _FakeResult:
        return _FakeResult(["config-unm"])

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
    ) -> _FakeResult:
        self.unm_kwargs = {
            "data_dir": data_dir,
            "target_subdir": target_subdir,
            "source_subdir": source_subdir,
            "output_subdir": output_subdir,
            "prep_subdir": prep_subdir,
            "desired_unique_count": desired_unique_count,
            "allocation_policy": allocation_policy,
            "metric": metric,
            "candidate_expansion_factor": candidate_expansion_factor,
            "source_embedding_column": source_embedding_column,
            "target_embedding_column": target_embedding_column,
            "source_filepath_column": source_filepath_column,
            "target_filepath_column": target_filepath_column,
            "exclude_path": exclude_path,
            "source_detection_file": source_detection_file,
            "target_detection_file": target_detection_file,
            "detection_format": detection_format,
            "rare_class_list": rare_class_list,
            "save_embeddings": save_embeddings,
            "visualize": visualize,
            "dry_run": dry_run,
        }
        return _FakeResult(["direct-unm"])


def test_run_data_mining_selection_passes_embed_columns() -> None:
    runner = _FakeRunner()
    result = run_data_mining_selection(
        RunDataMiningSelectionRequest(
            data_dir="/data",
            distance_threshold=0.2,
            source_embed_column_name="src",
            target_embed_column_name="tgt",
        ),
        runner_factory=lambda: runner,
    )
    assert result.command == ["direct-nn"]
    assert result.embedding_backend == "ce1"
    assert runner.nn_kwargs["distance_threshold"] == 0.2
    assert runner.nn_kwargs["source_embed_column_name"] == "src"
    assert runner.nn_kwargs["target_embed_column_name"] == "tgt"


def test_run_data_mining_selection_config_and_in_process() -> None:
    runner = _FakeRunner()
    result = run_data_mining_selection(
        RunDataMiningSelectionRequest(data_dir="/data", config_file="/tmp/tmm.yaml"),
        runner_factory=lambda: runner,
    )
    assert result.command == ["config-nn"]
    try:
        run_data_mining_selection(
            RunDataMiningSelectionRequest(data_dir="/data", in_process=True),
            runner_factory=lambda: runner,
        )
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        assert "in-process" in str(exc)


def test_run_unique_neighbor_matching_passes_stratified_knobs() -> None:
    runner = _FakeRunner()
    result = run_unique_neighbor_matching(
        RunUniqueNeighborMatchingRequest(
            data_dir="/data",
            allocation_policy="class_stratified",
            source_detection_file="src.json",
            target_detection_file="tgt.json",
            detection_format="coco",
            rare_class_list="person",
            save_embeddings=True,
            visualize=True,
        ),
        runner_factory=lambda: runner,
    )
    assert result.command == ["direct-unm"]
    assert result.embedding_backend == "ce1"
    assert runner.unm_kwargs["allocation_policy"] == "class_stratified"
    assert runner.unm_kwargs["source_detection_file"] == "src.json"
    assert runner.unm_kwargs["detection_format"] == "coco"
    assert runner.unm_kwargs["rare_class_list"] == "person"
    assert runner.unm_kwargs["save_embeddings"] is True
    assert runner.unm_kwargs["visualize"] is True


def test_run_unique_neighbor_matching_config_path() -> None:
    runner = _FakeRunner()
    result = run_unique_neighbor_matching(
        RunUniqueNeighborMatchingRequest(data_dir="/data", config_file="/tmp/unm.yaml"),
        runner_factory=lambda: runner,
    )
    assert result.command == ["config-unm"]
