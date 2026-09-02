# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for unique_neighbor_matching config validation and docker wiring."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from adapters.data_mining.tmm_parquet import TmmParquetError
from adapters.data_mining.tmm_unm import (
    resolve_optional_data_selection,
    validate_unm_config_inputs,
    validate_unm_direct_options,
    validate_unm_output,
)
from adapters.docker_jobs import DataMiningDockerRunner


def _write_parquet(path: Path, *, ids: list[str], dimension: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        {
            "filepath": ids,
            "embedding": [[float(i)] * dimension for i, _ in enumerate(ids)],
        }
    )
    frame.to_parquet(path, index=False)


class TestValidateUnmConfigInputs:
    def test_accepts_global_policy(self, tmp_path: Path) -> None:
        data = tmp_path / "data"
        _write_parquet(data / "S.parquet", ids=["t0"])
        _write_parquet(data / "B.parquet", ids=["s0"])
        config = tmp_path / "unm.yaml"
        config.write_text(
            yaml.safe_dump(
                {
                    "source_path": "/data/B.parquet",
                    "target_path": "/data/S.parquet",
                    "output_dir": "/data/unm_out",
                    "desired_unique_count": 10,
                    "allocation_policy": "global",
                    "distance_metric": "cosine",
                    "candidate_expansion_factor": 5,
                }
            ),
            encoding="utf-8",
        )
        assert validate_unm_config_inputs(config, data) == 2

    def test_class_stratified_requires_detections(self, tmp_path: Path) -> None:
        data = tmp_path / "data"
        _write_parquet(data / "S.parquet", ids=["t0"])
        _write_parquet(data / "B.parquet", ids=["s0"])
        config = tmp_path / "unm.yaml"
        config.write_text(
            yaml.safe_dump(
                {
                    "source_path": "/data/B.parquet",
                    "target_path": "/data/S.parquet",
                    "output_dir": "/data/unm_out",
                    "desired_unique_count": 10,
                    "allocation_policy": "class_stratified",
                }
            ),
            encoding="utf-8",
        )
        with pytest.raises(TmmParquetError, match="source_detection_file"):
            validate_unm_config_inputs(config, data)

    def test_class_stratified_accepts_detection_files(self, tmp_path: Path) -> None:
        data = tmp_path / "data"
        _write_parquet(data / "S.parquet", ids=["t0"])
        _write_parquet(data / "B.parquet", ids=["s0"])
        (data / "src_det.json").write_text("{}", encoding="utf-8")
        (data / "tgt_det.json").write_text("{}", encoding="utf-8")
        config = tmp_path / "unm.yaml"
        config.write_text(
            yaml.safe_dump(
                {
                    "source_path": "/data/B.parquet",
                    "target_path": "/data/S.parquet",
                    "output_dir": "/data/unm_out",
                    "desired_unique_count": 10,
                    "allocation_policy": "class_stratified",
                    "source_detection_file": "/data/src_det.json",
                    "target_detection_file": "/data/tgt_det.json",
                    "detection_format": "coco",
                    "rare_class_list": "person,bicycle",
                }
            ),
            encoding="utf-8",
        )
        assert validate_unm_config_inputs(config, data) == 2

    def test_class_stratified_accepts_kitti_label_directories(self, tmp_path: Path) -> None:
        data = tmp_path / "data"
        _write_parquet(data / "S.parquet", ids=["t0"])
        _write_parquet(data / "B.parquet", ids=["s0"])
        for role in ("src_labels", "tgt_labels"):
            labels = data / role
            labels.mkdir()
            (labels / "000000.txt").write_text(
                "car 0 0 0 1 2 3 4 0 0 0 0 0 0 0\n", encoding="utf-8"
            )
        config = tmp_path / "unm.yaml"
        config.write_text(
            yaml.safe_dump(
                {
                    "source_path": "/data/B.parquet",
                    "target_path": "/data/S.parquet",
                    "output_dir": "/data/unm_out",
                    "desired_unique_count": 10,
                    "allocation_policy": "class_stratified",
                    "source_detection_file": "/data/src_labels",
                    "target_detection_file": "/data/tgt_labels",
                    "detection_format": "kitti",
                    "rare_class_list": "car",
                }
            ),
            encoding="utf-8",
        )
        assert validate_unm_config_inputs(config, data) == 2

    def test_class_stratified_rejects_coco_detection_directories(self, tmp_path: Path) -> None:
        data = tmp_path / "data"
        _write_parquet(data / "S.parquet", ids=["t0"])
        _write_parquet(data / "B.parquet", ids=["s0"])
        (data / "src_det_dir").mkdir()
        (data / "tgt_det.json").write_text("{}", encoding="utf-8")
        config = tmp_path / "unm.yaml"
        config.write_text(
            yaml.safe_dump(
                {
                    "source_path": "/data/B.parquet",
                    "target_path": "/data/S.parquet",
                    "output_dir": "/data/unm_out",
                    "desired_unique_count": 10,
                    "allocation_policy": "class_stratified",
                    "source_detection_file": "/data/src_det_dir",
                    "target_detection_file": "/data/tgt_det.json",
                    "detection_format": "coco",
                    "rare_class_list": "person",
                }
            ),
            encoding="utf-8",
        )
        with pytest.raises(TmmParquetError, match="must be a COCO JSON file"):
            validate_unm_config_inputs(config, data)

    def test_rejects_missing_desired_unique_count(self, tmp_path: Path) -> None:
        data = tmp_path / "data"
        _write_parquet(data / "S.parquet", ids=["t0"])
        _write_parquet(data / "B.parquet", ids=["s0"])
        config = tmp_path / "unm.yaml"
        config.write_text(
            yaml.safe_dump(
                {
                    "source_path": "/data/B.parquet",
                    "target_path": "/data/S.parquet",
                    "output_dir": "/data/unm_out",
                }
            ),
            encoding="utf-8",
        )
        with pytest.raises(TmmParquetError, match="desired_unique_count"):
            validate_unm_config_inputs(config, data)

    def test_cookbook_unm_yaml_declares_desired_unique_count(self) -> None:
        """Regression: cookbook example must pass validate_unm_config_inputs field guard."""
        cookbook = (
            Path(__file__).resolve().parents[2]
            / "cookbook"
            / "unique-neighbor-matching"
            / "unm.yaml"
        )
        payload = yaml.safe_load(cookbook.read_text(encoding="utf-8"))
        desired = payload.get("desired_unique_count")
        assert isinstance(desired, int) and not isinstance(desired, bool)
        assert desired >= 1


class TestValidateUnmDirectOptions:
    def test_global_defaults(self) -> None:
        options = validate_unm_direct_options()
        assert options["allocation_policy"] == "global"
        assert options["distance_metric"] == "euclidean"
        assert options["save_embeddings"] is False
        assert options["visualize"] is False

    def test_class_stratified_requires_rare_class_list(self) -> None:
        with pytest.raises(TmmParquetError, match="rare_class_list"):
            validate_unm_direct_options(
                allocation_policy="class_stratified",
                source_detection_file="/data/src.json",
                target_detection_file="/data/tgt.json",
                detection_format="coco",
                rare_class_list="",
            )

    def test_class_stratified_requires_target_detection_and_format(self) -> None:
        with pytest.raises(TmmParquetError, match="target_detection_file"):
            validate_unm_direct_options(
                allocation_policy="class_stratified",
                source_detection_file="/data/src.json",
                target_detection_file=None,
                detection_format="coco",
                rare_class_list="person",
            )
        with pytest.raises(TmmParquetError, match="detection_format"):
            validate_unm_direct_options(
                allocation_policy="class_stratified",
                source_detection_file="/data/src.json",
                target_detection_file="/data/tgt.json",
                detection_format=None,
                rare_class_list="person",
            )

    def test_rejects_invalid_detection_format(self) -> None:
        with pytest.raises(TmmParquetError, match="detection_format"):
            validate_unm_direct_options(detection_format="yolo")

    def test_rejects_invalid_counts_and_flags(self) -> None:
        with pytest.raises(TmmParquetError, match="desired_unique_count"):
            validate_unm_direct_options(desired_unique_count=0)
        with pytest.raises(TmmParquetError, match="candidate_expansion_factor"):
            validate_unm_direct_options(candidate_expansion_factor=True)  # type: ignore[arg-type]
        with pytest.raises(TmmParquetError, match="allocation_policy"):
            validate_unm_direct_options(allocation_policy="balanced")
        with pytest.raises(TmmParquetError, match="distance_metric"):
            validate_unm_direct_options(metric="l2")
        with pytest.raises(TmmParquetError, match="save_embeddings"):
            validate_unm_direct_options(save_embeddings="yes")  # type: ignore[arg-type]
        with pytest.raises(TmmParquetError, match="visualize"):
            validate_unm_direct_options(visualize="no")  # type: ignore[arg-type]


class TestResolveOptionalDataSelection:
    def test_none_and_blank_return_none(self, tmp_path: Path) -> None:
        data = tmp_path / "data"
        data.mkdir()
        assert resolve_optional_data_selection(data, None, role="exclude_path") is None
        assert resolve_optional_data_selection(data, "  ", role="exclude_path") is None

    def test_accepts_data_prefix_and_relative(self, tmp_path: Path) -> None:
        data = tmp_path / "data"
        (data / "exclude.parquet").parent.mkdir(parents=True, exist_ok=True)
        (data / "exclude.parquet").write_text("x", encoding="utf-8")
        assert (
            resolve_optional_data_selection(data, "exclude.parquet", role="exclude_path")
            == "/data/exclude.parquet"
        )
        assert (
            resolve_optional_data_selection(data, "/data/exclude.parquet", role="exclude_path")
            == "/data/exclude.parquet"
        )

    def test_rejects_outside_and_missing(self, tmp_path: Path) -> None:
        data = tmp_path / "data"
        data.mkdir()
        with pytest.raises(TmmParquetError, match="under DATA_DIR"):
            resolve_optional_data_selection(data, "../escape", role="exclude_path")
        with pytest.raises(TmmParquetError, match="not found"):
            resolve_optional_data_selection(data, "missing.parquet", role="exclude_path")

    def test_coco_detection_roles_reject_directories(self, tmp_path: Path) -> None:
        data = tmp_path / "data"
        (data / "detections").mkdir(parents=True)
        for role in ("source_detection_file", "target_detection_file"):
            with pytest.raises(TmmParquetError, match="must be a COCO JSON file"):
                resolve_optional_data_selection(
                    data,
                    "detections",
                    role=role,
                    detection_format="coco",
                )

    def test_detection_directory_without_format_is_rejected(self, tmp_path: Path) -> None:
        data = tmp_path / "data"
        (data / "detections").mkdir(parents=True)
        with pytest.raises(TmmParquetError, match="requires detection_format kitti"):
            resolve_optional_data_selection(data, "detections", role="source_detection_file")

    def test_kitti_detection_roles_accept_label_directories(self, tmp_path: Path) -> None:
        data = tmp_path / "data"
        labels = data / "labels"
        labels.mkdir(parents=True)
        (labels / "000000.txt").write_text("car 0 0 0 1 2 3 4 0 0 0 0 0 0 0\n", encoding="utf-8")

        assert (
            resolve_optional_data_selection(
                data,
                "labels",
                role="source_detection_file",
                detection_format="kitti",
            )
            == "/data/labels"
        )

    def test_kitti_detection_directory_requires_txt_labels(self, tmp_path: Path) -> None:
        data = tmp_path / "data"
        (data / "labels").mkdir(parents=True)
        with pytest.raises(TmmParquetError, match="no .txt label files"):
            resolve_optional_data_selection(
                data,
                "labels",
                role="source_detection_file",
                detection_format="kitti",
            )

    def test_kitti_detection_roles_reject_files(self, tmp_path: Path) -> None:
        data = tmp_path / "data"
        data.mkdir()
        (data / "det.json").write_text("{}", encoding="utf-8")
        with pytest.raises(TmmParquetError, match="must be a KITTI label directory"):
            resolve_optional_data_selection(
                data,
                "det.json",
                role="source_detection_file",
                detection_format="kitti",
            )

    def test_exclude_path_still_allows_existing_directory(self, tmp_path: Path) -> None:
        data = tmp_path / "data"
        exclude_dir = data / "exclude_dir"
        exclude_dir.mkdir(parents=True)
        assert (
            resolve_optional_data_selection(data, "exclude_dir", role="exclude_path")
            == "/data/exclude_dir"
        )


class TestValidateUnmOutput:
    def test_requires_final_parquet(self, tmp_path: Path) -> None:
        data = tmp_path / "data"
        out = data / "unm_out"
        out.mkdir(parents=True)
        with pytest.raises(TmmParquetError, match="final_unique_files.parquet"):
            validate_unm_output(out, data)

    def test_accepts_final_parquet_without_summary(self, tmp_path: Path) -> None:
        data = tmp_path / "data"
        out = data / "unm_out"
        out.mkdir(parents=True)
        (out / "final_unique_files.parquet").write_bytes(b"parquet")
        evidence = validate_unm_output(out, data)
        assert evidence["summary_json"] is None
        assert evidence["final_unique_files_parquet"].endswith("final_unique_files.parquet")

    def test_relative_output_and_outside_root(self, tmp_path: Path) -> None:
        data = tmp_path / "data"
        out = data / "unm_out"
        out.mkdir(parents=True)
        (out / "final_unique_files.parquet").write_bytes(b"parquet")
        (out / "summary.json").write_text("{}", encoding="utf-8")
        evidence = validate_unm_output("unm_out", data)
        assert evidence["summary_json"] is not None
        with pytest.raises(TmmParquetError, match="under DATA_DIR"):
            validate_unm_output(tmp_path / "other", data)


class TestUniqueNeighborMatchingDocker:
    def test_dry_run_builds_unique_neighbor_matching_command(self, tmp_path: Path) -> None:
        data = tmp_path / "data"
        data.mkdir()
        runner = DataMiningDockerRunner(image="tao-toolkit-ds:test")
        result = runner.tmm_unique_neighbor_matching(
            data_dir=str(data),
            desired_unique_count=25,
            allocation_policy="global",
            metric="cosine",
            dry_run=True,
        )
        assert result.returncode == 0
        assert "unique_neighbor_matching" in result.command
        assert "--entrypoint" in result.command
        assert "tmm" in result.command
        assert any("generated-tmm-unm.yaml" in part for part in result.command)
        spec = result.evidence["experiment_spec"]
        assert spec["desired_unique_count"] == 25
        assert spec["allocation_policy"] == "global"
        assert spec["source_embedding_column"] == "embedding"
        assert spec["exclude_path"] is None

    def test_dry_run_class_stratified_includes_detection_knobs(self, tmp_path: Path) -> None:
        data = tmp_path / "data"
        data.mkdir()
        runner = DataMiningDockerRunner(image="tao-toolkit-ds:test")
        result = runner.tmm_unique_neighbor_matching(
            data_dir=str(data),
            allocation_policy="class_stratified",
            source_detection_file="labels/source",
            target_detection_file="labels/target",
            detection_format="kitti",
            rare_class_list="car,truck",
            exclude_path="exclude.parquet",
            save_embeddings=True,
            visualize=True,
            source_embedding_column="vec",
            dry_run=True,
        )
        assert result.returncode == 0
        spec = result.evidence["experiment_spec"]
        assert spec["allocation_policy"] == "class_stratified"
        assert spec["source_detection_file"] == "/data/labels/source"
        assert spec["target_detection_file"] == "/data/labels/target"
        assert spec["detection_format"] == "kitti"
        assert spec["rare_class_list"] == "car,truck"
        assert spec["exclude_path"] == "/data/exclude.parquet"
        assert spec["save_embeddings"] is True
        assert spec["visualize"] is True
        assert spec["source_embedding_column"] == "vec"

    def test_nearest_neighbors_includes_distance_threshold(self, tmp_path: Path) -> None:
        data = tmp_path / "data"
        data.mkdir()
        runner = DataMiningDockerRunner(image="tao-toolkit-ds:test")
        result = runner.tmm_nearest_neighbors(
            data_dir=str(data),
            distance_threshold=0.4,
            source_embed_column_name="src_emb",
            target_embed_column_name="tgt_emb",
            dry_run=True,
        )
        assert result.returncode == 0
        assert "nearest_neighbors" in result.command
        evidence = result.evidence or {}
        assert evidence.get("experiment_spec", {}).get("distance_threshold") == 0.4
        assert evidence["experiment_spec"]["knn_metric"] == "cosine"
        assert evidence["experiment_spec"]["source_embed_column_name"] == "src_emb"
        assert evidence["experiment_spec"]["target_embed_column_name"] == "tgt_emb"
