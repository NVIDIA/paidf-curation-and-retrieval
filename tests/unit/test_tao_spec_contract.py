# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contract tests pinning generated specs to the TAO DS RC36 schema.

The vendor field sets below were dumped from the dataclasses in
``nvidia_tao_ds.config.mining`` inside the pinned TAO Data Services image
(``nvcr.io/nvidia/tao/tao-toolkit:7.2.0-data-services``). They exist so
that a vendor schema change, or a knob we forget to plumb, fails here instead of
inside a GPU run.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from adapters.docker_jobs import DataMiningDockerRunner

IMAGE_EMBEDDINGS_FIELDS = frozenset(
    {
        "input_parquet",
        "output_parquet",
        "model",
        "model_path",
        "model_config_path",
        "batch_size",
    }
)
TEXT_EMBEDDINGS_FIELDS = frozenset(
    {
        "input_parquet",
        "output_parquet",
        "model",
        "model_path",
        "batch_size",
    }
)
NEAREST_NEIGHBORS_FIELDS = frozenset(
    {
        "source_parquet",
        "target_parquet",
        "output_parquet",
        "topn",
        "knn_metric",
        "source_embed_column_name",
        "target_embed_column_name",
        "filter_by_label",
        "distance_threshold",
    }
)
UNIQUE_NEIGHBOR_MATCHING_FIELDS = frozenset(
    {
        "source_path",
        "target_path",
        "output_dir",
        "desired_unique_count",
        "allocation_policy",
        "distance_metric",
        "candidate_expansion_factor",
        "source_embedding_column",
        "target_embedding_column",
        "source_filepath_column",
        "target_filepath_column",
        "exclude_path",
        "source_detection_file",
        "target_detection_file",
        "detection_format",
        "rare_class_list",
        "save_embeddings",
        "visualize",
    }
)


def _write_embedding_pair(data_dir: Path) -> None:
    frame = pd.DataFrame({"filepath": ["a.jpg", "b.jpg"], "embedding": [[1.0, 0.0], [0.0, 1.0]]})
    frame.to_parquet(data_dir / "S.parquet", index=False)
    frame.to_parquet(data_dir / "B.parquet", index=False)


@pytest.fixture()
def runner() -> DataMiningDockerRunner:
    return DataMiningDockerRunner(image="tao-toolkit-ds:test")


class TestGeneratedSpecsMatchVendorSchema:
    """Every generated spec key must exist in the vendor dataclass."""

    def test_image_embeddings_spec_keys(
        self, runner: DataMiningDockerRunner, tmp_path: Path
    ) -> None:
        (tmp_path / "a.jpg").write_bytes(b"jpeg")
        pd.DataFrame({"filepath": ["a.jpg"], "label": ["x"]}).to_parquet(
            tmp_path / "input.parquet", index=False
        )

        result = runner.image_embeddings(
            data_dir=str(tmp_path),
            input_parquet=str(tmp_path / "input.parquet"),
            output_parquet=str(tmp_path / "output.parquet"),
            model_type="clip",
            model_name_or_path="openai/clip-vit-base-patch32",
            batch_size=4,
            dry_run=True,
        )

        spec = result.evidence["experiment_spec"]
        assert set(spec) <= IMAGE_EMBEDDINGS_FIELDS
        assert IMAGE_EMBEDDINGS_FIELDS - set(spec) == {"model_config_path"}

    def test_text_embeddings_spec_keys(
        self, runner: DataMiningDockerRunner, tmp_path: Path
    ) -> None:
        pd.DataFrame({"text": ["a caption"]}).to_parquet(tmp_path / "input.parquet", index=False)

        result = runner.text_embeddings(
            data_dir=str(tmp_path),
            input_parquet=str(tmp_path / "input.parquet"),
            output_parquet=str(tmp_path / "output.parquet"),
            model="clip",
            model_path="openai/clip-vit-base-patch32",
            dry_run=True,
        )

        assert set(result.evidence["experiment_spec"]) == TEXT_EMBEDDINGS_FIELDS

    def test_nearest_neighbors_spec_keys(
        self, runner: DataMiningDockerRunner, tmp_path: Path
    ) -> None:
        _write_embedding_pair(tmp_path)

        result = runner.tmm_nearest_neighbors(
            data_dir=str(tmp_path),
            target_subdir="S.parquet",
            source_subdir="B.parquet",
            dry_run=True,
        )

        assert set(result.evidence["experiment_spec"]) == NEAREST_NEIGHBORS_FIELDS

    def test_unique_neighbor_matching_spec_keys(
        self, runner: DataMiningDockerRunner, tmp_path: Path
    ) -> None:
        _write_embedding_pair(tmp_path)
        (tmp_path / "labels").mkdir()
        (tmp_path / "labels" / "a.txt").write_text("car 0 0 0 1 1 2 2\n", encoding="utf-8")
        (tmp_path / "exclude.parquet").write_bytes(b"")

        result = runner.tmm_unique_neighbor_matching(
            data_dir=str(tmp_path),
            target_subdir="S.parquet",
            source_subdir="B.parquet",
            allocation_policy="class_stratified",
            desired_unique_count=1,
            source_detection_file="labels",
            target_detection_file="labels",
            detection_format="kitti",
            rare_class_list="car",
            exclude_path="exclude.parquet",
            save_embeddings=True,
            visualize=True,
            dry_run=True,
        )

        assert set(result.evidence["experiment_spec"]) == UNIQUE_NEIGHBOR_MATCHING_FIELDS
