# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for TMM parquet preparation (dir → single file + filepath column)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from adapters.data_mining.tmm_parquet import (
    TMM_METRICS,
    TmmColumnSelection,
    TmmParquetError,
    container_data_path,
    prepare_tmm_parquet_pair,
    validate_embed_column_name,
    validate_tmm_config_inputs,
    validate_tmm_parquet_pair,
    validate_tmm_run_options,
)


def _write_parquet(
    path: Path,
    *,
    id_col: str,
    ids: list[str],
    dimension: int = 2,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        {
            id_col: ids,
            "embedding": [[0.1] * dimension for _ in ids],
        }
    )
    frame.to_parquet(path, index=False)


def _write_custom_columns_parquet(
    path: Path,
    *,
    embed_column: str,
    filepath_column: str,
    ids: list[str],
    dimension: int = 2,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            filepath_column: ids,
            embed_column: [[0.1] * dimension for _ in ids],
        }
    ).to_parquet(path, index=False)


class TestCustomColumnSelection:
    """Custom embedding/filepath column names must survive host-side prep."""

    def test_prepare_accepts_custom_embed_and_filepath_columns(self, tmp_path: Path) -> None:
        data = tmp_path / "data"
        _write_custom_columns_parquet(
            data / "S.parquet",
            embed_column="target_vec",
            filepath_column="asset_uri",
            ids=["t0"],
        )
        _write_custom_columns_parquet(
            data / "B.parquet",
            embed_column="source_vec",
            filepath_column="asset_uri",
            ids=["s0"],
        )
        columns = TmmColumnSelection(
            target_embed="target_vec",
            source_embed="source_vec",
            target_filepath="asset_uri",
            source_filepath="asset_uri",
        )

        target, source = prepare_tmm_parquet_pair(
            data,
            target_subdir="S.parquet",
            source_subdir="B.parquet",
            columns=columns,
        )

        assert list(pd.read_parquet(target).columns) == ["asset_uri", "target_vec"]
        assert list(pd.read_parquet(source).columns) == ["asset_uri", "source_vec"]
        assert validate_tmm_parquet_pair(target, source, columns=columns) == 2

    def test_missing_custom_embed_column_raises(self, tmp_path: Path) -> None:
        data = tmp_path / "data"
        _write_parquet(data / "S.parquet", id_col="filepath", ids=["t0"])
        _write_parquet(data / "B.parquet", id_col="filepath", ids=["s0"])

        with pytest.raises(TmmParquetError, match="must include the target_vec embedding column"):
            prepare_tmm_parquet_pair(
                data,
                target_subdir="S.parquet",
                source_subdir="B.parquet",
                columns=TmmColumnSelection(target_embed="target_vec"),
            )

    def test_missing_custom_filepath_column_raises(self, tmp_path: Path) -> None:
        data = tmp_path / "data"
        _write_parquet(data / "S.parquet", id_col="filepath", ids=["t0"])
        _write_parquet(data / "B.parquet", id_col="filepath", ids=["s0"])

        with pytest.raises(TmmParquetError, match="must include the asset_uri column"):
            prepare_tmm_parquet_pair(
                data,
                target_subdir="S.parquet",
                source_subdir="B.parquet",
                columns=TmmColumnSelection(target_filepath="asset_uri"),
            )

    def test_config_validation_honours_declared_embed_columns(self, tmp_path: Path) -> None:
        data = tmp_path / "data"
        _write_custom_columns_parquet(
            data / "S.parquet",
            embed_column="target_vec",
            filepath_column="filepath",
            ids=["t0"],
        )
        _write_custom_columns_parquet(
            data / "B.parquet",
            embed_column="source_vec",
            filepath_column="filepath",
            ids=["s0"],
        )
        config = tmp_path / "tmm.yaml"
        config.write_text(
            yaml.safe_dump(
                {
                    "target_parquet": "/data/S.parquet",
                    "source_parquet": "/data/B.parquet",
                    "output_parquet": "/data/out/mined.parquet",
                    "topn": 1,
                    "knn_metric": "cosine",
                    "filter_by_label": "false",
                    "target_embed_column_name": "target_vec",
                    "source_embed_column_name": "source_vec",
                }
            ),
            encoding="utf-8",
        )

        assert validate_tmm_config_inputs(config, data) == 2


class TestPrepareTmmParquetPair:
    def test_concat_dirs_and_rename_file_name(self, tmp_path: Path) -> None:
        data = tmp_path / "data"
        _write_parquet(data / "S" / "a.parquet", id_col="file_name", ids=["t1"])
        _write_parquet(data / "S" / "b.parquet", id_col="file_name", ids=["t2"])
        _write_parquet(data / "B" / "c.parquet", id_col="file_name", ids=["s1", "s2"])

        target, source = prepare_tmm_parquet_pair(data)
        assert target.name == "target.parquet"
        assert source.name == "source.parquet"

        tgt = pd.read_parquet(target)
        src = pd.read_parquet(source)
        assert "filepath" in tgt.columns
        assert "file_name" not in tgt.columns
        assert set(tgt["filepath"]) == {"t1", "t2"}
        assert set(src["filepath"]) == {"s1", "s2"}
        assert "embedding" in tgt.columns

    def test_single_file_subdir_with_filepath(self, tmp_path: Path) -> None:
        data = tmp_path / "data"
        _write_parquet(data / "S.parquet", id_col="filepath", ids=["only"])
        _write_parquet(data / "B.parquet", id_col="filepath", ids=["src"])
        # Treat file paths as subdir args relative to data_dir
        target, source = prepare_tmm_parquet_pair(
            data, target_subdir="S.parquet", source_subdir="B.parquet"
        )
        assert pd.read_parquet(target)["filepath"].tolist() == ["only"]
        assert pd.read_parquet(source)["filepath"].tolist() == ["src"]

    def test_default_selection_accepts_s_and_b_parquet_files(self, tmp_path: Path) -> None:
        data = tmp_path / "data"
        _write_parquet(data / "S.parquet", id_col="filepath", ids=["target"])
        _write_parquet(data / "B.parquet", id_col="filepath", ids=["source"])

        target, source = prepare_tmm_parquet_pair(data)

        assert target.is_file()
        assert source.is_file()

    def test_missing_embedding_raises(self, tmp_path: Path) -> None:
        data = tmp_path / "data"
        (data / "S").mkdir(parents=True)
        (data / "B").mkdir(parents=True)
        pd.DataFrame({"filepath": ["x"]}).to_parquet(data / "S" / "a.parquet", index=False)
        pd.DataFrame({"filepath": ["y"], "embedding": [[1.0]]}).to_parquet(
            data / "B" / "a.parquet", index=False
        )
        with pytest.raises(TmmParquetError, match="embedding"):
            prepare_tmm_parquet_pair(data)

    def test_empty_dir_raises(self, tmp_path: Path) -> None:
        data = tmp_path / "data"
        (data / "S").mkdir(parents=True)
        (data / "B").mkdir(parents=True)
        with pytest.raises(TmmParquetError, match="No .parquet"):
            prepare_tmm_parquet_pair(data)

    def test_mixed_dimensions_within_selection_raise(self, tmp_path: Path) -> None:
        data = tmp_path / "data"
        _write_parquet(data / "S" / "a.parquet", id_col="filepath", ids=["a"], dimension=2)
        _write_parquet(data / "S" / "b.parquet", id_col="filepath", ids=["b"], dimension=3)
        _write_parquet(data / "B" / "a.parquet", id_col="filepath", ids=["source"], dimension=2)

        with pytest.raises(TmmParquetError, match="mixed embedding dimensions"):
            prepare_tmm_parquet_pair(data)

    def test_cross_file_dimension_mismatch_raises(self, tmp_path: Path) -> None:
        data = tmp_path / "data"
        _write_parquet(data / "S.parquet", id_col="filepath", ids=["target"], dimension=2)
        _write_parquet(data / "B.parquet", id_col="filepath", ids=["source"], dimension=3)

        with pytest.raises(TmmParquetError, match="do not match"):
            validate_tmm_parquet_pair(data / "S.parquet", data / "B.parquet")

    def test_invalid_embedding_values_raise(self, tmp_path: Path) -> None:
        data = tmp_path / "data"
        data.mkdir()
        pd.DataFrame({"filepath": ["target"], "embedding": [[1.0, float("nan")]]}).to_parquet(
            data / "S.parquet", index=False
        )
        _write_parquet(data / "B.parquet", id_col="filepath", ids=["source"])

        with pytest.raises(TmmParquetError, match="finite numbers"):
            prepare_tmm_parquet_pair(data)

    def test_same_or_outside_selection_raises(self, tmp_path: Path) -> None:
        data = tmp_path / "data"
        _write_parquet(data / "same.parquet", id_col="filepath", ids=["row"])

        with pytest.raises(TmmParquetError, match="must be different"):
            prepare_tmm_parquet_pair(
                data,
                target_subdir="same.parquet",
                source_subdir="same.parquet",
            )
        with pytest.raises(TmmParquetError, match="under DATA_DIR"):
            prepare_tmm_parquet_pair(
                data,
                target_subdir="../outside.parquet",
                source_subdir="same.parquet",
            )

    @pytest.mark.parametrize("prep_subdir", ["", ".", "../outside"])
    def test_output_selection_must_be_bounded(
        self,
        tmp_path: Path,
        prep_subdir: str,
    ) -> None:
        data = tmp_path / "data"
        _write_parquet(data / "target.parquet", id_col="id", ids=["target"])
        _write_parquet(data / "source.parquet", id_col="id", ids=["source"])

        with pytest.raises(TmmParquetError, match="output"):
            prepare_tmm_parquet_pair(
                data,
                target_subdir="target.parquet",
                source_subdir="source.parquet",
                prep_subdir=prep_subdir,
            )

    def test_engine_config_inputs_are_validated_before_execution(self, tmp_path: Path) -> None:
        data = tmp_path / "data"
        _write_parquet(data / "S.parquet", id_col="filepath", ids=["target"], dimension=4)
        _write_parquet(data / "B.parquet", id_col="filepath", ids=["source"], dimension=4)
        config = tmp_path / "tmm.yaml"
        payload = {
            "target_parquet": "/data/S.parquet",
            "source_parquet": "/data/B.parquet",
            "output_parquet": "/data/out/mined.parquet",
            "topn": 1,
            "knn_metric": "manhattan",
            "filter_by_label": "false",
            "future_metadata": {"owner": "operator"},
        }
        config.write_text(yaml.safe_dump(payload), encoding="utf-8")

        assert validate_tmm_config_inputs(config, data) == 4

        payload["target_parquet"] = "/tmp/S.parquet"
        config.write_text(yaml.safe_dump(payload), encoding="utf-8")
        with pytest.raises(TmmParquetError, match="/data/"):
            validate_tmm_config_inputs(config, data)

    @pytest.mark.parametrize("metric", sorted(TMM_METRICS))
    def test_tao_71_tmm_metric_allowlist(self, tmp_path: Path, metric: str) -> None:
        data = tmp_path / "data"
        _write_parquet(data / "S.parquet", id_col="filepath", ids=["target"])
        _write_parquet(data / "B.parquet", id_col="filepath", ids=["source"])
        config = tmp_path / "tmm.yaml"
        config.write_text(
            yaml.safe_dump(
                {
                    "target_parquet": "/data/S.parquet",
                    "source_parquet": "/data/B.parquet",
                    "output_parquet": "/data/out.parquet",
                    "topn": 1,
                    "knn_metric": metric,
                    "filter_by_label": "true",
                }
            ),
            encoding="utf-8",
        )

        assert validate_tmm_config_inputs(config, data) == 2

    def test_tmm_config_accepts_distance_threshold(self, tmp_path: Path) -> None:
        data = tmp_path / "data"
        _write_parquet(data / "S.parquet", id_col="filepath", ids=["target"])
        _write_parquet(data / "B.parquet", id_col="filepath", ids=["source"])
        config = tmp_path / "tmm.yaml"
        config.write_text(
            yaml.safe_dump(
                {
                    "target_parquet": "/data/S.parquet",
                    "source_parquet": "/data/B.parquet",
                    "output_parquet": "/data/out.parquet",
                    "topn": 1,
                    "knn_metric": "cosine",
                    "filter_by_label": "false",
                    "distance_threshold": 0.35,
                }
            ),
            encoding="utf-8",
        )

        assert validate_tmm_config_inputs(config, data) == 2

    @pytest.mark.parametrize(
        ("updates", "message"),
        [
            ({"source_parquet": "/data/missing.parquet"}, "not found"),
            ({"target_parquet": None}, "target_parquet"),
            ({"output_parquet": "/tmp/out.parquet"}, "output_parquet"),
            ({"output_parquet": "/data/../out.parquet"}, "under /data"),
            ({"topn": 0}, "positive integer"),
            ({"topn": True}, "positive integer"),
            ({"knn_metric": "l2"}, "cosine, euclidean, manhattan"),
            ({"filter_by_label": False}, 'string "true" or "false"'),
            ({"filter_by_label": "TRUE"}, 'string "true" or "false"'),
            ({"distance_threshold": "high"}, "finite number"),
            ({"distance_threshold": True}, "finite number"),
        ],
    )
    def test_tmm_config_rejects_unsupported_runtime_values(
        self,
        tmp_path: Path,
        updates: dict[str, object],
        message: str,
    ) -> None:
        data = tmp_path / "data"
        _write_parquet(data / "S.parquet", id_col="filepath", ids=["target"])
        _write_parquet(data / "B.parquet", id_col="filepath", ids=["source"])
        payload: dict[str, object] = {
            "target_parquet": "/data/S.parquet",
            "source_parquet": "/data/B.parquet",
            "output_parquet": "/data/out.parquet",
            "topn": 1,
            "knn_metric": "cosine",
            "filter_by_label": "false",
        }
        payload.update(updates)
        config = tmp_path / "tmm.yaml"
        config.write_text(yaml.safe_dump(payload), encoding="utf-8")

        with pytest.raises(TmmParquetError, match=message):
            validate_tmm_config_inputs(config, data)

    def test_container_data_path(self, tmp_path: Path) -> None:
        data = tmp_path / "data"
        prep = data / "_tmm_prep" / "target.parquet"
        prep.parent.mkdir(parents=True)
        prep.write_text("x", encoding="utf-8")
        assert container_data_path(prep, data) == "/data/_tmm_prep/target.parquet"

    def test_container_path_outside_data_dir_raises(self, tmp_path: Path) -> None:
        with pytest.raises(TmmParquetError, match="not under DATA_DIR"):
            container_data_path(tmp_path / "other.parquet", tmp_path / "data")

    def test_run_options_bound_outputs_and_normalize_metric(self, tmp_path: Path) -> None:
        data = tmp_path / "data"
        data.mkdir()

        output, prep, metric, threshold = validate_tmm_run_options(
            data,
            output_subdir="out",
            prep_subdir="_prep",
            topn=3,
            metric="Cosine",
            distance_threshold=0.25,
        )

        assert output == data.resolve() / "out"
        assert prep == data.resolve() / "_prep"
        assert metric == "cosine"
        assert threshold == 0.25

    @pytest.mark.parametrize(
        ("topn", "metric", "output_subdir", "prep_subdir", "message"),
        [
            (0, "cosine", "out", "_prep", "topn"),
            (3, "l2", "out", "_prep", "knn_metric"),
            (3, "cosine", "../outside", "_prep", "output"),
            (3, "cosine", "out", ".", "output"),
        ],
    )
    def test_run_options_reject_invalid_values(
        self,
        tmp_path: Path,
        topn: int,
        metric: str,
        output_subdir: str,
        prep_subdir: str,
        message: str,
    ) -> None:
        data = tmp_path / "data"
        data.mkdir()

        with pytest.raises(TmmParquetError, match=message):
            validate_tmm_run_options(
                data,
                output_subdir=output_subdir,
                prep_subdir=prep_subdir,
                topn=topn,
                metric=metric,
            )


class TestValidateEmbedColumnName:
    def test_accepts_nonempty_name(self) -> None:
        assert validate_embed_column_name("  vec  ", field="source_embed_column_name") == "vec"

    def test_rejects_empty_name(self) -> None:
        with pytest.raises(TmmParquetError, match="non-empty string"):
            validate_embed_column_name("  ", field="target_embed_column_name")
