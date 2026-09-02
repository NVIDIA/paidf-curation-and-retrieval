# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for docker job command construction (dry-run)."""

from __future__ import annotations

import inspect
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest
import yaml

import adapters.docker_runtime as docker_runtime
from adapters.data_mining.image_embeddings import ImageEmbeddingParquetError
from adapters.data_mining.tmm_parquet import TmmParquetError
from adapters.docker_jobs import (
    CuratorDockerRunner,
    DataMiningDockerRunner,
    DockerJobError,
)


def test_data_mining_runner_requires_explicit_nonempty_image() -> None:
    with pytest.raises(TypeError):
        DataMiningDockerRunner()  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="non-empty"):
        DataMiningDockerRunner(image=" ")


def test_docker_runtime_has_no_product_version_helpers_or_tag() -> None:
    assert not hasattr(docker_runtime, "data_mining_image_tag")
    assert not hasattr(docker_runtime, "default_data_mining_image")
    assert "7.1.0-data-services" not in inspect.getsource(docker_runtime)


class TestDockerJobs:
    @staticmethod
    def _image_input(data: Path) -> Path:
        image = data / "images/a.jpg"
        image.parent.mkdir(parents=True)
        image.write_bytes(b"image")
        input_path = data / "input.parquet"
        pd.DataFrame({"filepath": ["/data/images/a.jpg"], "label": ["car"]}).to_parquet(
            input_path,
            index=False,
        )
        return input_path

    def test_curator_dry_run_command(self, tmp_path: Path):
        cfg = tmp_path / "split.yaml"
        cfg.write_text("pipeline: split\n", encoding="utf-8")
        data = tmp_path / "data"
        data.mkdir()
        runner = CuratorDockerRunner(image="cosmos-curator:2.3.0")
        result = runner.run_pipeline(str(cfg), data_dir=str(data), dry_run=True)
        assert result.returncode == 0
        assert "docker" in result.command[0]
        assert "cosmos-curator:2.3.0" in result.command
        assert any("pipeline_config.yaml" in c for c in result.command)

    def test_curator_missing_config(self):
        runner = CuratorDockerRunner()
        with pytest.raises(FileNotFoundError):
            runner.run_pipeline("/no/such/config.yaml", dry_run=True)

    def test_data_mining_tmm_dry_run(self, tmp_path: Path):
        data = tmp_path / "data"
        data.mkdir()
        runner = DataMiningDockerRunner(image="tao-toolkit-ds:test")
        result = runner.tmm_nearest_neighbors(
            data_dir=str(data),
            dry_run=True,
            topn=3,
            filter_by_label=True,
        )
        assert result.returncode == 0
        assert "--entrypoint" in result.command
        assert "tmm" in result.command
        assert "nearest_neighbors" in result.command
        assert "tao-toolkit-ds:test" in result.command
        assert result.command[-3:] == [
            "nearest_neighbors",
            "-e",
            "/config/generated-tmm.yaml",
        ]
        assert all("=" not in argument for argument in result.command[-3:])
        assert result.evidence["experiment_spec"] == {
            "source_parquet": "/data/_tmm_prep/source.parquet",
            "target_parquet": "/data/_tmm_prep/target.parquet",
            "output_parquet": "/data/divknn_out/mined.parquet",
            "topn": 3,
            "knn_metric": "cosine",
            "filter_by_label": "true",
            "distance_threshold": -1.0,
            "source_embed_column_name": "embedding",
            "target_embed_column_name": "embedding",
        }
        assert (
            yaml.safe_load(result.evidence["experiment_spec_yaml"])
            == result.evidence["experiment_spec"]
        )
        mount = result.command[result.command.index("-v", 6) + 1]
        assert mount.endswith(":/config/generated-tmm.yaml:ro")
        assert not Path(result.evidence["experiment_spec_host_path"]).exists()
        assert result.evidence["dry_run_replayable"] is False
        assert "--gpus=all" in result.command
        assert "--shm-size=16g" in result.command

    def test_data_mining_gpus_and_shm_overrides(self, tmp_path: Path):
        data = tmp_path / "data"
        data.mkdir()
        runner = DataMiningDockerRunner(
            image="tao-toolkit-ds:local",
            gpus='"device=0"',
            shm_size="32g",
        )
        result = runner.tmm_nearest_neighbors(data_dir=str(data), dry_run=True)
        assert '--gpus="device=0"' in result.command
        assert "--shm-size=32g" in result.command

    def test_divknn_alias_uses_tmm(self, tmp_path: Path):
        data = tmp_path / "data"
        data.mkdir()
        runner = DataMiningDockerRunner(image="tao-toolkit-ds:local")
        result = runner.divknn(data_dir=str(data), dry_run=True, topn=2)
        assert "tmm" in result.command
        assert "nearest_neighbors" in result.command
        assert "divknn" not in result.command

    def test_tmm_nearest_neighbors_config_dry_run(self, tmp_path: Path):
        data = tmp_path / "data"
        data.mkdir()
        pd.DataFrame({"filepath": ["target"], "embedding": [[1.0, 0.0]]}).to_parquet(
            data / "S.parquet",
            index=False,
        )
        pd.DataFrame({"filepath": ["source"], "embedding": [[0.0, 1.0]]}).to_parquet(
            data / "B.parquet",
            index=False,
        )
        cfg = tmp_path / "tmm.yaml"
        cfg.write_text(
            "target_parquet: /data/S.parquet\n"
            "source_parquet: /data/B.parquet\n"
            "output_parquet: /data/out.parquet\n"
            "topn: 5\n"
            "knn_metric: cosine\n"
            'filter_by_label: "false"\n',
            encoding="utf-8",
        )
        runner = DataMiningDockerRunner(image="tao-toolkit-ds:local")
        result = runner.tmm_nearest_neighbors_config(
            config_file=str(cfg),
            data_dir=str(data),
            dry_run=True,
        )
        assert result.returncode == 0
        assert any("/config/tmm.yaml:ro" in c for c in result.command)
        assert "-e" in result.command
        assert "/config/tmm.yaml" in result.command

    @pytest.mark.parametrize(
        ("topn", "metric", "output_subdir", "prep_subdir", "message"),
        [
            (0, "cosine", "divknn_out", "_tmm_prep", "topn"),
            (5, "l2", "divknn_out", "_tmm_prep", "knn_metric"),
            (5, "cosine", "../outside", "_tmm_prep", "output"),
            (5, "cosine", "divknn_out", ".", "output"),
        ],
    )
    def test_tmm_nearest_neighbors_rejects_invalid_run_options(
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
        runner = DataMiningDockerRunner(image="tao-toolkit-ds:local")

        with pytest.raises(TmmParquetError, match=message):
            runner.tmm_nearest_neighbors(
                data_dir=str(data),
                output_subdir=output_subdir,
                prep_subdir=prep_subdir,
                topn=topn,
                metric=metric,
                dry_run=True,
            )

    def test_tmm_nearest_neighbors_config_missing_file(self, tmp_path: Path):
        data = tmp_path / "data"
        data.mkdir()
        runner = DataMiningDockerRunner(image="tao-toolkit-ds:local")
        with pytest.raises(FileNotFoundError, match="TMM_CONFIG_FILE"):
            runner.tmm_nearest_neighbors_config(
                config_file="/no/such/tmm.yaml",
                data_dir=str(data),
                dry_run=True,
            )

    def test_tmm_nearest_neighbors_requires_boolean_filter(self, tmp_path: Path) -> None:
        runner = DataMiningDockerRunner(image="tao-toolkit-ds:test")

        with pytest.raises(ValueError, match="filter_by_label"):
            runner.tmm_nearest_neighbors(
                data_dir=str(tmp_path),
                filter_by_label="false",  # type: ignore[arg-type]
                dry_run=True,
            )

    def test_tmm_config_validates_inputs_before_container_execution(
        self,
        tmp_path: Path,
        monkeypatch,
    ):
        data = tmp_path / "data"
        data.mkdir()
        cfg = tmp_path / "tmm.yaml"
        cfg.write_text(
            "source_parquet: /data/B.parquet\ntarget_parquet: /data/S.parquet\n",
            encoding="utf-8",
        )
        validate = MagicMock(return_value=4)
        monkeypatch.setattr(
            "adapters.data_mining.tmm_parquet.validate_tmm_config_inputs",
            validate,
        )
        runner = DataMiningDockerRunner(image="tao-toolkit-ds:local")
        run = MagicMock(return_value=MagicMock(returncode=0, command=["docker"]))
        monkeypatch.setattr(runner, "run", run)

        runner.tmm_nearest_neighbors_config(
            config_file=str(cfg),
            data_dir=str(data),
            dry_run=False,
        )

        validate.assert_called_once_with(str(cfg), str(data))
        run.assert_called_once()

    def test_image_embeddings_direct_dry_run_uses_embedding_entrypoint(
        self,
        tmp_path: Path,
    ) -> None:
        data = tmp_path / "data"
        data.mkdir()
        input_path = self._image_input(data)
        runner = DataMiningDockerRunner(image="tao-toolkit-ds:test")

        result = runner.image_embeddings(
            data_dir=str(data),
            input_parquet=str(input_path),
            output_parquet=str(data / "output.parquet"),
            model_type="clip",
            model_name_or_path="openai/clip-vit-base-patch32;not-a-shell",
            batch_size=8,
            dry_run=True,
        )

        assert result.command[result.command.index("--entrypoint") + 1] == "embedding"
        assert "image_embeddings" in result.command
        assert result.command[-3:] == [
            "image_embeddings",
            "-e",
            "/config/generated-image-embeddings.yaml",
        ]
        assert result.evidence["experiment_spec"] == {
            "input_parquet": "/data/input.parquet",
            "output_parquet": "/data/output.parquet",
            "model": "CLIP",
            "model_path": "openai/clip-vit-base-patch32;not-a-shell",
            "batch_size": 8,
        }
        assert any(
            mount.endswith(":/config/generated-image-embeddings.yaml:ro")
            for index, mount in enumerate(result.command)
            if result.command[index - 1] == "-v"
        )
        assert not Path(result.evidence["experiment_spec_host_path"]).exists()
        assert "bash" not in result.command
        assert "sh" not in result.command

    def test_image_embeddings_config_mounts_only_data_and_config(self, tmp_path: Path) -> None:
        data = tmp_path / "data"
        data.mkdir()
        self._image_input(data)
        config = tmp_path / "image_embeddings.yaml"
        config.write_text(
            "input_parquet: /data/input.parquet\n"
            "output_parquet: /data/output.parquet\n"
            "model: SigLIP\n"
            "model_path: google/siglip-base-patch16-224\n",
            encoding="utf-8",
        )
        runner = DataMiningDockerRunner(image="tao-toolkit-ds:local")

        result = runner.image_embeddings_config(
            config_file=str(config),
            data_dir=str(data),
            dry_run=True,
        )

        mounts = [
            result.command[index + 1] for index, value in enumerate(result.command) if value == "-v"
        ]
        assert mounts == [
            f"{data.resolve()}:/data",
            f"{config.resolve()}:/config/image_embeddings.yaml:ro",
        ]
        assert result.command[-3:] == [
            "image_embeddings",
            "-e",
            "/config/image_embeddings.yaml",
        ]

    def test_generated_image_spec_includes_optional_model_config(self, tmp_path: Path) -> None:
        data = tmp_path / "data"
        data.mkdir()
        input_path = self._image_input(data)
        checkpoint = data / "model.pth"
        checkpoint.write_bytes(b"weights")
        model_config = data / "model.yaml"
        model_config.write_text("model: clip\n", encoding="utf-8")
        runner = DataMiningDockerRunner(image="tao-toolkit-ds:test")

        result = runner.image_embeddings(
            data_dir=str(data),
            input_parquet=str(input_path),
            output_parquet=str(data / "output.parquet"),
            model_type="clip",
            model_name_or_path=str(checkpoint),
            model_config_path=str(model_config),
            batch_size=1,
            dry_run=True,
        )

        assert result.evidence["experiment_spec"] == {
            "input_parquet": "/data/input.parquet",
            "output_parquet": "/data/output.parquet",
            "model": "CLIP",
            "model_path": "/data/model.pth",
            "batch_size": 1,
            "model_config_path": "/data/model.yaml",
        }

    @pytest.mark.parametrize("fails", [False, True])
    def test_generated_image_spec_is_cleaned_after_docker_call(
        self,
        tmp_path: Path,
        monkeypatch,
        fails: bool,
    ) -> None:
        data = tmp_path / "data"
        data.mkdir()
        input_path = self._image_input(data)
        runner = DataMiningDockerRunner(image="tao-toolkit-ds:test")
        observed_path: Path | None = None

        def fake_run(*args, **kwargs):
            nonlocal observed_path
            mount = kwargs["extra_mounts"][0]
            observed_path = Path(mount.split(":", maxsplit=1)[0])
            assert observed_path.is_file()
            assert yaml.safe_load(observed_path.read_text(encoding="utf-8"))["model"] == "CLIP"
            if fails:
                raise DockerJobError("expected failure")
            return MagicMock(returncode=0, command=["docker"], evidence={})

        monkeypatch.setattr(runner, "run", fake_run)
        monkeypatch.setattr(
            "adapters.data_mining.image_embeddings.validate_image_embedding_output",
            MagicMock(return_value={"rows": 1}),
        )

        def call():
            return runner.image_embeddings(
                data_dir=str(data),
                input_parquet=str(input_path),
                output_parquet=str(data / "output.parquet"),
                model_type="clip",
                model_name_or_path="openai/clip-vit-base-patch32",
            )

        if fails:
            with pytest.raises(DockerJobError, match="expected failure"):
                call()
        else:
            assert call().evidence["rows"] == 1

        assert observed_path is not None
        assert not observed_path.exists()

    def test_image_embeddings_fails_when_success_has_no_output(
        self,
        tmp_path: Path,
        monkeypatch,
    ) -> None:
        data = tmp_path / "data"
        data.mkdir()
        input_path = self._image_input(data)
        runner = DataMiningDockerRunner(image="tao-toolkit-ds:local")
        monkeypatch.setattr(
            runner,
            "run",
            MagicMock(return_value=MagicMock(returncode=0, command=["docker"], evidence={})),
        )

        with pytest.raises(ImageEmbeddingParquetError, match="not found"):
            runner.image_embeddings(
                data_dir=str(data),
                input_parquet=str(input_path),
                output_parquet=str(data / "missing.parquet"),
                model_type="clip",
                model_name_or_path="openai/clip-vit-base-patch32",
            )

    def test_docker_subprocess_uses_argument_list_without_shell(
        self,
        tmp_path: Path,
        monkeypatch,
    ) -> None:
        completed = MagicMock(returncode=0)
        run = MagicMock(return_value=completed)
        monkeypatch.setattr("adapters.docker_runtime.subprocess.run", run)
        runner = DataMiningDockerRunner(image="tao-toolkit-ds:local")

        runner.run(
            "image_embeddings",
            ["model_path=value;touch /tmp/not-run"],
            data_dir=str(tmp_path),
        )

        command = run.call_args.args[0]
        assert isinstance(command, list)
        assert "value;touch /tmp/not-run" in command[-1]
        assert run.call_args.kwargs == {"check": False, "shell": False}
