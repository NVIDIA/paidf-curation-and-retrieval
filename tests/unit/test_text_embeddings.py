# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the TAO DS text-embedding handoff contract."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from adapters.data_mining.text_embeddings import (
    TextEmbeddingParquetError,
    validate_text_embedding_config,
    validate_text_embedding_input,
    validate_text_embedding_model,
    validate_text_embedding_output,
)
from adapters.docker_jobs import DataMiningDockerRunner
from apps.workflows import (
    RunTextEmbeddingsRequest,
    ValidateTextEmbeddingOutputRequest,
    run_text_embeddings,
    validate_text_embedding_output_handoff,
)


def _write_input(data_dir: Path, *, rows: list[dict] | None = None) -> Path:
    payload = rows or [
        {"text": "a red bus at night", "label": "bus"},
        {"text": "a pedestrian crossing", "label": "person"},
    ]
    path = data_dir / "captions.parquet"
    pd.DataFrame(payload).to_parquet(path, index=False)
    return path


def _write_output(data_dir: Path, *, rows: list[dict] | None = None) -> Path:
    payload = rows or [
        {"text": "a red bus at night", "embedding": [0.1, 0.2], "label": "bus"},
        {"text": "a pedestrian crossing", "embedding": [0.3, 0.4], "label": "person"},
    ]
    path = data_dir / "text_embeddings.parquet"
    pd.DataFrame(payload).to_parquet(path, index=False)
    return path


class TestValidateTextEmbeddingInput:
    """Input parquet must carry a usable text column."""

    def test_accepts_text_column_with_metadata(self, tmp_path: Path) -> None:
        input_path = _write_input(tmp_path)

        summary = validate_text_embedding_input(input_path, data_dir=tmp_path)

        assert summary["rows"] == 2
        assert summary["columns"] == ["text", "label"]

    def test_rejects_missing_text_column(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.parquet"
        pd.DataFrame([{"caption": "no text column"}]).to_parquet(path, index=False)

        with pytest.raises(TextEmbeddingParquetError, match="missing required columns"):
            validate_text_embedding_input(path, data_dir=tmp_path)

    def test_rejects_reserved_embedding_column(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.parquet"
        pd.DataFrame([{"text": "hi", "embedding": [0.1]}]).to_parquet(path, index=False)

        with pytest.raises(TextEmbeddingParquetError, match="reserved embedding column"):
            validate_text_embedding_input(path, data_dir=tmp_path)

    def test_rejects_empty_frame(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.parquet"
        pd.DataFrame({"text": pd.Series([], dtype="object")}).to_parquet(path, index=False)

        with pytest.raises(TextEmbeddingParquetError, match="at least one row"):
            validate_text_embedding_input(path, data_dir=tmp_path)

    def test_rejects_blank_text_row(self, tmp_path: Path) -> None:
        path = tmp_path / "blank.parquet"
        pd.DataFrame([{"text": "ok"}, {"text": "  "}]).to_parquet(path, index=False)

        with pytest.raises(TextEmbeddingParquetError, match="Text row 2"):
            validate_text_embedding_input(path, data_dir=tmp_path)

    def test_rejects_path_outside_data_dir(self, tmp_path: Path) -> None:
        data = tmp_path / "data"
        data.mkdir()
        outside = _write_input(tmp_path)

        with pytest.raises(TextEmbeddingParquetError, match="contained in DATA_DIR"):
            validate_text_embedding_input(outside, data_dir=data)


class TestValidateTextEmbeddingModel:
    """Model selection follows the TAO DS CLIP/SigLIP/SigLIP2 options."""

    @pytest.mark.parametrize(
        ("supplied", "canonical"),
        [("clip", "CLIP"), ("SigLIP", "SigLIP"), ("siglip2", "SigLIP2"), ("SIGLIP-2", "SigLIP2")],
    )
    def test_canonicalizes_supported_models(
        self,
        tmp_path: Path,
        supplied: str,
        canonical: str,
    ) -> None:
        result = validate_text_embedding_model(supplied, "hf/model-id", data_dir=tmp_path)

        assert result == {"model": canonical, "model_path": "hf/model-id"}

    def test_rejects_unsupported_model(self, tmp_path: Path) -> None:
        with pytest.raises(TextEmbeddingParquetError, match="model must be one of"):
            validate_text_embedding_model("bert", "hf/model-id", data_dir=tmp_path)

    def test_rejects_blank_model_path(self, tmp_path: Path) -> None:
        with pytest.raises(TextEmbeddingParquetError, match="non-empty string"):
            validate_text_embedding_model("clip", "  ", data_dir=tmp_path)

    def test_maps_local_model_directory_into_data(self, tmp_path: Path) -> None:
        model_dir = tmp_path / "models" / "siglip"
        model_dir.mkdir(parents=True)

        result = validate_text_embedding_model("siglip", str(model_dir), data_dir=tmp_path)

        assert result == {"model": "SigLIP", "model_path": "/data/models/siglip"}

    def test_rejects_local_model_outside_data_dir(self, tmp_path: Path) -> None:
        data = tmp_path / "data"
        data.mkdir()
        model_dir = tmp_path / "models"
        model_dir.mkdir()

        with pytest.raises(TextEmbeddingParquetError, match="contained in DATA_DIR"):
            validate_text_embedding_model("clip", str(model_dir), data_dir=data)


class TestValidateTextEmbeddingOutput:
    """Output parquet must carry finite vectors and preserve input rows."""

    def test_accepts_matching_output(self, tmp_path: Path) -> None:
        input_path = _write_input(tmp_path)
        output_path = _write_output(tmp_path)

        evidence = validate_text_embedding_output(
            output_path,
            data_dir=tmp_path,
            input_path=input_path,
        )

        assert evidence["rows"] == 2
        assert evidence["dimension"] == 2
        assert evidence["metadata_columns"] == ["label"]

    def test_accepts_output_without_input_comparison(self, tmp_path: Path) -> None:
        output_path = _write_output(tmp_path)

        evidence = validate_text_embedding_output(output_path, data_dir=tmp_path)

        assert evidence["metadata_columns"] == []

    def test_rejects_missing_embedding_column(self, tmp_path: Path) -> None:
        path = tmp_path / "out.parquet"
        pd.DataFrame([{"text": "hi"}]).to_parquet(path, index=False)

        with pytest.raises(TextEmbeddingParquetError, match="missing required columns"):
            validate_text_embedding_output(path, data_dir=tmp_path)

    def test_rejects_empty_output(self, tmp_path: Path) -> None:
        path = tmp_path / "out.parquet"
        pd.DataFrame({"text": pd.Series([], dtype="object"), "embedding": []}).to_parquet(
            path, index=False
        )

        with pytest.raises(TextEmbeddingParquetError, match="at least one row"):
            validate_text_embedding_output(path, data_dir=tmp_path)

    def test_rejects_mixed_dimensions(self, tmp_path: Path) -> None:
        output_path = _write_output(
            tmp_path,
            rows=[
                {"text": "one", "embedding": [0.1, 0.2]},
                {"text": "two", "embedding": [0.3]},
            ],
        )

        with pytest.raises(TextEmbeddingParquetError, match="mixed embedding dimensions"):
            validate_text_embedding_output(output_path, data_dir=tmp_path)

    def test_rejects_row_count_mismatch(self, tmp_path: Path) -> None:
        input_path = _write_input(tmp_path)
        output_path = _write_output(
            tmp_path,
            rows=[{"text": "a red bus at night", "embedding": [0.1, 0.2], "label": "bus"}],
        )

        with pytest.raises(TextEmbeddingParquetError, match="row count"):
            validate_text_embedding_output(
                output_path,
                data_dir=tmp_path,
                input_path=input_path,
            )

    def test_rejects_reordered_text_values(self, tmp_path: Path) -> None:
        input_path = _write_input(tmp_path)
        output_path = _write_output(
            tmp_path,
            rows=[
                {"text": "a pedestrian crossing", "embedding": [0.3, 0.4], "label": "person"},
                {"text": "a red bus at night", "embedding": [0.1, 0.2], "label": "bus"},
            ],
        )

        with pytest.raises(TextEmbeddingParquetError, match="do not match the input parquet"):
            validate_text_embedding_output(
                output_path,
                data_dir=tmp_path,
                input_path=input_path,
            )

    def test_rejects_dropped_metadata_column(self, tmp_path: Path) -> None:
        input_path = _write_input(tmp_path)
        output_path = _write_output(
            tmp_path,
            rows=[
                {"text": "a red bus at night", "embedding": [0.1, 0.2]},
                {"text": "a pedestrian crossing", "embedding": [0.3, 0.4]},
            ],
        )

        with pytest.raises(TextEmbeddingParquetError, match="preserved metadata columns"):
            validate_text_embedding_output(
                output_path,
                data_dir=tmp_path,
                input_path=input_path,
            )


class TestValidateTextEmbeddingConfig:
    """Engine-native YAML is validated before Docker execution."""

    def _config(self, tmp_path: Path, body: str) -> Path:
        path = tmp_path / "text_embeddings.yaml"
        path.write_text(body, encoding="utf-8")
        return path

    def test_accepts_minimal_config(self, tmp_path: Path) -> None:
        _write_input(tmp_path)
        config = self._config(
            tmp_path,
            "input_parquet: /data/captions.parquet\n"
            "output_parquet: /data/out.parquet\n"
            "model: SigLIP2\n"
            "model_path: google/siglip2-base-patch16-224\n",
        )

        resolved = validate_text_embedding_config(config, data_dir=tmp_path)

        assert resolved["model"] == {
            "model": "SigLIP2",
            "model_path": "google/siglip2-base-patch16-224",
        }
        assert resolved["batch_size"] == 64

    def test_rejects_missing_config_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            validate_text_embedding_config(tmp_path / "absent.yaml", data_dir=tmp_path)

    def test_rejects_non_mapping_config(self, tmp_path: Path) -> None:
        config = self._config(tmp_path, "- input_parquet: /data/captions.parquet\n")

        with pytest.raises(TextEmbeddingParquetError, match="must contain a mapping"):
            validate_text_embedding_config(config, data_dir=tmp_path)

    def test_rejects_missing_required_fields(self, tmp_path: Path) -> None:
        config = self._config(tmp_path, "input_parquet: /data/captions.parquet\n")

        with pytest.raises(TextEmbeddingParquetError, match="missing required string fields"):
            validate_text_embedding_config(config, data_dir=tmp_path)

    @pytest.mark.parametrize("batch_size", ["0", "-1", "true", "1.5"])
    def test_rejects_invalid_batch_size(self, tmp_path: Path, batch_size: str) -> None:
        _write_input(tmp_path)
        config = self._config(
            tmp_path,
            "input_parquet: /data/captions.parquet\n"
            "output_parquet: /data/out.parquet\n"
            "model: CLIP\n"
            "model_path: openai/clip-vit-base-patch32\n"
            f"batch_size: {batch_size}\n",
        )

        with pytest.raises(TextEmbeddingParquetError, match="batch_size must be a positive"):
            validate_text_embedding_config(config, data_dir=tmp_path)


class TestTextEmbeddingsDockerWiring:
    """The runner builds a generated spec against the embedding entrypoint."""

    def test_direct_dry_run_generates_spec(self, tmp_path: Path) -> None:
        input_path = _write_input(tmp_path)
        runner = DataMiningDockerRunner(image="tao-toolkit-ds:test")

        result = runner.text_embeddings(
            data_dir=str(tmp_path),
            input_parquet=str(input_path),
            output_parquet=str(tmp_path / "out.parquet"),
            model="siglip2",
            model_path="google/siglip2-base-patch16-224",
            batch_size=8,
            dry_run=True,
        )

        assert result.command[result.command.index("--entrypoint") + 1] == "embedding"
        assert result.command[-3:] == [
            "text_embeddings",
            "-e",
            "/config/generated-text-embeddings.yaml",
        ]
        assert result.evidence["experiment_spec"] == {
            "input_parquet": "/data/captions.parquet",
            "output_parquet": "/data/out.parquet",
            "model": "SigLIP2",
            "model_path": "google/siglip2-base-patch16-224",
            "batch_size": 8,
        }
        assert not Path(result.evidence["experiment_spec_host_path"]).exists()

    def test_direct_run_validates_output(self, tmp_path: Path, monkeypatch) -> None:
        input_path = _write_input(tmp_path)
        output_path = _write_output(tmp_path)
        runner = DataMiningDockerRunner(image="tao-toolkit-ds:test")
        monkeypatch.setattr(
            runner,
            "run",
            MagicMock(return_value=MagicMock(returncode=0, command=["docker"])),
        )

        result = runner.text_embeddings(
            data_dir=str(tmp_path),
            input_parquet=str(input_path),
            output_parquet=str(output_path),
            model="clip",
            model_path="openai/clip-vit-base-patch32",
        )

        assert result.evidence["dimension"] == 2
        assert result.evidence["metadata_columns"] == ["label"]

    def test_rejects_invalid_batch_size(self, tmp_path: Path) -> None:
        input_path = _write_input(tmp_path)
        runner = DataMiningDockerRunner(image="tao-toolkit-ds:test")

        with pytest.raises(ValueError, match="batch_size must be a positive integer"):
            runner.text_embeddings(
                data_dir=str(tmp_path),
                input_parquet=str(input_path),
                output_parquet=str(tmp_path / "out.parquet"),
                model="clip",
                model_path="openai/clip-vit-base-patch32",
                batch_size=0,
                dry_run=True,
            )

    def test_config_run_mounts_only_data_and_config(self, tmp_path: Path) -> None:
        _write_input(tmp_path)
        config = tmp_path / "text_embeddings.yaml"
        config.write_text(
            "input_parquet: /data/captions.parquet\n"
            "output_parquet: /data/out.parquet\n"
            "model: CLIP\n"
            "model_path: openai/clip-vit-base-patch32\n",
            encoding="utf-8",
        )
        runner = DataMiningDockerRunner(image="tao-toolkit-ds:local")

        result = runner.text_embeddings_config(
            config_file=str(config),
            data_dir=str(tmp_path),
            dry_run=True,
        )

        mounts = [
            result.command[index + 1] for index, value in enumerate(result.command) if value == "-v"
        ]
        assert mounts == [
            f"{tmp_path.resolve()}:/data",
            f"{config.resolve()}:/config/text_embeddings.yaml:ro",
        ]


class TestRunTextEmbeddingsWorkflow:
    """The workflow routes between config-file and direct modes."""

    def test_direct_mode_calls_runner(self) -> None:
        runner = MagicMock()
        runner.text_embeddings.return_value = MagicMock(command=["docker"], evidence={"rows": 2})

        result = run_text_embeddings(
            RunTextEmbeddingsRequest(
                data_dir="/data",
                input_parquet="captions.parquet",
                output_parquet="out.parquet",
                model="clip",
                model_path="openai/clip-vit-base-patch32",
                batch_size=16,
            ),
            runner_factory=lambda: runner,
        )

        assert result["status"] == "completed"
        assert result["evidence"] == {"rows": 2}
        assert runner.text_embeddings.call_args.kwargs["batch_size"] == 16

    def test_config_mode_calls_runner(self) -> None:
        runner = MagicMock()
        runner.text_embeddings_config.return_value = MagicMock(command=["docker"], evidence={})

        result = run_text_embeddings(
            RunTextEmbeddingsRequest(
                data_dir="/data",
                config_file="text_embeddings.yaml",
                dry_run=True,
            ),
            runner_factory=lambda: runner,
        )

        assert result["status"] == "dry-run"
        runner.text_embeddings_config.assert_called_once()

    def test_rejects_config_file_with_direct_options(self) -> None:
        with pytest.raises(ValueError, match="cannot be combined"):
            run_text_embeddings(
                RunTextEmbeddingsRequest(
                    data_dir="/data",
                    config_file="text_embeddings.yaml",
                    model="clip",
                ),
                runner_factory=MagicMock(),
            )

    def test_rejects_incomplete_direct_options(self) -> None:
        with pytest.raises(ValueError, match="direct mode requires"):
            run_text_embeddings(
                RunTextEmbeddingsRequest(data_dir="/data", input_parquet="captions.parquet"),
                runner_factory=MagicMock(),
            )

    def test_validate_output_handoff_reports_valid(self, tmp_path: Path) -> None:
        input_path = _write_input(tmp_path)
        output_path = _write_output(tmp_path)

        result = validate_text_embedding_output_handoff(
            ValidateTextEmbeddingOutputRequest(
                input_parquet=str(input_path),
                output_parquet=str(output_path),
                data_dir=str(tmp_path),
            ),
            validate_output=validate_text_embedding_output,
        )

        assert result["status"] == "valid"
        assert result["rows"] == 2
