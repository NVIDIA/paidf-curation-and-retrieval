# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLI contract tests for external integration consumers."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest
from click.testing import CliRunner

from adapters.dataset_search.retrieval_adapter import BulkJobPollingTimeout
from apps.cli.__main__ import main
from apps.workflows import (
    BuildCaptionParquetRequest,
    BulkInsertCaptionParquetsRequest,
    CaptionSearchRequest,
    IngestCuratorExportRequest,
    PrepareCdsCe1ForTdmRequest,
    RunDataMiningSelectionRequest,
    RunImageEmbeddingsRequest,
    StageArtifactRequest,
    UploadCaptionParquetRequest,
    build_caption_parquet_handoff,
    bulk_insert_caption_parquets,
    ingest_curator_export,
    prepare_cds_ce1_for_tdm_handoff,
    run_data_mining_selection,
    run_image_embeddings,
    search_captions,
    stage_artifact_handoff,
    upload_caption_parquet_handoff,
)
from packages.domain.types import BulkJobStatus


def test_ingest_curator_rejects_iv2_for_cds_bound_ingest(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        main,
        [
            "ingest-curator",
            "--curator-dir",
            str(tmp_path),
            "--output-parquet",
            str(tmp_path / "out.parquet"),
            "--collection",
            "demo",
            "--embedding-backend",
            "iv2",
            "--ingest",
        ],
    )

    assert result.exit_code != 0
    assert "requires the CE1" in result.output


def test_ingest_curator_auto_selects_ce1_and_keeps_fallback_disabled(
    tmp_path: Path,
    monkeypatch,
) -> None:
    output = tmp_path / "out.parquet"
    convert = MagicMock(return_value=str(output))
    client = MagicMock()
    client.ingest_parquet.return_value = "job-1"
    monkeypatch.setattr(
        "apps.cli.__main__.CuratorExportAdapter.to_cds_parquet",
        convert,
    )
    monkeypatch.setattr("apps.cli.__main__.build_dataset_search_adapter", lambda _: client)

    result = CliRunner().invoke(
        main,
        [
            "ingest-curator",
            "--curator-dir",
            str(tmp_path),
            "--output-parquet",
            str(output),
            "--collection",
            "demo",
            "--embedding-backend",
            "auto",
            "--ingest",
            "--cds-url",
            "https://cds.example",
        ],
    )

    assert result.exit_code == 0
    assert json.loads(result.output.splitlines()[0])["embedding_backend"] == "ce1"
    assert convert.call_args.kwargs["backend"] == "ce1"
    assert client.ingest_parquet.call_args.kwargs["embedding_family"] == "ce1"
    assert client.ingest_parquet.call_args.kwargs["allow_document_fallback"] is False


def test_ingest_curator_workflow_rejects_iv2_ingest_before_conversion() -> None:
    converter_factory = MagicMock()

    with pytest.raises(ValueError, match="requires the CE1"):
        ingest_curator_export(
            IngestCuratorExportRequest(
                curator_dir="curator",
                output_parquet="out.parquet",
                collection="demo",
                embedding_backend="iv2",
                convert_only=False,
            ),
            converter_factory=converter_factory,
        )

    converter_factory.assert_not_called()


def test_ingest_curator_workflow_converts_and_ingests() -> None:
    converter = MagicMock()
    converter.to_cds_parquet.return_value = "out.parquet"
    client = MagicMock()
    client.ingest_parquet.return_value = "job-1"

    result = ingest_curator_export(
        IngestCuratorExportRequest(
            curator_dir="curator",
            output_parquet="out.parquet",
            collection="demo",
            embedding_backend="auto",
            convert_only=False,
            allow_lab_document_fallback=True,
        ),
        converter_factory=lambda: converter,
        client_factory=lambda: client,
    )

    converter.to_cds_parquet.assert_called_once_with(
        "curator",
        "out.parquet",
        backend="ce1",
    )
    assert client.ingest_parquet.call_args.args[0].collection_id == "demo"
    assert client.ingest_parquet.call_args.args[1] == [str(Path("out.parquet").resolve())]
    assert client.ingest_parquet.call_args.kwargs["embedding_family"] == "ce1"
    assert client.ingest_parquet.call_args.kwargs["allow_document_fallback"] is True
    assert result.conversion_payload == {
        "cds_parquet": "out.parquet",
        "embedding_backend": "ce1",
    }
    assert result.ingest_payload == {
        "ingested": True,
        "collection": "demo",
        "job_id": "job-1",
        "embedding_family": "ce1",
    }


def test_bulk_insert_cli_resolves_credential_references(monkeypatch) -> None:
    client = MagicMock()
    client.bulk_insert.return_value = BulkJobStatus(job_id="job-1", status="accepted")
    monkeypatch.setattr("apps.cli.dataset_search_cmds._client", lambda _: client)

    result = CliRunner().invoke(
        main,
        [
            "ds",
            "bulk-insert",
            "--collection",
            "demo",
            "--embedding-family",
            "ce1",
            "--parquet",
            "s3://bucket/data.parquet",
            "--access-key-env",
            "TEST_ACCESS",
            "--secret-key-env",
            "TEST_SECRET",
            "--cds-url",
            "https://cds.example",
        ],
        env={"TEST_ACCESS": "access", "TEST_SECRET": "secret"},
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["job_id"] == "job-1"
    assert payload["embedding_family"] == "ce1"
    request = client.bulk_insert.call_args.args[0]
    assert request.access_key == "access"
    assert request.secret_key == "secret"


def test_train_refinement_cli_wraps_unreadable_spec(tmp_path: Path, monkeypatch) -> None:
    client = MagicMock()
    monkeypatch.setattr("apps.cli.dataset_search_cmds._client", lambda _: client)

    result = CliRunner().invoke(
        main,
        [
            "ds",
            "train-refinement",
            "--spec-file",
            str(tmp_path),
        ],
    )

    assert result.exit_code != 0
    assert "Error:" in result.output
    client.train_search_refinement.assert_not_called()


def test_bulk_insert_cli_exposes_no_direct_secret_option() -> None:
    result = CliRunner().invoke(main, ["ds", "bulk-insert", "--help"])

    assert result.exit_code == 0
    assert "--secret-key-env" in result.output
    assert "--secret-key " not in result.output


def test_bulk_insert_cli_rejects_missing_or_incompatible_family_without_client(
    monkeypatch,
) -> None:
    client_factory = MagicMock()
    monkeypatch.setattr("apps.cli.dataset_search_cmds._client", client_factory)

    for family_args in ([], ["--embedding-family", "iv2"]):
        result = CliRunner().invoke(
            main,
            [
                "ds",
                "bulk-insert",
                "--collection",
                "demo",
                "--parquet",
                "s3://bucket/data.parquet",
                *family_args,
            ],
        )
        assert result.exit_code != 0

    client_factory.assert_not_called()


def test_job_status_wait_emits_timeout_json_and_nonzero(monkeypatch) -> None:
    client = MagicMock()
    client.wait_for_job.side_effect = BulkJobPollingTimeout("deadline reached")
    monkeypatch.setattr("apps.cli.dataset_search_cmds._client", lambda _: client)

    result = CliRunner().invoke(
        main,
        [
            "ds",
            "job-status",
            "job-1",
            "--wait",
            "--timeout-seconds",
            "0",
        ],
    )

    assert result.exit_code == 2
    assert json.loads(result.output)["status"] == "timeout"


def test_job_status_wait_emits_failed_terminal_json_and_nonzero(monkeypatch) -> None:
    client = MagicMock()
    client.wait_for_job.return_value = BulkJobStatus(job_id="job-1", status="failed")
    monkeypatch.setattr("apps.cli.dataset_search_cmds._client", lambda _: client)

    result = CliRunner().invoke(main, ["ds", "job-status", "job-1", "--wait"])

    assert result.exit_code == 2
    assert json.loads(result.output)["status"] == "failed"


def test_data_mining_cli_emits_backend_and_command_json(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runner = MagicMock()
    runner.tmm_nearest_neighbors.return_value = SimpleNamespace(command=["docker", "run"])
    factory = MagicMock(return_value=runner)
    monkeypatch.setattr("apps.cli.__main__.build_data_mining_runner", factory)

    result = CliRunner().invoke(
        main,
        [
            "data-mining-select",
            "--data-dir",
            str(tmp_path),
            "--embedding-backend",
            "ce1",
            "--image",
            "tao-toolkit-ds:test",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["status"] == "dry-run"
    assert payload["embedding_backend"] == "ce1"
    assert payload["dry_run"] is True
    assert payload["evidence"] == {}
    factory.assert_called_once_with(image="tao-toolkit-ds:test", gpus="all", shm_size="16g")


def test_data_mining_cli_defaults_to_ce1_embedding_backend(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runner = MagicMock()
    runner.tmm_nearest_neighbors.return_value = SimpleNamespace(command=["docker", "run"])
    monkeypatch.setattr("apps.cli.__main__.build_data_mining_runner", lambda **kwargs: runner)

    result = CliRunner().invoke(
        main,
        [
            "data-mining-select",
            "--data-dir",
            str(tmp_path),
            "--image",
            "tao-toolkit-ds:test",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0
    assert json.loads(result.output)["embedding_backend"] == "ce1"


def test_data_mining_unique_match_cli_defaults_to_ce1_embedding_backend(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runner = MagicMock()
    runner.tmm_unique_neighbor_matching.return_value = SimpleNamespace(command=["docker", "run"])
    monkeypatch.setattr("apps.cli.__main__.build_data_mining_runner", lambda **kwargs: runner)

    result = CliRunner().invoke(
        main,
        [
            "data-mining-unique-match",
            "--data-dir",
            str(tmp_path),
            "--image",
            "tao-toolkit-ds:test",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0
    assert json.loads(result.output)["embedding_backend"] == "ce1"


def test_data_mining_cli_requires_explicit_image(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        main,
        ["data-mining-select", "--data-dir", str(tmp_path), "--dry-run"],
    )

    assert result.exit_code != 0
    assert "Missing option '--image'" in result.output


@pytest.mark.parametrize("embedding_backend", ["clip", "siglip"])
def test_data_mining_cli_accepts_tao_image_embedding_backend(
    tmp_path: Path,
    monkeypatch,
    embedding_backend: str,
) -> None:
    runner = MagicMock()
    runner.tmm_nearest_neighbors.return_value = SimpleNamespace(command=["docker", "run"])
    monkeypatch.setattr("apps.cli.__main__.build_data_mining_runner", lambda **kwargs: runner)

    result = CliRunner().invoke(
        main,
        [
            "data-mining-select",
            "--data-dir",
            str(tmp_path),
            "--embedding-backend",
            embedding_backend,
            "--image",
            "tao-toolkit-ds:test",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0
    assert json.loads(result.output)["embedding_backend"] == embedding_backend


def test_data_mining_cli_rejects_unknown_embedding_backend(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        main,
        [
            "data-mining-select",
            "--data-dir",
            str(tmp_path),
            "--embedding-backend",
            "unknown",
            "--image",
            "tao-toolkit-ds:test",
            "--dry-run",
        ],
    )

    assert result.exit_code != 0
    assert "Invalid value" in result.output


def test_data_mining_cli_rejects_nonpositive_topn(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        main,
        [
            "data-mining-select",
            "--data-dir",
            str(tmp_path),
            "--topn",
            "0",
            "--image",
            "tao-toolkit-ds:test",
            "--dry-run",
        ],
    )

    assert result.exit_code != 0
    assert "Invalid value" in result.output


def test_data_mining_workflow_routes_config_request() -> None:
    runner = MagicMock()
    runner.tmm_nearest_neighbors_config.return_value = SimpleNamespace(command=("docker", "run"))

    result = run_data_mining_selection(
        RunDataMiningSelectionRequest(
            data_dir="data",
            config_file="tmm.yaml",
            embedding_backend="CE1",
            dry_run=True,
        ),
        runner_factory=lambda: runner,
    )

    runner.tmm_nearest_neighbors_config.assert_called_once_with(
        config_file="tmm.yaml",
        data_dir="data",
        dry_run=True,
    )
    runner.tmm_nearest_neighbors.assert_not_called()
    assert result.embedding_backend == "ce1"
    assert result.command == ["docker", "run"]


def test_data_mining_workflow_rejects_in_process_before_runner_construction() -> None:
    runner_factory = MagicMock()

    with pytest.raises(ValueError, match="in-process mode"):
        run_data_mining_selection(
            RunDataMiningSelectionRequest(data_dir="data", in_process=True),
            runner_factory=runner_factory,
        )

    runner_factory.assert_not_called()


def test_caption_build_cli_emits_machine_readable_summary(tmp_path: Path) -> None:
    source = tmp_path / "captions.json"
    source.write_text(
        json.dumps([{"clip_id": "clip-1", "summary": "A vehicle turns."}]),
        encoding="utf-8",
    )
    output = tmp_path / "captions.parquet"

    result = CliRunner().invoke(
        main,
        [
            "integration",
            "captions",
            "build",
            "--input-json",
            str(source),
            "--output-parquet",
            str(output),
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["status"] == "built"
    assert payload["rows"] == 1
    assert output.is_file()


def test_caption_build_workflow_rejects_non_array_json_before_build(tmp_path: Path) -> None:
    source = tmp_path / "captions.json"
    source.write_text("{}", encoding="utf-8")
    build_caption = MagicMock()

    with pytest.raises(ValueError, match="JSON array"):
        build_caption_parquet_handoff(
            BuildCaptionParquetRequest(
                input_json=str(source),
                output_parquet=str(tmp_path / "captions.parquet"),
            ),
            build_caption=build_caption,
            validate_caption=MagicMock(),
        )

    build_caption.assert_not_called()


def test_caption_build_workflow_preserves_indexed_ids_for_validation(tmp_path: Path) -> None:
    source = tmp_path / "captions.json"
    source.write_text(
        json.dumps([{"clip_id": "clip-1", "summary": "A vehicle turns."}]),
        encoding="utf-8",
    )
    output = tmp_path / "captions.parquet"
    indexed_clip_ids = {"clip-1"}
    build_caption = MagicMock(return_value=output)
    validate_caption = MagicMock(return_value={"rows": 1, "identity_aligned": True})

    result = build_caption_parquet_handoff(
        BuildCaptionParquetRequest(
            input_json=str(source),
            output_parquet=str(output),
            indexed_clip_ids=indexed_clip_ids,
        ),
        build_caption=build_caption,
        validate_caption=validate_caption,
    )

    build_caption.assert_called_once_with(
        [{"clip_id": "clip-1", "summary": "A vehicle turns."}],
        str(output),
        indexed_clip_ids=indexed_clip_ids,
    )
    validate_caption.assert_called_once_with(output, indexed_clip_ids=indexed_clip_ids)
    assert result == {"status": "built", "rows": 1, "identity_aligned": True}


def test_prepare_cds_ce1_for_tdm_cli_emits_artifact_manifest(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    pd.DataFrame({"id": ["target"], "embedding": [[1.0, 2.0]]}).to_parquet(
        data / "target.parquet",
        index=False,
    )
    pd.DataFrame({"id": ["source"], "embedding": [[3.0, 4.0]]}).to_parquet(
        data / "source.parquet",
        index=False,
    )

    result = CliRunner().invoke(
        main,
        [
            "integration",
            "prepare-cds-ce1-for-tdm",
            "--data-dir",
            str(data),
            "--target-selection",
            "target.parquet",
            "--source-selection",
            "source.parquet",
            "--embedding-family",
            "ce1",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["status"] == "prepared"
    assert payload["embedding_family"] == "ce1"
    assert payload["dimension"] == 2
    assert Path(payload["target_parquet"]).is_file()
    assert Path(payload["source_parquet"]).is_file()
    assert payload["target_container_path"] == "/data/_tmm_prep/target.parquet"


def test_prepare_cds_ce1_for_tdm_rejects_dimension_mismatch_before_output(
    tmp_path: Path,
) -> None:
    pd.DataFrame({"id": ["target"], "embedding": [[1.0, 2.0]]}).to_parquet(
        tmp_path / "target.parquet",
        index=False,
    )
    pd.DataFrame({"id": ["source"], "embedding": [[3.0, 4.0, 5.0]]}).to_parquet(
        tmp_path / "source.parquet",
        index=False,
    )

    result = CliRunner().invoke(
        main,
        [
            "integration",
            "prepare-cds-ce1-for-tdm",
            "--data-dir",
            str(tmp_path),
            "--target-selection",
            "target.parquet",
            "--source-selection",
            "source.parquet",
            "--embedding-family",
            "ce1",
        ],
    )

    assert result.exit_code == 2
    assert json.loads(result.output)["status"] == "error"
    assert not (tmp_path / "_tmm_prep" / "target.parquet").exists()
    assert not (tmp_path / "_tmm_prep" / "source.parquet").exists()


def test_prepare_cds_ce1_for_tdm_rejects_incompatible_family(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        main,
        [
            "integration",
            "prepare-cds-ce1-for-tdm",
            "--data-dir",
            str(tmp_path),
            "--target-selection",
            "target.parquet",
            "--source-selection",
            "source.parquet",
            "--embedding-family",
            "iv2",
        ],
    )

    assert result.exit_code != 0
    assert not (tmp_path / "_tmm_prep").exists()


def test_caption_search_cli_emits_machine_readable_results(monkeypatch) -> None:
    client = MagicMock()
    client.search.return_value = {"clip_ids": ["clip-1"], "count": 1}
    monkeypatch.setattr("apps.cli.integration_cmds.build_caption_adapter", lambda _: client)

    result = CliRunner().invoke(
        main,
        [
            "integration",
            "captions",
            "search",
            "--query",
            "pedestrian",
            "--limit",
            "25",
            "--data-source",
            "traffic",
            "--cds-url",
            "https://cds.example",
        ],
    )

    assert result.exit_code == 0
    assert json.loads(result.output) == {
        "status": "ok",
        "clip_ids": ["clip-1"],
        "count": 1,
    }
    assert client.search.call_args.args == ("pedestrian",)
    assert client.search.call_args.kwargs["limit"] == 25
    assert client.search.call_args.kwargs["data_sources"] == ("traffic",)


def test_caption_search_cli_invalid_limit_is_json(monkeypatch) -> None:
    client = MagicMock()
    client.search.side_effect = ValueError("caption search limit must be between 1 and 50000")
    monkeypatch.setattr("apps.cli.integration_cmds.build_caption_adapter", lambda _: client)

    result = CliRunner().invoke(
        main,
        [
            "integration",
            "captions",
            "search",
            "--query",
            "pedestrian",
            "--limit",
            "0",
            "--cds-url",
            "https://cds.example",
        ],
    )

    assert result.exit_code == 2
    assert json.loads(result.output)["status"] == "error"


def test_caption_search_workflow_wraps_result_status() -> None:
    client = MagicMock()
    client.search.return_value = {"clip_ids": ["clip-1"], "count": 1}

    result = search_captions(
        CaptionSearchRequest(query="pedestrian", limit=25, data_sources=("traffic",)),
        client_factory=lambda: client,
    )

    client.search.assert_called_once_with(
        "pedestrian",
        limit=25,
        data_sources=("traffic",),
    )
    assert result == {"status": "ok", "clip_ids": ["clip-1"], "count": 1}


def test_stage_artifact_cli_emits_minimal_handoff_manifest(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "artifact.parquet"
    source.write_bytes(b"parquet")
    monkeypatch.setattr(
        "apps.cli.integration_cmds.S3ObjectStoreStager.stage",
        lambda self, source, destination: destination,
    )

    result = CliRunner().invoke(
        main,
        [
            "integration",
            "stage-artifact",
            "--source",
            str(source),
            "--destination",
            "s3://bucket/artifact.parquet",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload == {
        "status": "staged",
        "input": str(source),
        "output": "s3://bucket/artifact.parquet",
    }


def test_stage_artifact_workflow_returns_handoff_manifest() -> None:
    stager = MagicMock()
    stager.stage.return_value = "s3://bucket/artifact.parquet"

    result = stage_artifact_handoff(
        StageArtifactRequest(source="artifact.parquet", destination="s3://bucket/artifact.parquet"),
        stager_factory=lambda: stager,
    )

    stager.stage.assert_called_once_with("artifact.parquet", "s3://bucket/artifact.parquet")
    assert result == {
        "status": "staged",
        "input": "artifact.parquet",
        "output": "s3://bucket/artifact.parquet",
    }


def test_caption_upload_workflow_validates_before_upload() -> None:
    validate_caption = MagicMock()
    client = MagicMock()
    client.upload_parquet.return_value = {"job_id": "job-1"}

    result = upload_caption_parquet_handoff(
        UploadCaptionParquetRequest(
            parquet_path="captions.parquet",
            model_name="default",
            data_source="traffic",
            indexed_clip_ids={"clip-1"},
        ),
        validate_caption=validate_caption,
        client_factory=lambda: client,
    )

    validate_caption.assert_called_once_with(
        "captions.parquet",
        indexed_clip_ids={"clip-1"},
    )
    client.upload_parquet.assert_called_once_with(
        "captions.parquet",
        model_name="default",
        data_source="traffic",
    )
    assert result == {"status": "submitted", "result": {"job_id": "job-1"}}


def test_caption_bulk_insert_workflow_passes_credentials() -> None:
    client = MagicMock()
    client.bulk_insert.return_value = {"accepted": 1}

    result = bulk_insert_caption_parquets(
        BulkInsertCaptionParquetsRequest(
            parquet_paths=("s3://bucket/captions.parquet",),
            access_key="access",
            secret_key="secret",
            endpoint_url="https://s3.example",
            allow_lab_http_endpoint=True,
            model_name_override="model",
            data_source_override="traffic",
        ),
        client_factory=lambda: client,
    )

    client.bulk_insert.assert_called_once_with(
        ("s3://bucket/captions.parquet",),
        access_key="access",
        secret_key="secret",
        endpoint_url="https://s3.example",
        allow_insecure_endpoint=True,
        model_name_override="model",
        data_source_override="traffic",
    )
    assert result == {"status": "submitted", "result": {"accepted": 1}}


def test_image_embedding_build_cli_emits_json_and_preserves_metadata(tmp_path: Path) -> None:
    data = tmp_path / "data"
    image = data / "images/a.jpg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"image")
    source = data / "rows.json"
    source.write_text(
        json.dumps([{"filepath": "images/a.jpg", "label": "car"}]),
        encoding="utf-8",
    )
    output = data / "input.parquet"

    result = CliRunner().invoke(
        main,
        [
            "integration",
            "image-embeddings",
            "build",
            "--input-json",
            str(source),
            "--data-dir",
            str(data),
            "--output-parquet",
            str(output),
        ],
    )

    assert result.exit_code == 0
    assert json.loads(result.output)["status"] == "built"
    assert pd.read_parquet(output).to_dict("records") == [
        {"filepath": "/data/images/a.jpg", "label": "car"}
    ]


def test_image_embedding_build_cli_emits_json_error(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    source = data / "rows.json"
    source.write_text("not-json", encoding="utf-8")

    result = CliRunner().invoke(
        main,
        [
            "integration",
            "image-embeddings",
            "build",
            "--input-json",
            str(source),
            "--data-dir",
            str(data),
            "--output-parquet",
            str(data / "input.parquet"),
        ],
    )

    assert result.exit_code == 2
    assert json.loads(result.output)["status"] == "error"


def test_image_embedding_run_cli_emits_command_json(
    tmp_path: Path,
    monkeypatch,
) -> None:
    data = tmp_path / "data"
    data.mkdir()
    input_path = data / "input.parquet"
    input_path.write_bytes(b"parquet")
    docker_runner = MagicMock()
    docker_runner.image_embeddings.return_value = SimpleNamespace(
        command=["docker", "run", "--entrypoint", "embedding"],
        evidence={},
    )
    monkeypatch.setattr(
        "apps.cli.integration_cmds.build_data_mining_runner",
        lambda **kwargs: docker_runner,
    )

    result = CliRunner().invoke(
        main,
        [
            "integration",
            "image-embeddings",
            "run",
            "--data-dir",
            str(data),
            "--input-parquet",
            str(input_path),
            "--output-parquet",
            str(data / "output.parquet"),
            "--model-type",
            "clip",
            "--model-name-or-path",
            "openai/clip-vit-base-patch32",
            "--image",
            "tao-toolkit-ds:test",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["status"] == "dry-run"
    assert payload["command"][-1] == "embedding"
    assert docker_runner.image_embeddings.call_args.kwargs["model_type"] == "clip"


def test_image_embedding_run_cli_requires_explicit_image(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        main,
        [
            "integration",
            "image-embeddings",
            "run",
            "--data-dir",
            str(tmp_path),
            "--dry-run",
        ],
    )

    assert result.exit_code != 0
    assert "Missing option '--image'" in result.output


def test_image_embedding_run_cli_errors_are_json(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    input_path = data / "input.parquet"
    input_path.write_bytes(b"parquet")

    result = CliRunner().invoke(
        main,
        [
            "integration",
            "image-embeddings",
            "run",
            "--data-dir",
            str(data),
            "--input-parquet",
            str(input_path),
            "--output-parquet",
            str(data / "output.parquet"),
            "--model-type",
            "unsupported",
            "--model-name-or-path",
            "model",
            "--image",
            "tao-toolkit-ds:test",
            "--dry-run",
        ],
    )

    assert result.exit_code == 2
    assert json.loads(result.output)["status"] == "error"


def test_prepare_cds_ce1_for_tdm_workflow_returns_manifest(tmp_path: Path) -> None:
    target = tmp_path / "target.parquet"
    source = tmp_path / "source.parquet"
    prepare_pair = MagicMock(return_value=(target, source))
    validate_pair = MagicMock(return_value=128)

    result = prepare_cds_ce1_for_tdm_handoff(
        PrepareCdsCe1ForTdmRequest(
            data_dir=str(tmp_path),
            target_selection="target.parquet",
            source_selection="source.parquet",
            output_subdir="_prep",
            embedding_family="CE1",
        ),
        validate_embedding_family=lambda family: family.lower(),
        prepare_pair=prepare_pair,
        validate_pair=validate_pair,
        container_path=lambda path, data_dir: f"/data/{Path(path).name}",
    )

    prepare_pair.assert_called_once_with(
        str(tmp_path),
        target_subdir="target.parquet",
        source_subdir="source.parquet",
        prep_subdir="_prep",
    )
    validate_pair.assert_called_once_with(target, source)
    assert result == {
        "status": "prepared",
        "embedding_family": "ce1",
        "dimension": 128,
        "target_selection": "target.parquet",
        "source_selection": "source.parquet",
        "target_parquet": str(target),
        "source_parquet": str(source),
        "target_container_path": "/data/target.parquet",
        "source_container_path": "/data/source.parquet",
    }


def test_image_embedding_workflow_rejects_mixed_config_and_direct_options() -> None:
    runner_factory = MagicMock()

    with pytest.raises(ValueError, match="cannot be combined"):
        run_image_embeddings(
            RunImageEmbeddingsRequest(
                data_dir="data",
                config_file="config.yaml",
                input_parquet="input.parquet",
            ),
            runner_factory=runner_factory,
        )

    runner_factory.assert_not_called()


def test_image_embedding_workflow_routes_direct_run() -> None:
    runner = MagicMock()
    runner.image_embeddings.return_value = SimpleNamespace(
        command=("docker", "run", "image_embeddings"),
        evidence={"rows": 1},
    )

    result = run_image_embeddings(
        RunImageEmbeddingsRequest(
            data_dir="data",
            input_parquet="input.parquet",
            output_parquet="output.parquet",
            model_type="clip",
            model_name_or_path="openai/clip-vit-base-patch32",
            batch_size=8,
            dry_run=True,
        ),
        runner_factory=lambda: runner,
    )

    runner.image_embeddings.assert_called_once_with(
        data_dir="data",
        input_parquet="input.parquet",
        output_parquet="output.parquet",
        model_type="clip",
        model_name_or_path="openai/clip-vit-base-patch32",
        model_config_path=None,
        batch_size=8,
        dry_run=True,
    )
    runner.image_embeddings_config.assert_not_called()
    assert result == {
        "status": "dry-run",
        "dry_run": True,
        "command": ["docker", "run", "image_embeddings"],
        "evidence": {"rows": 1},
    }


def test_image_embedding_validate_output_cli_emits_dimension_json(tmp_path: Path) -> None:
    data = tmp_path / "data"
    image = data / "a.jpg"
    image.parent.mkdir()
    image.write_bytes(b"image")
    input_path = data / "input.parquet"
    output_path = data / "output.parquet"
    pd.DataFrame({"filepath": ["/data/a.jpg"], "label": ["car"]}).to_parquet(
        input_path,
        index=False,
    )
    pd.DataFrame(
        {
            "filepath": ["/data/a.jpg"],
            "embedding": [[1.0, 2.0]],
            "label": ["car"],
        }
    ).to_parquet(output_path, index=False)

    result = CliRunner().invoke(
        main,
        [
            "integration",
            "image-embeddings",
            "validate-output",
            "--input-parquet",
            str(input_path),
            "--output-parquet",
            str(output_path),
            "--data-dir",
            str(data),
        ],
    )

    assert result.exit_code == 0
    assert json.loads(result.output)["dimension"] == 2
