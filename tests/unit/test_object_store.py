# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for secure, non-interactive S3-compatible artifact staging."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from adapters.object_store import (
    ObjectStoreStagingError,
    S3ObjectStoreStager,
    S3StagingConfig,
    resolve_credential_pair,
)


def test_staging_config_requires_https_by_default() -> None:
    with pytest.raises(ValueError, match="lab-only"):
        S3StagingConfig(endpoint_url="http://minio.example:9000")

    config = S3StagingConfig(
        endpoint_url="http://minio.example:9000",
        allow_insecure_endpoint=True,
    )
    assert config.endpoint_url is not None
    assert config.endpoint_url.startswith("http://")


def test_staging_config_rejects_invalid_endpoint_and_env_reference() -> None:
    with pytest.raises(ValueError, match="absolute"):
        S3StagingConfig(endpoint_url="minio.example")
    with pytest.raises(ValueError, match="environment reference"):
        S3StagingConfig(access_key_env="not-valid")


def test_resolve_credential_pair_requires_both_without_leaking_values() -> None:
    secret = "do-not-print"
    with pytest.raises(ObjectStoreStagingError) as error:
        resolve_credential_pair(
            "ACCESS_REF",
            "SECRET_REF",
            {"ACCESS_REF": secret},
        )
    assert secret not in str(error.value)


def test_stage_builds_noninteractive_command_and_maps_referenced_credentials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "artifact.parquet"
    source.write_bytes(b"parquet")
    monkeypatch.setattr("adapters.object_store.shutil.which", lambda _: "/usr/bin/aws")
    monkeypatch.setenv("LAB_ACCESS", "access-value")
    monkeypatch.setenv("LAB_SECRET", "secret-value")
    runner = MagicMock(return_value=subprocess.CompletedProcess([], 0))
    stager = S3ObjectStoreStager(
        S3StagingConfig(
            endpoint_url="https://objects.example",
            access_key_env="LAB_ACCESS",
            secret_key_env="LAB_SECRET",
        ),
        runner=runner,
    )

    result = stager.stage(source, "s3://bucket/artifact.parquet")

    assert result == "s3://bucket/artifact.parquet"
    command = runner.call_args.args[0]
    assert command == [
        "/usr/bin/aws",
        "--endpoint-url",
        "https://objects.example",
        "s3",
        "cp",
        str(source),
        "s3://bucket/artifact.parquet",
        "--only-show-errors",
        "--no-progress",
    ]
    environment = runner.call_args.kwargs["env"]
    assert environment["AWS_ACCESS_KEY_ID"] == "access-value"
    assert environment["AWS_SECRET_ACCESS_KEY"] == "secret-value"


def test_stage_redacts_subprocess_output_and_credentials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "artifact.parquet"
    source.write_bytes(b"parquet")
    monkeypatch.setattr("adapters.object_store.shutil.which", lambda _: "/usr/bin/aws")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "sensitive-access")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "sensitive-secret")
    runner = MagicMock(
        side_effect=subprocess.CalledProcessError(
            9,
            ["aws"],
            stderr="sensitive-secret",
        )
    )

    with pytest.raises(ObjectStoreStagingError) as error:
        S3ObjectStoreStager(S3StagingConfig(), runner=runner).stage(
            source,
            "s3://bucket/artifact.parquet",
        )

    message = str(error.value)
    assert "exit code 9" in message
    assert "sensitive" not in message


def test_stage_rejects_invalid_destination_and_missing_cli(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "artifact.parquet"
    source.write_bytes(b"parquet")
    stager = S3ObjectStoreStager(S3StagingConfig())
    with pytest.raises(ValueError, match="s3://bucket/key"):
        stager.command(source, "https://bucket/artifact.parquet")

    monkeypatch.setattr("adapters.object_store.shutil.which", lambda _: None)
    with pytest.raises(ObjectStoreStagingError, match="AWS CLI"):
        stager.command(source, "s3://bucket/artifact.parquet")
