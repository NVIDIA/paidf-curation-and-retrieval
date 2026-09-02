# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Secure S3-compatible staging through the deployment-provided AWS CLI."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

_ENV_NAME = re.compile(r"^[A-Z][A-Z0-9_]*$")


class ObjectStoreStagingError(RuntimeError):
    """Object-store staging failed without exposing credential values."""


def resolve_credential_pair(
    access_key_env: str,
    secret_key_env: str,
    source: Mapping[str, str] = os.environ,
) -> tuple[str | None, str | None]:
    """Resolve credentials by validated environment names without exposing values."""
    for name in (access_key_env, secret_key_env):
        if not _ENV_NAME.fullmatch(name):
            raise ValueError(f"Invalid credential environment reference: {name!r}")
    access_key = source.get(access_key_env)
    secret_key = source.get(secret_key_env)
    if bool(access_key) != bool(secret_key):
        raise ObjectStoreStagingError(
            "Both referenced object-store credential variables must be set together"
        )
    return access_key, secret_key


def validate_s3_endpoint(endpoint_url: str | None, *, allow_insecure: bool = False) -> None:
    """Require an absolute HTTPS endpoint unless lab-only HTTP is explicit."""
    if endpoint_url is None:
        return
    parsed = urlparse(endpoint_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("endpoint_url must be an absolute HTTP(S) URL")
    if parsed.scheme != "https" and not allow_insecure:
        raise ValueError("HTTP object-store endpoints require explicit lab-only opt-in")


@dataclass(frozen=True)
class S3StagingConfig:
    """Non-secret S3 endpoint and credential-reference configuration."""

    endpoint_url: str | None = None
    access_key_env: str = "AWS_ACCESS_KEY_ID"
    secret_key_env: str = "AWS_SECRET_ACCESS_KEY"
    allow_insecure_endpoint: bool = False

    def __post_init__(self) -> None:
        for name in (self.access_key_env, self.secret_key_env):
            if not _ENV_NAME.fullmatch(name):
                raise ValueError(f"Invalid credential environment reference: {name!r}")
        validate_s3_endpoint(
            self.endpoint_url,
            allow_insecure=self.allow_insecure_endpoint,
        )


class S3ObjectStoreStager:
    """Stage one local artifact to S3 without reading or logging credentials."""

    def __init__(
        self,
        config: S3StagingConfig,
        runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    ) -> None:
        self._config = config
        self._runner = runner

    def command(self, source: str | Path, destination_uri: str) -> Sequence[str]:
        """Build the non-interactive AWS CLI command after validating inputs."""
        source_path = Path(source)
        if not source_path.is_file():
            raise FileNotFoundError(f"Staging source is not a file: {source_path}")
        parsed = urlparse(destination_uri)
        if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.strip("/"):
            raise ValueError("destination_uri must be s3://bucket/key")
        executable = shutil.which("aws")
        if executable is None:
            raise ObjectStoreStagingError(
                "AWS CLI is required for S3-compatible staging; install it in the worker image"
            )
        command = [executable]
        if self._config.endpoint_url:
            command.extend(["--endpoint-url", self._config.endpoint_url])
        command.extend(
            ["s3", "cp", str(source_path), destination_uri, "--only-show-errors", "--no-progress"]
        )
        return command

    def stage(self, source: str | Path, destination_uri: str) -> str:
        """Upload one file, inheriting credentials from referenced environment variables."""
        command = self.command(source, destination_uri)
        environment = self._credential_environment(os.environ)
        try:
            self._runner(
                command,
                check=True,
                capture_output=True,
                text=True,
                env=environment,
            )
        except subprocess.CalledProcessError as exc:
            raise ObjectStoreStagingError(
                f"S3 staging failed with exit code {exc.returncode}; worker logs are redacted"
            ) from exc
        return destination_uri

    def _credential_environment(self, source: Mapping[str, str]) -> dict[str, str]:
        """Map referenced credentials without including values in diagnostics."""
        environment = dict(source)
        access_key, secret_key = resolve_credential_pair(
            self._config.access_key_env,
            self._config.secret_key_env,
            source,
        )
        if access_key and secret_key:
            environment["AWS_ACCESS_KEY_ID"] = access_key
            environment["AWS_SECRET_ACCESS_KEY"] = secret_key
        return environment
