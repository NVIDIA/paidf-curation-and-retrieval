# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Low-level Docker argv construction and subprocess execution helpers.

Used by :mod:`adapters.docker_jobs` to build Curator and TAO Data Services
``docker run`` commands without embedding product-specific parquet logic.
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "DockerJobError",
    "DockerRunResult",
    "build_curator_pipeline_command",
    "build_data_mining_command",
    "default_curator_image",
    "run_docker_command",
]


@dataclass
class DockerRunResult:
    returncode: int
    command: list[str]
    evidence: dict[str, Any] = field(default_factory=dict)


class DockerJobError(RuntimeError):
    pass


def default_curator_image() -> str:
    """Local Curator ``name:tag`` from env (Make exports pins; see Makefile)."""
    explicit = os.environ.get("COSMOS_CURATOR_DOCKER_IMAGE")
    if explicit:
        return explicit
    name = os.environ.get("COSMOS_CURATOR_IMAGE", "cosmos-curator")
    # Tag lives in the final path component (name:tag). A colon earlier can be
    # host:port (e.g. registry:5000/cosmos-curator) and must still get a tag.
    if ":" in name.rsplit("/", 1)[-1]:
        return name
    tag = os.environ.get("COSMOS_CURATOR_TAG", "2.3.0")
    return f"{name}:{tag}"


def run_docker_command(cmd: Sequence[str], *, dry_run: bool = False) -> DockerRunResult:
    """Run a Docker command argument list, or return it unchanged for dry runs."""
    command = list(cmd)
    if dry_run:
        return DockerRunResult(returncode=0, command=command)
    completed = subprocess.run(command, check=False, shell=False)
    if completed.returncode != 0:
        command_text = " ".join(command)
        raise DockerJobError(f"Command failed ({completed.returncode}): {command_text}")
    return DockerRunResult(returncode=completed.returncode, command=command)


def build_curator_pipeline_command(
    *,
    image: str,
    config_file: str,
    gpus: str,
    shm_size: str,
    pixi_env: str,
    data_dir: str | None = None,
    models_dir: str | None = None,
    ffmpeg_dir: str | None = None,
    extra_args: Sequence[str] | None = None,
) -> list[str]:
    """Build the Cosmos Curator pipeline Docker command."""
    cmd: list[str] = [
        "docker",
        "run",
        "--rm",
        f"--gpus={gpus}",
        f"--shm-size={shm_size}",
        "--network=host",
        "-v",
        f"{os.path.abspath(config_file)}:/config/pipeline_config.yaml:ro",
    ]
    if models_dir:
        cmd += [
            "-v",
            f"{os.path.abspath(models_dir)}:/config/models:ro",
            "-e",
            "HF_HOME=/config/models",
        ]
    if data_dir:
        abs_data = os.path.abspath(data_dir)
        cmd += ["-v", f"{abs_data}:{abs_data}"]
    if ffmpeg_dir:
        abs_ff = os.path.abspath(ffmpeg_dir)
        cmd += ["-v", f"{abs_ff}/bin:/usr/local/bin/ffmpeg-bin:ro"]
    if extra_args:
        cmd += list(extra_args)
    cmd += [
        image,
        "bash",
        "-c",
        (
            "cd /opt/cosmos-curator && "
            f"pixi run -e {pixi_env} --as-is "
            "python -m cosmos_curator.pipelines.video.run_pipeline "
            "/config/pipeline_config.yaml"
        ),
    ]
    return cmd


def build_data_mining_command(
    *,
    image: str,
    data_dir: str,
    gpus: str,
    shm_size: str,
    entry_cmd: str,
    args: Sequence[str],
    entrypoint: str | None = None,
    extra_mounts: Sequence[str] | None = None,
) -> list[str]:
    """Build a TAO Toolkit Data Services Docker command."""
    abs_data = os.path.abspath(data_dir)
    if not os.path.isdir(abs_data):
        raise DockerJobError(f"DATA_DIR not found: {abs_data}")
    cmd: list[str] = [
        "docker",
        "run",
        "--rm",
        f"--gpus={gpus}",
        f"--shm-size={shm_size}",
        "-v",
        f"{abs_data}:/data",
    ]
    for mount in extra_mounts or ():
        cmd += ["-v", mount]
    if entrypoint:
        cmd += ["--entrypoint", entrypoint]
    cmd += [image, entry_cmd, *list(args)]
    return cmd
