# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Docker job runners for Cosmos Curator and TAO Data Services images.

Integration layer only: construct ``docker run`` commands against pulled
vendor images. No engine source is vendored here.

``CuratorDockerRunner`` launches Cosmos Curator pipelines.
``DataMiningDockerRunner`` launches TAO ``embedding`` (image/text) and
``tmm`` (nearest neighbors / unique neighbor matching) entry points, after
the host-side parquet/config helpers have prepared experiment specs.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import yaml

from adapters.docker_runtime import (
    DockerJobError,
    DockerRunResult,
    build_curator_pipeline_command,
    build_data_mining_command,
    default_curator_image,
    run_docker_command,
)

__all__ = [
    "CuratorDockerRunner",
    "DataMiningDockerRunner",
    "DockerJobError",
    "DockerRunResult",
    "default_curator_image",
]

_GENERATED_TMM_SPEC = "/config/generated-tmm.yaml"
_GENERATED_IMAGE_EMBEDDING_SPEC = "/config/generated-image-embeddings.yaml"
_GENERATED_TEXT_EMBEDDING_SPEC = "/config/generated-text-embeddings.yaml"


@contextmanager
def _temporary_experiment_spec(
    payload: Mapping[str, Any],
    *,
    container_path: str,
) -> Iterator[tuple[str, dict[str, Any]]]:
    """Own one generated TAO experiment spec for a synchronous Docker call."""
    spec_yaml = yaml.safe_dump(dict(payload), sort_keys=False)
    with tempfile.TemporaryDirectory(prefix="paidf-tao-spec-") as directory:
        host_path = Path(directory) / "experiment.yaml"
        host_path.write_text(spec_yaml, encoding="utf-8")
        evidence = {
            "experiment_spec": dict(payload),
            "experiment_spec_yaml": spec_yaml,
            "experiment_spec_host_path": str(host_path),
            "experiment_spec_container_path": container_path,
            "dry_run_replayable": False,
        }
        yield f"{host_path}:{container_path}:ro", evidence


class CuratorDockerRunner:
    """Run the Cosmos Curator batch curation Docker image."""

    def __init__(
        self,
        image: str | None = None,
        *,
        gpus: str = "all",
        shm_size: str = "16g",
        pixi_env: str = "cuml",
    ) -> None:
        self.image = image or default_curator_image()
        self.gpus = gpus
        self.shm_size = shm_size
        self.pixi_env = pixi_env

    def run_pipeline(
        self,
        config_file: str,
        *,
        data_dir: str | None = None,
        models_dir: str | None = None,
        ffmpeg_dir: str | None = None,
        dry_run: bool = False,
        extra_args: Sequence[str] | None = None,
    ) -> DockerRunResult:
        if not os.path.isfile(config_file):
            raise FileNotFoundError(f"CONFIG_FILE not found: {config_file}")

        cmd = build_curator_pipeline_command(
            image=self.image,
            config_file=config_file,
            gpus=self.gpus,
            shm_size=self.shm_size,
            pixi_env=self.pixi_env,
            data_dir=data_dir,
            models_dir=models_dir,
            ffmpeg_dir=ffmpeg_dir,
            extra_args=extra_args,
        )
        return run_docker_command(cmd, dry_run=dry_run)


class DataMiningDockerRunner:
    """Run TAO Toolkit Data Services jobs for embeddings and TMM mining.

    Supported container entry points:

    * ``embedding image_embeddings`` / ``embedding text_embeddings``
    * ``tmm nearest_neighbors`` / ``tmm unique_neighbor_matching``

    Each public method generates or mounts an experiment spec, invokes Docker,
    and attaches host-side validation evidence. Cluster / diversity helpers in
    this repository remain in-process CPU adapters and are not DS console
    scripts.
    """

    def __init__(
        self,
        image: str,
        *,
        gpus: str = "all",
        shm_size: str = "16g",
    ) -> None:
        if not isinstance(image, str) or not image.strip():
            raise ValueError("TAO Data Services image must be a non-empty string")
        self.image = image.strip()
        self.gpus = gpus
        self.shm_size = shm_size

    def run(
        self,
        entry_cmd: str,
        args: Sequence[str],
        *,
        data_dir: str,
        dry_run: bool = False,
        entrypoint: str | None = None,
        extra_mounts: Sequence[str] | None = None,
    ) -> DockerRunResult:
        cmd = build_data_mining_command(
            image=self.image,
            data_dir=data_dir,
            gpus=self.gpus,
            shm_size=self.shm_size,
            entry_cmd=entry_cmd,
            args=args,
            entrypoint=entrypoint,
            extra_mounts=extra_mounts,
        )
        return run_docker_command(cmd, dry_run=dry_run)

    def image_embeddings_config(
        self,
        *,
        config_file: str,
        data_dir: str,
        dry_run: bool = False,
    ) -> DockerRunResult:
        """Run ``embedding image_embeddings`` with a validated vendor YAML."""
        from adapters.data_mining.image_embeddings import (
            validate_image_embedding_config,
            validate_image_embedding_output,
        )

        config = validate_image_embedding_config(config_file, data_dir=data_dir)
        config_mount = f"{os.path.abspath(config_file)}:/config/image_embeddings.yaml:ro"
        result = self.run(
            "image_embeddings",
            ["-e", "/config/image_embeddings.yaml"],
            data_dir=data_dir,
            dry_run=dry_run,
            entrypoint="embedding",
            extra_mounts=[config_mount],
        )
        if dry_run:
            return result
        evidence = validate_image_embedding_output(
            config["output_path"],
            data_dir=data_dir,
            input_path=config["input_path"],
        )
        return DockerRunResult(result.returncode, result.command, evidence)

    def image_embeddings(
        self,
        *,
        data_dir: str,
        input_parquet: str,
        output_parquet: str,
        model_type: str,
        model_name_or_path: str,
        model_config_path: str | None = None,
        batch_size: int = 64,
        dry_run: bool = False,
    ) -> DockerRunResult:
        """Run ``embedding image_embeddings`` with a generated TAO experiment spec."""
        from adapters.data_mining.image_embeddings import (
            container_data_path,
            resolve_data_path,
            validate_image_embedding_input,
            validate_image_embedding_model,
            validate_image_embedding_output,
        )

        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        input_path = resolve_data_path(
            input_parquet,
            data_dir,
            role="Input parquet",
            must_exist=True,
        )
        output_path = resolve_data_path(
            output_parquet,
            data_dir,
            role="Output parquet",
            must_exist=False,
        )
        validate_image_embedding_input(input_path, data_dir=data_dir)
        model = validate_image_embedding_model(
            model_type,
            model_name_or_path,
            data_dir=data_dir,
            model_config_path=model_config_path,
        )
        payload = {
            "input_parquet": container_data_path(input_path, data_dir, role="Input parquet"),
            "output_parquet": container_data_path(output_path, data_dir, role="Output parquet"),
            "model": model["model"],
            "model_path": model["model_path"],
            "batch_size": batch_size,
        }
        if "model_config_path" in model:
            payload["model_config_path"] = model["model_config_path"]

        with _temporary_experiment_spec(
            payload,
            container_path=_GENERATED_IMAGE_EMBEDDING_SPEC,
        ) as (config_mount, spec_evidence):
            result = self.run(
                "image_embeddings",
                ["-e", _GENERATED_IMAGE_EMBEDDING_SPEC],
                data_dir=data_dir,
                dry_run=dry_run,
                entrypoint="embedding",
                extra_mounts=[config_mount],
            )
        if dry_run:
            return DockerRunResult(result.returncode, result.command, spec_evidence)
        evidence = validate_image_embedding_output(
            output_path,
            data_dir=data_dir,
            input_path=input_path,
        )
        return DockerRunResult(result.returncode, result.command, {**spec_evidence, **evidence})

    def text_embeddings_config(
        self,
        *,
        config_file: str,
        data_dir: str,
        dry_run: bool = False,
    ) -> DockerRunResult:
        """Run ``embedding text_embeddings`` with a validated vendor YAML."""
        from adapters.data_mining.text_embeddings import (
            validate_text_embedding_config,
            validate_text_embedding_output,
        )

        config = validate_text_embedding_config(config_file, data_dir=data_dir)
        config_mount = f"{os.path.abspath(config_file)}:/config/text_embeddings.yaml:ro"
        result = self.run(
            "text_embeddings",
            ["-e", "/config/text_embeddings.yaml"],
            data_dir=data_dir,
            dry_run=dry_run,
            entrypoint="embedding",
            extra_mounts=[config_mount],
        )
        if dry_run:
            return result
        evidence = validate_text_embedding_output(
            config["output_path"],
            data_dir=data_dir,
            input_path=config["input_path"],
        )
        return DockerRunResult(result.returncode, result.command, evidence)

    def text_embeddings(
        self,
        *,
        data_dir: str,
        input_parquet: str,
        output_parquet: str,
        model: str,
        model_path: str,
        batch_size: int = 64,
        dry_run: bool = False,
    ) -> DockerRunResult:
        """Run ``embedding text_embeddings`` with a generated TAO experiment spec."""
        from adapters.data_mining.text_embeddings import (
            resolve_text_data_path,
            text_container_data_path,
            validate_text_embedding_input,
            validate_text_embedding_model,
            validate_text_embedding_output,
        )

        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        input_path = resolve_text_data_path(
            input_parquet,
            data_dir,
            role="Input parquet",
            must_exist=True,
        )
        output_path = resolve_text_data_path(
            output_parquet,
            data_dir,
            role="Output parquet",
            must_exist=False,
        )
        validate_text_embedding_input(input_path, data_dir=data_dir)
        model_fields = validate_text_embedding_model(model, model_path, data_dir=data_dir)
        payload = {
            "input_parquet": text_container_data_path(input_path, data_dir, role="Input parquet"),
            "output_parquet": text_container_data_path(
                output_path, data_dir, role="Output parquet"
            ),
            "model": model_fields["model"],
            "model_path": model_fields["model_path"],
            "batch_size": batch_size,
        }
        with _temporary_experiment_spec(
            payload,
            container_path=_GENERATED_TEXT_EMBEDDING_SPEC,
        ) as (config_mount, spec_evidence):
            result = self.run(
                "text_embeddings",
                ["-e", _GENERATED_TEXT_EMBEDDING_SPEC],
                data_dir=data_dir,
                dry_run=dry_run,
                entrypoint="embedding",
                extra_mounts=[config_mount],
            )
        if dry_run:
            return DockerRunResult(result.returncode, result.command, spec_evidence)
        evidence = validate_text_embedding_output(
            output_path,
            data_dir=data_dir,
            input_path=input_path,
        )
        return DockerRunResult(result.returncode, result.command, {**spec_evidence, **evidence})

    def tmm_nearest_neighbors_config(
        self,
        *,
        config_file: str,
        data_dir: str,
        dry_run: bool = False,
    ) -> DockerRunResult:
        """Run TMM with an engine-native TAO experiment YAML."""
        if not os.path.isfile(config_file):
            raise FileNotFoundError(f"TMM_CONFIG_FILE not found: {config_file}")
        from adapters.data_mining.tmm_parquet import validate_tmm_config_inputs

        validate_tmm_config_inputs(config_file, data_dir)
        config_mount = f"{os.path.abspath(config_file)}:/config/tmm.yaml:ro"
        return self.run(
            "nearest_neighbors",
            ["-e", "/config/tmm.yaml"],
            data_dir=data_dir,
            dry_run=dry_run,
            entrypoint="tmm",
            extra_mounts=[config_mount],
        )

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
    ) -> DockerRunResult:
        """Run ``tmm nearest_neighbors`` against prepared parquet under ``/data``."""
        from adapters.data_mining.tmm_parquet import (
            TmmColumnSelection,
            container_data_path,
            prepare_tmm_parquet_pair,
            validate_embed_column_name,
            validate_tmm_run_options,
        )

        abs_data = os.path.abspath(data_dir)
        output_dir, prep_dir, metric, distance_threshold = validate_tmm_run_options(
            abs_data,
            output_subdir=output_subdir,
            prep_subdir=prep_subdir,
            topn=topn,
            metric=metric,
            distance_threshold=distance_threshold,
        )
        if not isinstance(filter_by_label, bool):
            raise ValueError("filter_by_label must be a boolean")
        source_col = validate_embed_column_name(
            source_embed_column_name, field="source_embed_column_name"
        )
        target_col = validate_embed_column_name(
            target_embed_column_name, field="target_embed_column_name"
        )
        if not dry_run:
            output_dir.mkdir(parents=True, exist_ok=True)
            target_host, source_host = prepare_tmm_parquet_pair(
                abs_data,
                target_subdir=target_subdir,
                source_subdir=source_subdir,
                prep_subdir=prep_subdir,
                columns=TmmColumnSelection(target_embed=target_col, source_embed=source_col),
            )
            target_c = container_data_path(target_host, abs_data)
            source_c = container_data_path(source_host, abs_data)
        else:
            target_c = container_data_path(prep_dir / "target.parquet", abs_data)
            source_c = container_data_path(prep_dir / "source.parquet", abs_data)

        output_c = container_data_path(output_dir / "mined.parquet", abs_data)
        payload = {
            "source_parquet": source_c,
            "target_parquet": target_c,
            "output_parquet": output_c,
            "topn": topn,
            "knn_metric": metric,
            "filter_by_label": "true" if filter_by_label else "false",
            "distance_threshold": distance_threshold,
            "source_embed_column_name": source_col,
            "target_embed_column_name": target_col,
        }
        with _temporary_experiment_spec(
            payload,
            container_path=_GENERATED_TMM_SPEC,
        ) as (config_mount, spec_evidence):
            result = self.run(
                "nearest_neighbors",
                ["-e", _GENERATED_TMM_SPEC],
                data_dir=data_dir,
                dry_run=dry_run,
                entrypoint="tmm",
                extra_mounts=[config_mount],
            )
        return DockerRunResult(result.returncode, result.command, spec_evidence)

    def tmm_unique_neighbor_matching_config(
        self,
        *,
        config_file: str,
        data_dir: str,
        dry_run: bool = False,
    ) -> DockerRunResult:
        """Run unique_neighbor_matching with an engine-native TAO experiment YAML."""
        if not os.path.isfile(config_file):
            raise FileNotFoundError(f"TMM_UNM_CONFIG_FILE not found: {config_file}")
        from adapters.data_mining.tmm_unm import validate_unm_config_inputs

        validate_unm_config_inputs(config_file, data_dir)
        config_mount = f"{os.path.abspath(config_file)}:/config/tmm_unm.yaml:ro"
        return self.run(
            "unique_neighbor_matching",
            ["-e", "/config/tmm_unm.yaml"],
            data_dir=data_dir,
            dry_run=dry_run,
            entrypoint="tmm",
            extra_mounts=[config_mount],
        )

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
    ) -> DockerRunResult:
        """Run ``tmm unique_neighbor_matching`` against prepared parquet under ``/data``."""
        from adapters.data_mining.tmm_parquet import (
            TmmColumnSelection,
            container_data_path,
            prepare_tmm_parquet_pair,
            validate_tmm_run_options,
        )
        from adapters.data_mining.tmm_unm import (
            resolve_optional_data_selection,
            validate_unm_direct_options,
            validate_unm_output,
        )

        abs_data = os.path.abspath(data_dir)
        require_optional_paths = not dry_run
        exclude_c = resolve_optional_data_selection(
            abs_data,
            exclude_path,
            role="exclude_path",
            must_exist=require_optional_paths,
        )
        source_det_c = resolve_optional_data_selection(
            abs_data,
            source_detection_file,
            role="source_detection_file",
            must_exist=require_optional_paths,
            detection_format=detection_format,
        )
        target_det_c = resolve_optional_data_selection(
            abs_data,
            target_detection_file,
            role="target_detection_file",
            must_exist=require_optional_paths,
            detection_format=detection_format,
        )
        options = validate_unm_direct_options(
            allocation_policy=allocation_policy,
            metric=metric,
            desired_unique_count=desired_unique_count,
            candidate_expansion_factor=candidate_expansion_factor,
            source_embedding_column=source_embedding_column,
            target_embedding_column=target_embedding_column,
            source_filepath_column=source_filepath_column,
            target_filepath_column=target_filepath_column,
            exclude_path=exclude_c,
            source_detection_file=source_det_c,
            target_detection_file=target_det_c,
            detection_format=detection_format,
            rare_class_list=rare_class_list,
            save_embeddings=save_embeddings,
            visualize=visualize,
        )

        output_dir, prep_dir, _, _ = validate_tmm_run_options(
            abs_data,
            output_subdir=output_subdir,
            prep_subdir=prep_subdir,
            topn=1,
            metric=options["distance_metric"],
        )
        if not dry_run:
            output_dir.mkdir(parents=True, exist_ok=True)
            target_host, source_host = prepare_tmm_parquet_pair(
                abs_data,
                target_subdir=target_subdir,
                source_subdir=source_subdir,
                prep_subdir=prep_subdir,
                columns=TmmColumnSelection(
                    target_embed=options["target_embedding_column"],
                    source_embed=options["source_embedding_column"],
                    target_filepath=options["target_filepath_column"],
                    source_filepath=options["source_filepath_column"],
                ),
            )
            target_c = container_data_path(target_host, abs_data)
            source_c = container_data_path(source_host, abs_data)
        else:
            target_c = container_data_path(prep_dir / "target.parquet", abs_data)
            source_c = container_data_path(prep_dir / "source.parquet", abs_data)

        output_c = container_data_path(output_dir, abs_data)
        payload = {
            "source_path": source_c,
            "target_path": target_c,
            "output_dir": output_c,
            **options,
        }
        with _temporary_experiment_spec(
            payload,
            container_path="/config/generated-tmm-unm.yaml",
        ) as (config_mount, spec_evidence):
            result = self.run(
                "unique_neighbor_matching",
                ["-e", "/config/generated-tmm-unm.yaml"],
                data_dir=data_dir,
                dry_run=dry_run,
                entrypoint="tmm",
                extra_mounts=[config_mount],
            )
        if dry_run:
            return DockerRunResult(result.returncode, result.command, spec_evidence)
        output_evidence = validate_unm_output(output_dir, abs_data)
        return DockerRunResult(
            result.returncode,
            result.command,
            {**spec_evidence, **output_evidence},
        )

    def divknn(
        self,
        *,
        data_dir: str,
        target_subdir: str = "S",
        source_subdir: str = "B",
        output_subdir: str = "divknn_out",
        topn: int = 5,
        metric: str = "cosine",
        dry_run: bool = False,
    ) -> DockerRunResult:
        """Compatibility alias — DS image mining is ``tmm nearest_neighbors``."""
        return self.tmm_nearest_neighbors(
            data_dir=data_dir,
            target_subdir=target_subdir,
            source_subdir=source_subdir,
            output_subdir=output_subdir,
            topn=topn,
            metric=metric,
            dry_run=dry_run,
        )
