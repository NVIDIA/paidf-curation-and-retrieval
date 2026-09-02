# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Non-interactive Make-facing integration commands.

Exposes finite handoff subcommands (image/text embeddings, artifact staging,
and related helpers) consumed by repository Make targets. Commands print JSON
status/evidence and exit non-zero on contract failures; they do not replace
the public Make UX.
"""

from __future__ import annotations

import json
import os
from typing import Any, NoReturn

import click

from adapters.data_mining.image_embeddings import (
    ImageEmbeddingParquetError,
    build_image_embedding_input,
    validate_image_embedding_input,
    validate_image_embedding_output,
)
from adapters.data_mining.text_embeddings import (
    TextEmbeddingParquetError,
    validate_text_embedding_input,
    validate_text_embedding_output,
)
from adapters.data_mining.tmm_parquet import (
    TmmParquetError,
    container_data_path,
    prepare_tmm_parquet_pair,
    validate_tmm_parquet_pair,
)
from adapters.dataset_search.caption_adapter import CaptionCapabilityError
from adapters.dataset_search.readiness import external_caption_readiness
from adapters.dataset_search.retrieval_adapter import validate_cds_embedding_family
from adapters.docker_jobs import DockerJobError
from adapters.object_store import (
    ObjectStoreStagingError,
    S3ObjectStoreStager,
    S3StagingConfig,
    resolve_credential_pair,
)
from adapters.schema.caption_parquet import (
    CaptionParquetError,
    build_caption_parquet,
    load_indexed_clip_ids,
    validate_caption_parquet,
)
from apps.composition import (
    build_caption_adapter,
    build_data_mining_runner,
    build_dataset_search_adapter,
)
from apps.workflows import (
    BuildCaptionParquetRequest,
    BuildImageEmbeddingInputRequest,
    BulkInsertCaptionParquetsRequest,
    CaptionReadinessRequest,
    CaptionSearchRequest,
    PrepareCdsCe1ForTdmRequest,
    RunImageEmbeddingsRequest,
    RunTextEmbeddingsRequest,
    StageArtifactRequest,
    UploadCaptionParquetRequest,
    ValidateCaptionParquetRequest,
    ValidateImageEmbeddingInputRequest,
    ValidateImageEmbeddingOutputRequest,
    ValidateTextEmbeddingOutputRequest,
    build_caption_parquet_handoff,
    build_image_embedding_input_handoff,
    bulk_insert_caption_parquets,
    prepare_cds_ce1_for_tdm_handoff,
    run_caption_readiness,
    run_image_embeddings,
    run_text_embeddings,
    search_captions,
    stage_artifact_handoff,
    upload_caption_parquet_handoff,
    validate_caption_parquet_handoff,
    validate_image_embedding_input_handoff,
    validate_image_embedding_output_handoff,
    validate_text_embedding_output_handoff,
)


def _echo_json(payload: Any) -> None:
    click.echo(json.dumps(payload, indent=2, default=str))


def _json_error(exc: Exception) -> NoReturn:
    _echo_json({"status": "error", "error": str(exc)})
    raise click.exceptions.Exit(2)


def _credentials(access_key_env: str, secret_key_env: str) -> tuple[str | None, str | None]:
    try:
        return resolve_credential_pair(access_key_env, secret_key_env)
    except (ObjectStoreStagingError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc


def _indexed_ids(indexed_id_file: str | None) -> set[str] | None:
    if indexed_id_file is None:
        return None
    try:
        return load_indexed_clip_ids(indexed_id_file)
    except (CaptionParquetError, FileNotFoundError) as exc:
        raise click.ClickException(str(exc)) from exc


@click.group("integration")
def integration_group() -> None:
    """Finite artifact/API handoffs for the external PAIDF orchestration team."""


@integration_group.group("image-embeddings")
def image_embeddings_group() -> None:
    """Build, run, and validate the TAO DS image-embedding handoff."""


@image_embeddings_group.command("build")
@click.option("--input-json", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--data-dir", required=True, type=click.Path(exists=True, file_okay=False))
@click.option("--output-parquet", required=True, type=click.Path(dir_okay=False))
def image_embeddings_build(input_json: str, data_dir: str, output_parquet: str) -> None:
    """Build an input parquet from a JSON array of image rows."""
    try:
        result = build_image_embedding_input_handoff(
            BuildImageEmbeddingInputRequest(
                input_json=input_json,
                data_dir=data_dir,
                output_parquet=output_parquet,
            ),
            build_input=build_image_embedding_input,
            validate_input=validate_image_embedding_input,
        )
    except (
        ImageEmbeddingParquetError,
        FileNotFoundError,
        json.JSONDecodeError,
        OSError,
        ValueError,
    ) as exc:
        _json_error(exc)
    _echo_json(result)


@image_embeddings_group.command("validate-input")
@click.option("--parquet", "parquet_path", required=True, type=click.Path(exists=True))
@click.option("--data-dir", required=True, type=click.Path(exists=True, file_okay=False))
def image_embeddings_validate_input(parquet_path: str, data_dir: str) -> None:
    """Validate local image paths and the TAO DS input schema."""
    try:
        result = validate_image_embedding_input_handoff(
            ValidateImageEmbeddingInputRequest(
                parquet_path=parquet_path,
                data_dir=data_dir,
            ),
            validate_input=validate_image_embedding_input,
        )
    except (ImageEmbeddingParquetError, FileNotFoundError, OSError) as exc:
        _json_error(exc)
    _echo_json(result)


@image_embeddings_group.command("run")
@click.option("--data-dir", required=True, type=click.Path(exists=True, file_okay=False))
@click.option("--input-parquet", default=None, type=click.Path(exists=True, dir_okay=False))
@click.option("--output-parquet", default=None, type=click.Path(dir_okay=False))
@click.option(
    "--model-type",
    default=None,
    help="Required in direct mode: clip or siglip",
)
@click.option("--model-name-or-path", default=None)
@click.option("--model-config-path", default=None, type=click.Path(exists=True, dir_okay=False))
@click.option("--config-file", default=None, type=click.Path(exists=True, dir_okay=False))
@click.option("--batch-size", default=64, type=click.IntRange(min=1), show_default=True)
@click.option("--gpus", default=lambda: os.environ.get("GPUS", "all"))
@click.option(
    "--shm-size",
    default=lambda: os.environ.get("DATA_MINING_SHM_SIZE", "16g"),
)
@click.option(
    "--image",
    required=True,
    help="TAO Data Services image reference; Make supplies its configured registry/tag",
)
@click.option("--dry-run/--no-dry-run", default=False)
def image_embeddings_run(
    data_dir: str,
    input_parquet: str | None,
    output_parquet: str | None,
    model_type: str | None,
    model_name_or_path: str | None,
    model_config_path: str | None,
    config_file: str | None,
    batch_size: int,
    gpus: str,
    shm_size: str,
    image: str,
    dry_run: bool,
) -> None:
    """Run TAO DS with direct overrides or one validated engine-native YAML."""
    try:
        result = run_image_embeddings(
            RunImageEmbeddingsRequest(
                data_dir=data_dir,
                input_parquet=input_parquet,
                output_parquet=output_parquet,
                model_type=model_type,
                model_name_or_path=model_name_or_path,
                model_config_path=model_config_path,
                config_file=config_file,
                batch_size=batch_size,
                dry_run=dry_run,
            ),
            runner_factory=lambda: build_data_mining_runner(
                image=image,
                gpus=gpus,
                shm_size=shm_size,
            ),
        )
    except (
        DockerJobError,
        ImageEmbeddingParquetError,
        FileNotFoundError,
        OSError,
        ValueError,
    ) as exc:
        _json_error(exc)
    _echo_json(result)


@image_embeddings_group.command("validate-output")
@click.option("--input-parquet", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--output-parquet", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--data-dir", required=True, type=click.Path(exists=True, file_okay=False))
def image_embeddings_validate_output(
    input_parquet: str,
    output_parquet: str,
    data_dir: str,
) -> None:
    """Validate TAO DS output vectors and metadata preservation."""
    try:
        result = validate_image_embedding_output_handoff(
            ValidateImageEmbeddingOutputRequest(
                input_parquet=input_parquet,
                output_parquet=output_parquet,
                data_dir=data_dir,
            ),
            validate_output=validate_image_embedding_output,
        )
    except (ImageEmbeddingParquetError, FileNotFoundError, OSError) as exc:
        _json_error(exc)
    _echo_json(result)


@integration_group.group("text-embeddings")
def text_embeddings_group() -> None:
    """Run and validate the TAO DS text-embedding handoff."""


@text_embeddings_group.command("validate-input")
@click.option("--parquet", "parquet_path", required=True, type=click.Path(exists=True))
@click.option("--data-dir", required=True, type=click.Path(exists=True, file_okay=False))
def text_embeddings_validate_input(parquet_path: str, data_dir: str) -> None:
    """Validate the TAO DS text-embedding input schema."""
    try:
        result = validate_text_embedding_input(parquet_path, data_dir=data_dir)
    except (TextEmbeddingParquetError, FileNotFoundError, OSError) as exc:
        _json_error(exc)
    _echo_json({"status": "valid", **result})


@text_embeddings_group.command("run")
@click.option("--data-dir", required=True, type=click.Path(exists=True, file_okay=False))
@click.option("--input-parquet", default=None, type=click.Path(exists=True, dir_okay=False))
@click.option("--output-parquet", default=None, type=click.Path(dir_okay=False))
@click.option(
    "--model",
    default=None,
    help="Required in direct mode: clip, siglip, or siglip2",
)
@click.option("--model-path", default=None, help="Hugging Face model id or DATA_DIR-local path")
@click.option("--config-file", default=None, type=click.Path(exists=True, dir_okay=False))
@click.option("--batch-size", default=64, type=click.IntRange(min=1), show_default=True)
@click.option("--gpus", default=lambda: os.environ.get("GPUS", "all"))
@click.option(
    "--shm-size",
    default=lambda: os.environ.get("DATA_MINING_SHM_SIZE", "16g"),
)
@click.option(
    "--image",
    required=True,
    help="TAO Data Services image reference; Make supplies its configured registry/tag",
)
@click.option("--dry-run/--no-dry-run", default=False)
def text_embeddings_run(
    data_dir: str,
    input_parquet: str | None,
    output_parquet: str | None,
    model: str | None,
    model_path: str | None,
    config_file: str | None,
    batch_size: int,
    gpus: str,
    shm_size: str,
    image: str,
    dry_run: bool,
) -> None:
    """Run TAO DS text embeddings with direct overrides or a validated YAML."""
    try:
        result = run_text_embeddings(
            RunTextEmbeddingsRequest(
                data_dir=data_dir,
                input_parquet=input_parquet,
                output_parquet=output_parquet,
                model=model,
                model_path=model_path,
                config_file=config_file,
                batch_size=batch_size,
                dry_run=dry_run,
            ),
            runner_factory=lambda: build_data_mining_runner(
                image=image,
                gpus=gpus,
                shm_size=shm_size,
            ),
        )
    except (
        DockerJobError,
        TextEmbeddingParquetError,
        FileNotFoundError,
        OSError,
        ValueError,
    ) as exc:
        _json_error(exc)
    _echo_json(result)


@text_embeddings_group.command("validate-output")
@click.option("--input-parquet", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--output-parquet", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--data-dir", required=True, type=click.Path(exists=True, file_okay=False))
def text_embeddings_validate_output(
    input_parquet: str,
    output_parquet: str,
    data_dir: str,
) -> None:
    """Validate TAO DS text vectors and metadata preservation."""
    try:
        result = validate_text_embedding_output_handoff(
            ValidateTextEmbeddingOutputRequest(
                input_parquet=input_parquet,
                output_parquet=output_parquet,
                data_dir=data_dir,
            ),
            validate_output=validate_text_embedding_output,
        )
    except (TextEmbeddingParquetError, FileNotFoundError, OSError) as exc:
        _json_error(exc)
    _echo_json(result)


@integration_group.command("stage-artifact")
@click.option("--source", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--destination", required=True, help="S3 URI: s3://bucket/key")
@click.option("--endpoint-url", default=None)
@click.option("--access-key-env", default="AWS_ACCESS_KEY_ID", show_default=True)
@click.option("--secret-key-env", default="AWS_SECRET_ACCESS_KEY", show_default=True)
@click.option("--allow-lab-http-endpoint", is_flag=True)
def stage_artifact(
    source: str,
    destination: str,
    endpoint_url: str | None,
    access_key_env: str,
    secret_key_env: str,
    allow_lab_http_endpoint: bool,
) -> None:
    """Stage one CDS-readable artifact through the deployment-provided AWS CLI."""
    config = S3StagingConfig(
        endpoint_url=endpoint_url,
        access_key_env=access_key_env,
        secret_key_env=secret_key_env,
        allow_insecure_endpoint=allow_lab_http_endpoint,
    )
    try:
        result = stage_artifact_handoff(
            StageArtifactRequest(source=source, destination=destination),
            stager_factory=lambda: S3ObjectStoreStager(config),
        )
    except (ObjectStoreStagingError, ValueError, FileNotFoundError) as exc:
        raise click.ClickException(str(exc)) from exc
    _echo_json(result)


@integration_group.command("prepare-cds-ce1-for-tdm")
@click.option("--data-dir", required=True, type=click.Path(exists=True, file_okay=False))
@click.option(
    "--target-selection",
    required=True,
    help="CDS-compatible CE1 target parquet file or directory under DATA_DIR",
)
@click.option(
    "--source-selection",
    required=True,
    help="CDS-compatible CE1 source parquet file or directory under DATA_DIR",
)
@click.option("--output-subdir", default="_tmm_prep", show_default=True)
@click.option(
    "--embedding-family",
    required=True,
    type=click.Choice(["ce1"], case_sensitive=False),
    help="Required producer/artifact declaration; vectors do not identify their family",
)
def prepare_cds_ce1_for_tdm(
    data_dir: str,
    target_selection: str,
    source_selection: str,
    output_subdir: str,
    embedding_family: str,
) -> None:
    """Prepare selected CDS-compatible CE1 parquet as finite TMM S/B inputs."""
    try:
        result = prepare_cds_ce1_for_tdm_handoff(
            PrepareCdsCe1ForTdmRequest(
                data_dir=data_dir,
                target_selection=target_selection,
                source_selection=source_selection,
                output_subdir=output_subdir,
                embedding_family=embedding_family,
            ),
            validate_embedding_family=validate_cds_embedding_family,
            prepare_pair=prepare_tmm_parquet_pair,
            validate_pair=validate_tmm_parquet_pair,
            container_path=container_data_path,
        )
    except (TmmParquetError, OSError, ValueError) as exc:
        _json_error(exc)
    _echo_json(result)


@integration_group.group("captions")
def captions_group() -> None:
    """Build, validate, and submit the CDS EA external-caption contract."""


@captions_group.command("readiness")
@click.option("--cds-url", required=True)
@click.option(
    "--cds-profile",
    default=lambda: os.environ.get("CDS_PROFILE", "public"),
    type=click.Choice(["public", "internal"], case_sensitive=False),
    show_default="CDS_PROFILE or public",
)
@click.option("--pipeline", "required_pipeline", required=True)
def caption_readiness(cds_url: str, cds_profile: str, required_pipeline: str) -> None:
    """Emit selected-flow EA external-caption readiness as JSON."""
    result = run_caption_readiness(
        CaptionReadinessRequest(
            cds_profile=cds_profile,
            required_pipeline=required_pipeline,
        ),
        cds_client=build_dataset_search_adapter(cds_url),
        caption_client=build_caption_adapter(cds_url),
        readiness_check=external_caption_readiness,
    )
    _echo_json(result)
    if not result["ready"]:
        raise click.exceptions.Exit(2)


@captions_group.command("search")
@click.option("--query", required=True, help="Caption keyword query")
@click.option("--limit", default=5000, type=int, show_default=True)
@click.option("--data-source", "data_sources", multiple=True)
@click.option("--cds-url", required=True)
def caption_search(
    query: str,
    limit: int,
    data_sources: tuple[str, ...],
    cds_url: str,
) -> None:
    """Run finite EA caption-keyword acceptance; no fused retrieval."""
    try:
        result = search_captions(
            CaptionSearchRequest(
                query=query,
                limit=limit,
                data_sources=data_sources or None,
            ),
            client_factory=lambda: build_caption_adapter(cds_url),
        )
    except (CaptionCapabilityError, ValueError) as exc:
        _json_error(exc)
    _echo_json(result)


@captions_group.command("build")
@click.option("--input-json", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--output-parquet", required=True, type=click.Path(dir_okay=False))
@click.option("--indexed-id-file", default=None, type=click.Path(exists=True, dir_okay=False))
def caption_build(input_json: str, output_parquet: str, indexed_id_file: str | None) -> None:
    """Build `clip_id`/`summary` parquet from a JSON array of row objects."""
    try:
        result = build_caption_parquet_handoff(
            BuildCaptionParquetRequest(
                input_json=input_json,
                output_parquet=output_parquet,
                indexed_clip_ids=_indexed_ids(indexed_id_file),
            ),
            build_caption=build_caption_parquet,
            validate_caption=validate_caption_parquet,
        )
    except (CaptionParquetError, FileNotFoundError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    _echo_json(result)


@captions_group.command("validate")
@click.option("--parquet", "parquet_path", required=True, type=click.Path(exists=True))
@click.option("--indexed-id-file", default=None, type=click.Path(exists=True, dir_okay=False))
def caption_validate(parquet_path: str, indexed_id_file: str | None) -> None:
    """Validate caption schema and optional exact indexed-ID alignment."""
    try:
        result = validate_caption_parquet_handoff(
            ValidateCaptionParquetRequest(
                parquet_path=parquet_path,
                indexed_clip_ids=_indexed_ids(indexed_id_file),
            ),
            validate_caption=validate_caption_parquet,
        )
    except (CaptionParquetError, FileNotFoundError) as exc:
        raise click.ClickException(str(exc)) from exc
    _echo_json(result)


@captions_group.command("upload")
@click.option("--parquet", "parquet_path", required=True, type=click.Path(exists=True))
@click.option("--model-name", default="default", show_default=True)
@click.option("--data-source", default="")
@click.option("--indexed-id-file", default=None, type=click.Path(exists=True, dir_okay=False))
@click.option("--cds-url", required=True)
def caption_upload(
    parquet_path: str,
    model_name: str,
    data_source: str,
    indexed_id_file: str | None,
    cds_url: str,
) -> None:
    """Validate and upload one local caption parquet without retrying POST."""
    try:
        result = upload_caption_parquet_handoff(
            UploadCaptionParquetRequest(
                parquet_path=parquet_path,
                model_name=model_name,
                data_source=data_source,
                indexed_clip_ids=_indexed_ids(indexed_id_file),
            ),
            validate_caption=validate_caption_parquet,
            client_factory=lambda: build_caption_adapter(cds_url),
        )
    except (CaptionCapabilityError, CaptionParquetError, FileNotFoundError) as exc:
        raise click.ClickException(str(exc)) from exc
    _echo_json(result)


@captions_group.command("bulk-insert")
@click.option("--parquet", "parquet_paths", multiple=True, required=True)
@click.option("--endpoint-url", default=None)
@click.option("--access-key-env", default="AWS_ACCESS_KEY_ID", show_default=True)
@click.option("--secret-key-env", default="AWS_SECRET_ACCESS_KEY", show_default=True)
@click.option("--allow-lab-http-endpoint", is_flag=True)
@click.option("--model-name-override", default=None)
@click.option("--data-source-override", default=None)
@click.option("--cds-url", required=True)
def caption_bulk_insert(
    parquet_paths: tuple[str, ...],
    endpoint_url: str | None,
    access_key_env: str,
    secret_key_env: str,
    allow_lab_http_endpoint: bool,
    model_name_override: str | None,
    data_source_override: str | None,
    cds_url: str,
) -> None:
    """Submit S3 caption parquet references without retrying POST."""
    access_key, secret_key = _credentials(access_key_env, secret_key_env)
    try:
        result = bulk_insert_caption_parquets(
            BulkInsertCaptionParquetsRequest(
                parquet_paths=parquet_paths,
                access_key=access_key,
                secret_key=secret_key,
                endpoint_url=endpoint_url,
                allow_lab_http_endpoint=allow_lab_http_endpoint,
                model_name_override=model_name_override,
                data_source_override=data_source_override,
            ),
            client_factory=lambda: build_caption_adapter(cds_url),
        )
    except (CaptionCapabilityError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    _echo_json(result)
