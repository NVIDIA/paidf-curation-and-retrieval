# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Internal ``paidf_curation_and_retrieval`` CLI entry module.

Make targets invoke this Click app for Curator pipelines, Dataset Search
handoffs, and TAO Data Services embedding/mining jobs. Operators should use
Make; this module is the glue behind those targets.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import cast

import click

from adapters.cosmos_curator.export_adapter import CuratorExportAdapter, CuratorExportError
from adapters.data_mining.tmm_parquet import TMM_METRICS, TmmParquetError
from adapters.data_mining.tmm_unm import UNM_ALLOCATION_POLICIES, UNM_DETECTION_FORMATS
from adapters.docker_jobs import DockerJobError, default_curator_image
from apps.cli.curate import curate_domain_subset
from apps.cli.dataset_search_cmds import dataset_search_group
from apps.cli.integration_cmds import integration_group
from apps.cli.pipeline_config import validate_curator_pipeline_config
from apps.composition import (
    build_curator_runner,
    build_data_mining_runner,
    build_dataset_search_adapter,
    build_services,
)
from apps.workflows import (
    IngestCuratorExportRequest,
    RunCuratorPipelineRequest,
    RunDataMiningSelectionRequest,
    RunUniqueNeighborMatchingRequest,
    ingest_curator_export,
    run_curator_pipeline,
    run_data_mining_selection,
    run_unique_neighbor_matching,
)
from packages.domain.types import CollectionRef, EmbeddingRecord, SearchQuery


@click.group()
def main() -> None:
    """PAIDF integration layer (pulled images + contracts, no engine source)."""


main.add_command(dataset_search_group)
main.add_command(integration_group)


@main.command("curator-run")
@click.option("--config-file", required=True, type=click.Path(exists=True))
@click.option("--data-dir", default=None, type=click.Path())
@click.option("--models-dir", default=None, type=click.Path())
@click.option("--ffmpeg-dir", default=None, type=click.Path())
@click.option("--image", default=default_curator_image)
@click.option("--dry-run/--no-dry-run", default=False)
def curator_run(config_file, data_dir, models_dir, ffmpeg_dir, image, dry_run) -> None:
    """Run Cosmos Curator batch image — same idea as data-curation ``make run-pipeline``."""
    result = run_curator_pipeline(
        RunCuratorPipelineRequest(
            config_file=config_file,
            data_dir=data_dir,
            models_dir=models_dir,
            ffmpeg_dir=ffmpeg_dir,
            dry_run=dry_run,
        ),
        runner_factory=lambda: build_curator_runner(image=image),
        validate_config=validate_curator_pipeline_config,
    )
    click.echo(json.dumps({"dry_run": result.dry_run, "command": result.command}))


@main.command("ingest-curator")
@click.option(
    "--curator-dir",
    required=True,
    type=click.Path(exists=True),
    help="Dir with iv2_embd_parquet/ or ce1_embd*_parquet/",
)
@click.option("--output-parquet", required=True, type=click.Path(), help="CDS-shaped parquet path")
@click.option(
    "--collection", default=None, help="If set, ingest into CDS (bulk or GA documents fallback)"
)
@click.option("--cds-url", default=None)
@click.option(
    "--embedding-backend",
    default="auto",
    show_default=True,
    type=click.Choice(["auto", "iv2", "ce1"], case_sensitive=False),
    help="Which Curator embedding export to convert (auto detects IV2 or CE1)",
)
@click.option(
    "--convert-only/--ingest",
    default=True,
    help="Only convert parquet (default) or also ingest into CDS",
)
@click.option(
    "--allow-lab-document-fallback",
    is_flag=True,
    help="Lab only: use document ingest if POST /insert-data returns HTTP 404",
)
def ingest_curator(
    curator_dir,
    output_parquet,
    collection,
    cds_url,
    embedding_backend,
    convert_only,
    allow_lab_document_fallback,
) -> None:
    """Convert a Curator export; CDS-bound ingest accepts CE1 only."""
    try:
        result = ingest_curator_export(
            IngestCuratorExportRequest(
                curator_dir=curator_dir,
                output_parquet=output_parquet,
                collection=collection,
                embedding_backend=embedding_backend,
                convert_only=convert_only,
                allow_lab_document_fallback=allow_lab_document_fallback,
            ),
            converter_factory=CuratorExportAdapter,
            client_factory=lambda: build_dataset_search_adapter(cds_url),
        )
    except (CuratorExportError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(json.dumps(result.conversion_payload))

    if result.ingest_payload is not None:
        click.echo(json.dumps(result.ingest_payload))


@main.command("data-mining-select")
@click.option("--data-dir", required=True, type=click.Path(exists=True))
@click.option(
    "--config-file",
    default=None,
    type=click.Path(exists=True),
    help="TAO TMM experiment YAML; when set, YAML owns all mining parameters",
)
@click.option("--target-subdir", default="S", help="Target parquet file or dir under DATA_DIR")
@click.option("--source-subdir", default="B", help="Source parquet file or dir under DATA_DIR")
@click.option(
    "--output-subdir", default="divknn_out", help="Output dir under DATA_DIR for mined.parquet"
)
@click.option("--topn", default=5, type=click.IntRange(min=1))
@click.option(
    "--metric",
    default="cosine",
    type=click.Choice(sorted(TMM_METRICS), case_sensitive=False),
    help="knn_metric passed to tmm nearest_neighbors",
)
@click.option(
    "--embedding-backend",
    default="ce1",
    show_default=True,
    type=click.Choice(["iv2", "ce1", "clip", "siglip"], case_sensitive=False),
    help=(
        "Declared TDM embedding family; IV2/CE1 come from Curator and "
        "CLIP/SigLIP come from TAO DS image embeddings"
    ),
)
@click.option(
    "--gpus",
    default=lambda: os.environ.get("GPUS", "all"),
    help="docker --gpus value (default: all)",
)
@click.option(
    "--shm-size",
    default=lambda: os.environ.get("DATA_MINING_SHM_SIZE", "16g"),
    help="docker --shm-size (default: 16g)",
)
@click.option(
    "--image",
    required=True,
    help="TAO Data Services image reference; Make supplies its configured registry/tag",
)
@click.option("--dry-run/--no-dry-run", default=False)
@click.option(
    "--filter-by-label/--no-filter-by-label",
    default=False,
    help="Pass filter_by_label=true to tmm nearest_neighbors",
)
@click.option(
    "--distance-threshold",
    default=-1.0,
    type=float,
    show_default=True,
    help="Max allowed pair distance for nearest_neighbors (-1 disables)",
)
@click.option(
    "--source-embed-column-name",
    default="embedding",
    show_default=True,
    help="Source parquet embedding column name",
)
@click.option(
    "--target-embed-column-name",
    default="embedding",
    show_default=True,
    help="Target parquet embedding column name",
)
@click.option(
    "--in-process/--docker", default=False, help="Use in-process DivKNN adapter (no GPU image)"
)
def data_mining_select(
    data_dir,
    config_file,
    target_subdir,
    source_subdir,
    output_subdir,
    topn,
    metric,
    embedding_backend,
    gpus,
    shm_size,
    image,
    dry_run,
    filter_by_label,
    distance_threshold,
    source_embed_column_name,
    target_embed_column_name,
    in_process,
) -> None:
    """Run Data Mining via TAO Toolkit DS ``tmm nearest_neighbors``, or in-process CPU."""
    try:
        result = run_data_mining_selection(
            RunDataMiningSelectionRequest(
                data_dir=data_dir,
                config_file=config_file,
                target_subdir=target_subdir,
                source_subdir=source_subdir,
                output_subdir=output_subdir,
                topn=topn,
                metric=metric,
                embedding_backend=embedding_backend,
                dry_run=dry_run,
                filter_by_label=filter_by_label,
                distance_threshold=distance_threshold,
                source_embed_column_name=source_embed_column_name,
                target_embed_column_name=target_embed_column_name,
                in_process=in_process,
            ),
            runner_factory=lambda: build_data_mining_runner(
                image=image,
                gpus=gpus,
                shm_size=shm_size,
            ),
        )
    except (DockerJobError, TmmParquetError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(
        json.dumps(
            {
                "status": result.status,
                "dry_run": result.dry_run,
                "embedding_backend": result.embedding_backend,
                "command": result.command,
                "evidence": result.evidence,
            },
            default=str,
        )
    )


@main.command("data-mining-unique-match")
@click.option("--data-dir", required=True, type=click.Path(exists=True))
@click.option(
    "--config-file",
    default=None,
    type=click.Path(exists=True),
    help="TAO unique_neighbor_matching YAML; when set, YAML owns all mining parameters",
)
@click.option("--target-subdir", default="S", help="Target parquet file or dir under DATA_DIR")
@click.option("--source-subdir", default="B", help="Source parquet file or dir under DATA_DIR")
@click.option(
    "--output-subdir",
    default="unm_out",
    help="Output directory under DATA_DIR for unique_neighbor_matching artifacts",
)
@click.option("--desired-unique-count", default=100, type=click.IntRange(min=1))
@click.option(
    "--allocation-policy",
    default="global",
    type=click.Choice(sorted(UNM_ALLOCATION_POLICIES), case_sensitive=False),
)
@click.option(
    "--metric",
    default="euclidean",
    type=click.Choice(sorted(TMM_METRICS), case_sensitive=False),
    help="distance_metric passed to tmm unique_neighbor_matching",
)
@click.option("--candidate-expansion-factor", default=5, type=click.IntRange(min=1))
@click.option(
    "--source-embedding-column",
    default="embedding",
    show_default=True,
    help="Source parquet embedding column name",
)
@click.option(
    "--target-embedding-column",
    default="embedding",
    show_default=True,
    help="Target parquet embedding column name",
)
@click.option(
    "--source-filepath-column",
    default="filepath",
    show_default=True,
    help="Source parquet filepath column name",
)
@click.option(
    "--target-filepath-column",
    default="filepath",
    show_default=True,
    help="Target parquet filepath column name",
)
@click.option(
    "--exclude-path",
    default=None,
    help="Optional DATA_DIR-relative parquet/list of paths to exclude",
)
@click.option(
    "--source-detection-file",
    default=None,
    help="DATA_DIR-relative detection file (required for class_stratified)",
)
@click.option(
    "--target-detection-file",
    default=None,
    help="DATA_DIR-relative detection file (required for class_stratified)",
)
@click.option(
    "--detection-format",
    default=None,
    type=click.Choice(sorted(UNM_DETECTION_FORMATS), case_sensitive=False),
    help="Detection annotation format (required for class_stratified)",
)
@click.option(
    "--rare-class-list",
    default="",
    help="Comma-separated rare class names (required for class_stratified)",
)
@click.option(
    "--save-embeddings/--no-save-embeddings",
    default=False,
    help="Ask TAO UNM to persist selected embeddings",
)
@click.option(
    "--visualize/--no-visualize",
    default=False,
    help="Ask TAO UNM to emit visualization artifacts",
)
@click.option(
    "--embedding-backend",
    default="ce1",
    show_default=True,
    type=click.Choice(["iv2", "ce1", "clip", "siglip"], case_sensitive=False),
)
@click.option(
    "--gpus",
    default=lambda: os.environ.get("GPUS", "all"),
    help="docker --gpus value (default: all)",
)
@click.option(
    "--shm-size",
    default=lambda: os.environ.get("DATA_MINING_SHM_SIZE", "16g"),
    help="docker --shm-size (default: 16g)",
)
@click.option(
    "--image",
    required=True,
    help="TAO Data Services image reference; Make supplies its configured registry/tag",
)
@click.option("--dry-run/--no-dry-run", default=False)
def data_mining_unique_match(
    data_dir,
    config_file,
    target_subdir,
    source_subdir,
    output_subdir,
    desired_unique_count,
    allocation_policy,
    metric,
    candidate_expansion_factor,
    source_embedding_column,
    target_embedding_column,
    source_filepath_column,
    target_filepath_column,
    exclude_path,
    source_detection_file,
    target_detection_file,
    detection_format,
    rare_class_list,
    save_embeddings,
    visualize,
    embedding_backend,
    gpus,
    shm_size,
    image,
    dry_run,
) -> None:
    """Run Data Mining via TAO Toolkit DS ``tmm unique_neighbor_matching``."""
    try:
        result = run_unique_neighbor_matching(
            RunUniqueNeighborMatchingRequest(
                data_dir=data_dir,
                config_file=config_file,
                target_subdir=target_subdir,
                source_subdir=source_subdir,
                output_subdir=output_subdir,
                desired_unique_count=desired_unique_count,
                allocation_policy=allocation_policy,
                metric=metric,
                candidate_expansion_factor=candidate_expansion_factor,
                source_embedding_column=source_embedding_column,
                target_embedding_column=target_embedding_column,
                source_filepath_column=source_filepath_column,
                target_filepath_column=target_filepath_column,
                exclude_path=exclude_path,
                source_detection_file=source_detection_file,
                target_detection_file=target_detection_file,
                detection_format=detection_format,
                rare_class_list=rare_class_list,
                save_embeddings=save_embeddings,
                visualize=visualize,
                embedding_backend=embedding_backend,
                dry_run=dry_run,
            ),
            runner_factory=lambda: build_data_mining_runner(
                image=image,
                gpus=gpus,
                shm_size=shm_size,
            ),
        )
    except (DockerJobError, TmmParquetError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(
        json.dumps(
            {
                "status": result.status,
                "dry_run": result.dry_run,
                "embedding_backend": result.embedding_backend,
                "command": result.command,
                "evidence": result.evidence,
            },
            default=str,
        )
    )


@main.command("search")
@click.option("--collection", required=True)
@click.option("--query", required=True)
@click.option("--top-k", default=10, type=int)
@click.option("--cds-url", default=None)
def search(collection, query, top_k, cds_url) -> None:
    """Text search via Dataset Search shortcut for ``ds search --text``."""
    client = build_dataset_search_adapter(cds_url)
    hits = client.search(
        CollectionRef(collection_id=collection),
        SearchQuery(text=query, top_k=top_k),
    )
    click.echo(
        json.dumps([{"id": h.record_id, "score": h.score, "meta": dict(h.metadata)} for h in hits])
    )


@main.command("demo-curate")
@click.option("--diversify/--no-diversify", default=False)
def demo_curate(diversify: bool) -> None:
    """In-process uniqueness demo (no Docker) — verifies selection algorithms."""
    services = build_services("http://localhost:8000", use_gpu_knn=False)
    targets = [
        EmbeddingRecord("t0", [1.0, 0.0, 0.0]),
        EmbeddingRecord("t1", [0.0, 1.0, 0.0]),
    ]
    sources = [
        EmbeddingRecord("b0", [0.99, 0.01, 0.0]),
        EmbeddingRecord("b1", [0.98, 0.02, 0.0]),
        EmbeddingRecord("b2", [0.01, 0.99, 0.0]),
        EmbeddingRecord("b3", [0.0, 0.0, 1.0]),
    ]
    result = curate_domain_subset(
        services,
        targets,
        sources,
        top_n=1,
        backup_candidates=2,
        diversify=diversify,
        n_diverse=2,
    )
    dataset_search_rows = cast(list[object], result["dataset_search_rows"])
    click.echo(
        json.dumps(
            {
                "unique_source_ids": result["unique_source_ids"],
                "n_dataset_search_rows": len(dataset_search_rows),
            }
        )
    )


@main.command("text-match")
@click.option("--gallery-parquet", required=True, type=click.Path(exists=True))
@click.option("--query", "queries", multiple=True, required=True, help="Text query (repeatable)")
@click.option("--output-dir", required=True, type=click.Path())
@click.option("--video-dir", default=None, type=click.Path(exists=True))
@click.option(
    "--cosmos-embed-url", default="http://127.0.0.1:9000", help="Cosmos-Embed NIM base URL"
)
@click.option("--mode", type=click.Choice(["threshold", "top_k"]), default="threshold")
@click.option("--top-k", default=50, type=int)
@click.option("--k-std", default=1.0, type=float, help="threshold = mean + k_std * std")
@click.option("--threshold", default=None, type=float, help="Override auto threshold")
@click.option("--reduce", type=click.Choice(["max", "mean"]), default="max")
@click.option("--copy-clips/--no-copy-clips", default=True)
def text_match(
    gallery_parquet,
    queries,
    output_dir,
    video_dir,
    cosmos_embed_url,
    mode,
    top_k,
    k_std,
    threshold,
    reduce,
    copy_clips,
) -> None:
    """Rank / select videos by Cosmos-Embed text↔video similarity (gallery must be Cosmos-Embed vectors)."""
    from apps.cli.analytics_cmds import run_text_video_match_cli

    manifest = run_text_video_match_cli(
        gallery_parquet=Path(gallery_parquet),
        queries=list(queries),
        output_dir=Path(output_dir),
        video_dir=Path(video_dir) if video_dir else None,
        cosmos_embed_url=cosmos_embed_url,
        mode=mode,
        top_k=top_k,
        k_std=k_std,
        threshold=threshold,
        reduce=reduce,
        copy_clips=copy_clips,
    )
    click.echo(json.dumps(manifest, indent=2))


@main.command("plot-distribution")
@click.option(
    "--target-parquet", required=True, type=click.Path(exists=True), help="KPI / S embeddings"
)
@click.option(
    "--source-parquet", required=True, type=click.Path(exists=True), help="Lake / B embeddings"
)
@click.option("--output-dir", required=True, type=click.Path())
@click.option(
    "--cosmos-embed-url", default="http://127.0.0.1:9000", help="Cosmos-Embed NIM base URL"
)
@click.option("--query", "queries", multiple=True, help="Optional text queries for text-sim plots")
@click.option("--skip-umap/--umap", default=False)
@click.option("--skip-tsne/--tsne", default=False)
def plot_distribution(
    target_parquet, source_parquet, output_dir, cosmos_embed_url, queries, skip_umap, skip_tsne
) -> None:
    """PCA / t-SNE / UMAP / distance plots for S vs B embedding spaces."""
    from apps.cli.analytics_cmds import run_distribution_plots_cli

    summary = run_distribution_plots_cli(
        target_parquet=Path(target_parquet),
        source_parquet=Path(source_parquet),
        output_dir=Path(output_dir),
        cosmos_embed_url=cosmos_embed_url,
        text_queries=list(queries),
        skip_umap=skip_umap,
        skip_tsne=skip_tsne,
    )
    click.echo(json.dumps(summary, indent=2))


@main.command("divknn-select")
@click.option("--target-parquet", required=True, type=click.Path(exists=True))
@click.option("--source-parquet", required=True, type=click.Path(exists=True))
@click.option("--video-dir", required=True, type=click.Path(exists=True))
@click.option("--output-dir", required=True, type=click.Path())
@click.option("--topn", default=3, type=int)
@click.option("--backup", default=15, type=int)
@click.option("--target-count", default=50, type=int)
def divknn_select_cmd(
    target_parquet, source_parquet, video_dir, output_dir, topn, backup, target_count
) -> None:
    """In-process DivKNN uniqueness selection + copy clips (CPU)."""
    from apps.cli.analytics_cmds import run_divknn_select_cli

    manifest = run_divknn_select_cli(
        target_parquet=Path(target_parquet),
        source_parquet=Path(source_parquet),
        output_dir=Path(output_dir),
        video_dir=Path(video_dir),
        top_n=topn,
        backup=backup,
        target_count=target_count,
    )
    click.echo(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
