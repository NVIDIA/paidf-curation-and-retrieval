# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLI helpers for text↔video match and embedding distribution plots."""

from __future__ import annotations

import json
import shutil
from collections.abc import Sequence
from pathlib import Path, PurePosixPath, PureWindowsPath

import click
import numpy as np
import pandas as pd

from adapters.cosmos_embed.embed_client import CosmosEmbedClient, CosmosEmbedClientError
from packages.analytics.distribution_plots import (
    plot_centroid_scatter,
    plot_distance_histograms,
    plot_projection_s_vs_b,
    plot_text_similarity,
    write_summary_json,
)
from packages.analytics.divknn_select import knn_unique_select
from packages.analytics.embedding_distribution import (
    compute_distance_stats,
    project_pca,
    project_tsne,
    project_umap,
    text_similarity_to_gallery,
)
from packages.analytics.embeddings.vectors import load_embeddings_parquet
from packages.analytics.text_video_match import match_text_to_videos


def _safe_clip_basename(name: str) -> str:
    posix_path = PurePosixPath(name)
    windows_path = PureWindowsPath(name)
    if (
        not name
        or posix_path.is_absolute()
        or windows_path.is_absolute()
        or posix_path.name != name
        or windows_path.name != name
        or ".." in posix_path.parts
        or ".." in windows_path.parts
    ):
        raise click.ClickException(
            f"Unsafe clip file_name in parquet metadata: {name!r}; expected a basename"
        )
    return name


def _resolve_clip_source(
    name: str,
    video_dir: Path,
    *,
    mapped_path: str | None = None,
) -> tuple[str, Path]:
    safe_name = _safe_clip_basename(name)
    if mapped_path:
        mapped = Path(mapped_path)
        if mapped.is_file():
            return safe_name, mapped
    direct = video_dir / safe_name
    if direct.is_file():
        return safe_name, direct
    matches = sorted(path for path in video_dir.rglob(safe_name) if path.is_file())
    if not matches:
        raise click.ClickException(f"Selected clip not found: {safe_name}")
    if len(matches) > 1:
        raise click.ClickException(
            f"Selected clip is ambiguous: {safe_name} matched {len(matches)} files"
        )
    return safe_name, matches[0]


def _copy_selected(
    file_names: Sequence[str],
    video_dir: Path,
    output_clips_dir: Path,
    scores: np.ndarray | None = None,
    all_names: Sequence[str] | None = None,
) -> list[dict]:
    output_clips_dir.mkdir(parents=True, exist_ok=True)
    score_map = {}
    if scores is not None and all_names is not None:
        score_map = {str(n): float(s) for n, s in zip(all_names, scores, strict=True)}
    resolved = [
        (rank, *_resolve_clip_source(str(name), video_dir))
        for rank, name in enumerate(file_names, start=1)
    ]
    copied = []
    for rank, safe_name, src in resolved:
        dst = output_clips_dir / safe_name
        shutil.copy2(src, dst)
        row = {
            "rank": rank,
            "file_name": safe_name,
            "source_path": str(src),
            "dest_path": str(dst),
        }
        if safe_name in score_map:
            row["score"] = score_map[safe_name]
        copied.append(row)
    return copied


def run_text_video_match_cli(
    gallery_parquet: Path,
    queries: Sequence[str],
    output_dir: Path,
    video_dir: Path | None,
    cosmos_embed_url: str,
    mode: str,
    top_k: int,
    k_std: float,
    threshold: float | None,
    reduce: str,
    copy_clips: bool,
) -> dict:
    names, gallery, _df = load_embeddings_parquet(gallery_parquet)
    client = CosmosEmbedClient(base_url=cosmos_embed_url)
    if not client.health_ready():
        raise click.ClickException(f"Cosmos-Embed NIM not ready at {cosmos_embed_url}")
    try:
        text_embs = client.embed_texts(list(queries))
    except CosmosEmbedClientError as exc:
        raise click.ClickException(str(exc)) from exc

    result = match_text_to_videos(
        names,
        gallery,
        text_embs,
        query_labels=[f"q{i + 1}" for i in range(len(queries))],
        reduce=reduce,  # type: ignore[arg-type]
        mode=mode,  # type: ignore[arg-type]
        k_std=k_std,
        top_k=top_k,
        threshold=threshold,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    ranked = pd.DataFrame(
        {
            "rank": np.arange(1, len(names) + 1),
            "file_name": [names[i] for i in np.argsort(-result.scores)],
            "score": np.sort(result.scores)[::-1],
        }
    )
    ranked.to_csv(output_dir / "all_videos_ranked_by_text.csv", index=False)

    matched_names = result.matched_file_names
    matched_df = pd.DataFrame(
        {
            "rank": np.arange(1, len(matched_names) + 1),
            "file_name": matched_names,
            "score": [float(result.scores[i]) for i in result.matched_indices],
        }
    )
    matched_df.to_csv(output_dir / "matched_videos.csv", index=False)

    copied: list[dict] = []
    if copy_clips:
        if video_dir is None:
            raise click.ClickException("--video-dir required when --copy-clips is set")
        clips_dir = output_dir / "selected_clips"
        copied = _copy_selected(
            matched_names,
            video_dir,
            clips_dir,
            scores=result.scores,
            all_names=names,
        )

    manifest = {
        "method": "text_video_match",
        "queries": list(queries),
        "mode": mode,
        "reduce": reduce,
        "threshold": result.threshold,
        "k_std": k_std if mode == "threshold" else None,
        "top_k": top_k if mode == "top_k" else None,
        "gallery_n": len(names),
        "matched_count": len(matched_names),
        "copied": len(copied),
        "cosmos_embed_url": cosmos_embed_url,
        "files": copied,
    }
    (output_dir / "selection_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def run_distribution_plots_cli(
    target_parquet: Path,
    source_parquet: Path,
    output_dir: Path,
    cosmos_embed_url: str | None,
    text_queries: Sequence[str],
    skip_umap: bool,
    skip_tsne: bool,
) -> dict:
    s_names, xs, _ = load_embeddings_parquet(target_parquet)
    b_names, xb, _ = load_embeddings_parquet(source_parquet)
    output_dir.mkdir(parents=True, exist_ok=True)

    stats = compute_distance_stats(xs, xb)
    plot_distance_histograms(stats, output_dir / "01_distance_histograms.png")
    plot_centroid_scatter(stats, output_dir / "02_centroid_analysis.png")

    pca = project_pca(xs, xb, n_components=3)
    plot_projection_s_vs_b(pca, output_dir / "06_pca_pc1_pc2.png", dims=(0, 1))
    if pca.coords.shape[1] >= 3:
        plot_projection_s_vs_b(pca, output_dir / "06_pca_pc1_pc3.png", dims=(0, 2))

    if not skip_tsne:
        tsne = project_tsne(xs, xb)
        plot_projection_s_vs_b(tsne, output_dir / "05_tsne_S_vs_B.png")

    if not skip_umap:
        try:
            um = project_umap(xs, xb)
            plot_projection_s_vs_b(um, output_dir / "04_umap_S_vs_B.png")
        except ImportError as exc:
            click.echo(f"Skipping UMAP: {exc}", err=True)

    extra: dict[str, object] = {
        "n_S": len(s_names),
        "n_B": len(b_names),
        "pca_explained_pct": pca.explained_variance_pct,
        "text_queries": list(text_queries),
    }

    if text_queries and cosmos_embed_url:
        client = CosmosEmbedClient(base_url=cosmos_embed_url)
        if client.health_ready():
            try:
                text_embs = client.embed_texts(list(text_queries))
                # Use first query for S/B hist; if 2+, also collision vs normal style
                s_sim = text_similarity_to_gallery(xs, text_embs[0])
                b_sim = text_similarity_to_gallery(xb, text_embs[0])
                plot_text_similarity(
                    s_sim,
                    b_sim,
                    output_dir / "03_text_query_similarity.png",
                    xlabel=f"cosine to: {text_queries[0][:48]}",
                )
                extra["S_mean_sim_text_q1"] = float(s_sim.mean())
                extra["B_mean_sim_text_q1"] = float(b_sim.mean())
            except CosmosEmbedClientError as exc:
                click.echo(f"Skipping text plots: {exc}", err=True)
        else:
            click.echo(
                f"Skipping text plots: Cosmos-Embed NIM not ready at {cosmos_embed_url}", err=True
            )

    write_summary_json(output_dir / "summary.json", stats, extra=extra)
    return extra


def run_divknn_select_cli(
    target_parquet: Path,
    source_parquet: Path,
    output_dir: Path,
    video_dir: Path,
    top_n: int,
    backup: int,
    target_count: int,
) -> dict:
    t_names, t_emb, _ = load_embeddings_parquet(target_parquet)
    s_names, s_emb, s_df = load_embeddings_parquet(source_parquet)
    unique = knn_unique_select(t_names, t_emb, s_names, s_emb, top_n=top_n, backup=backup)

    if len(unique) < target_count:
        # Fill by mean distance to all targets
        t = t_emb
        s = s_emb
        mean_dist = np.sqrt(np.maximum(0.0, 2.0 - 2.0 * (s @ t.T).mean(axis=1)))
        have = set(unique)
        for idx in np.argsort(mean_dist):
            name = s_names[int(idx)]
            if name in have:
                continue
            unique.append(name)
            have.add(name)
            if len(unique) >= target_count:
                break

    selected = unique[:target_count]
    output_dir.mkdir(parents=True, exist_ok=True)
    clips_dir = output_dir / "selected_clips"
    # Prefer file_path from parquet when present and exists
    path_map = {}
    if "file_path" in s_df.columns:
        for _, row in s_df.iterrows():
            path_map[str(row["file_name"])] = str(row["file_path"])

    resolved = [
        (rank, *_resolve_clip_source(name, video_dir, mapped_path=path_map.get(name)))
        for rank, name in enumerate(selected, start=1)
    ]
    copied = []
    for rank, name, src in resolved:
        clips_dir.mkdir(parents=True, exist_ok=True)
        dst = clips_dir / src.name
        shutil.copy2(src, dst)
        copied.append(
            {"rank": rank, "file_name": name, "source_path": str(src), "dest_path": str(dst)}
        )

    manifest = {
        "method": "divknn_unique_select",
        "top_n": top_n,
        "backup": backup,
        "target_count": target_count,
        "unique_after_divknn": len(unique),
        "copied": len(copied),
        "files": copied,
    }
    (output_dir / "selection_manifest.json").write_text(json.dumps(manifest, indent=2))
    pd.DataFrame({"file_name": [c["file_name"] for c in copied]}).to_parquet(
        output_dir / "unique_selected_files.parquet", index=False
    )
    return manifest
