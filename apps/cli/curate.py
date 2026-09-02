# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offline curation use-case: S → diverse subset of B, then enrich for Dataset Search ingest.

Orchestrates Data Mining analytics without coupling to Dataset Search internals.
"""

from __future__ import annotations

from collections.abc import Sequence

from adapters.schema.mapper import enrich_dataset_search_meta, records_to_dataset_search_frame_dicts
from apps.composition import ProductServices
from packages.domain.types import EmbeddingRecord, SubsetSelection


def curate_domain_subset(
    services: ProductServices,
    targets: Sequence[EmbeddingRecord],
    sources: Sequence[EmbeddingRecord],
    *,
    top_n: int = 5,
    backup_candidates: int = 15,
    metric: str = "cosine",
    diversify: bool = False,
    n_diverse: int | None = None,
    n_clusters: int = 8,
) -> dict[str, object]:
    """Run DivKNN uniqueness selection (+ optional farthest-point diversity).

    Returns:
        dict with keys: unique_source_ids, matches, subset (optional),
        dataset_search_rows (enriched for ingest).
    """
    sim = services.batch_similarity.select_similar(
        targets,
        sources,
        top_n=top_n,
        backup_candidates=backup_candidates,
        metric=metric,
        apply_uniqueness=True,
    )

    selected_ids = list(sim.unique_source_ids)
    source_by_id = {r.record_id: r for r in sources}
    selected_records = [source_by_id[i] for i in selected_ids if i in source_by_id]

    subset: SubsetSelection | None = None
    if diversify and selected_records:
        cluster_result = services.clustering.fit_predict(
            selected_records, method="kmeans", n_clusters=min(n_clusters, len(selected_records))
        )
        label_map = {a.record_id: a.label for a in cluster_result.assignments}
        labels = [label_map[r.record_id] for r in selected_records]
        target_n = n_diverse or max(1, len(selected_records) // 2)
        subset = services.diversity.select_diverse(
            selected_records, n_samples=target_n, cluster_labels=labels
        )
        keep = set(subset.record_ids)
        selected_records = [r for r in selected_records if r.record_id in keep]
        selected_ids = [r.record_id for r in selected_records]

    enriched: list[EmbeddingRecord] = []
    for rank, rec in enumerate(selected_records):
        enriched.append(
            enrich_dataset_search_meta(
                rec, selected=True, diversity_rank=rank if diversify else None
            )
        )

    return {
        "unique_source_ids": selected_ids,
        "matches": list(sim.matches),
        "subset": subset,
        "dataset_search_rows": records_to_dataset_search_frame_dicts(enriched),
    }
