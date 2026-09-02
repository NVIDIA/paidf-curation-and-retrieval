# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure uniqueness helpers extracted from Data Mining DivKNN.

Behavior matches ``tao_data_curation.DivKNN.div_knn``:
- row-wise uniqueness within each query's top-N
- global uniqueness across queries (nearest top1_distance wins)

No cuML/cuDF dependency — safe for unit tests and the product domain layer.
GPU kNN remains in the TAO adapter; this module only post-processes neighbors.
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping, Sequence
from typing import Any

import pandas as pd


def ensure_top_n_unique(
    row: MutableMapping[str, Any],
    total_n: int,
    keep_n: int,
) -> MutableMapping[str, Any]:
    """Ensure unique values in topn columns by replacing duplicates with backups.

    Same algorithm as ``tao_data_curation.DivKNN.div_knn.ensure_top_n_unique``.
    """
    top_files = [row[f"top{i}_file_name"] for i in range(1, total_n + 1)]
    top_embeds = [row[f"top{i}_embed"] for i in range(1, total_n + 1)]
    top_distances = [row[f"top{i}_distance"] for i in range(1, total_n + 1)]

    unique_top_files: list[Any] = []
    unique_top_embeds: list[Any] = []
    unique_top_distances: list[Any] = []

    for i in range(keep_n):
        if top_files[i] not in unique_top_files:
            unique_top_files.append(top_files[i])
            unique_top_embeds.append(top_embeds[i])
            unique_top_distances.append(top_distances[i])
        else:
            for j in range(keep_n, total_n):
                if top_files[j] not in unique_top_files:
                    unique_top_files.append(top_files[j])
                    unique_top_embeds.append(top_embeds[j])
                    unique_top_distances.append(top_distances[j])
                    break

    for i in range(keep_n):
        if i < len(unique_top_files):
            row[f"top{i + 1}_file_name"] = unique_top_files[i]
            row[f"top{i + 1}_embed"] = unique_top_embeds[i]
            row[f"top{i + 1}_distance"] = unique_top_distances[i]

    return row


def ensure_global_top_k_unique(
    df: pd.DataFrame,
    total_n: int,
    keep_k: int,
) -> pd.DataFrame:
    """Ensure top-K results are globally unique across the DataFrame.

    Same algorithm as ``tao_data_curation.DivKNN.div_knn.ensure_global_top_k_unique``.
    Rows sorted by ``top1_distance`` so the closest match keeps the item.
    """
    if keep_k >= total_n:
        raise AssertionError("keep_k must be less than total_n")

    df = df.sort_values("top1_distance").reset_index(drop=True)
    seen: set[Any] = set()

    for idx, row in df.iterrows():
        for i in range(keep_k):
            cur_name = row[f"top{i + 1}_file_name"]
            if pd.isna(cur_name) or cur_name in seen:
                for j in range(keep_k, total_n):
                    alt_name = row.get(f"top{j + 1}_file_name")
                    if pd.notna(alt_name) and alt_name not in seen:
                        for col_type in ("file_name", "embed", "distance"):
                            df.at[idx, f"top{i + 1}_{col_type}"] = row[f"top{j + 1}_{col_type}"]
                            df.at[idx, f"top{j + 1}_{col_type}"] = float("nan")
                        seen.add(alt_name)
                        break
            else:
                seen.add(cur_name)

    return df


def apply_uniqueness_pipeline(
    neighbor_rows: Sequence[Mapping[str, Any]],
    total_n: int,
    keep_n: int,
) -> pd.DataFrame:
    """Row-wise then global uniqueness — the DivKNN post-process pipeline."""
    df = pd.DataFrame(list(neighbor_rows))
    df = df.apply(
        lambda row: ensure_top_n_unique(row.to_dict(), total_n, keep_n),
        axis=1,
        result_type="expand",
    )
    return ensure_global_top_k_unique(df, total_n, keep_n)


def unique_source_ids_from_frame(df: pd.DataFrame, keep_n: int) -> list[str]:
    """Melt top*_file_name columns and drop duplicates (DivKNN final list)."""
    cols = [f"top{i}_file_name" for i in range(1, keep_n + 1)]
    present = [c for c in cols if c in df.columns]
    melted = df[present].melt(value_name="file_name").dropna(subset=["file_name"])
    return [str(value) for value in melted["file_name"].drop_duplicates().tolist()]
