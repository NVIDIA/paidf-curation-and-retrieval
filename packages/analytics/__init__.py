# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Analytics package: pure algorithms extracted from TAO (no GPU I/O)."""

from packages.analytics.divknn_select import knn_unique_select
from packages.analytics.text_video_match import (
    match_text_to_videos,
    mean_std_threshold,
    score_gallery_against_texts,
)
from packages.analytics.uniqueness.divknn_uniqueness import (
    apply_uniqueness_pipeline,
    ensure_global_top_k_unique,
    ensure_top_n_unique,
    unique_source_ids_from_frame,
)

__all__ = [
    "apply_uniqueness_pipeline",
    "ensure_global_top_k_unique",
    "ensure_top_n_unique",
    "unique_source_ids_from_frame",
    "knn_unique_select",
    "match_text_to_videos",
    "mean_std_threshold",
    "score_gallery_against_texts",
]
