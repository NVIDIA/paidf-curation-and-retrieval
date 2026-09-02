# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for DivKNN uniqueness — behavior parity with TAO helpers."""

from __future__ import annotations

import pandas as pd
import pytest

from packages.analytics.uniqueness.divknn_uniqueness import (
    ensure_global_top_k_unique,
    ensure_top_n_unique,
    unique_source_ids_from_frame,
)


class TestEnsureTopNUnique:
    def test_keeps_unique_prefix(self):
        row = {
            "top1_file_name": "a",
            "top1_embed": [1],
            "top1_distance": 0.1,
            "top2_file_name": "b",
            "top2_embed": [2],
            "top2_distance": 0.2,
            "top3_file_name": "c",
            "top3_embed": [3],
            "top3_distance": 0.3,
        }
        out = ensure_top_n_unique(row, total_n=3, keep_n=2)
        assert out["top1_file_name"] == "a"
        assert out["top2_file_name"] == "b"

    def test_replaces_duplicate_with_backup(self):
        row = {
            "top1_file_name": "a",
            "top1_embed": [1],
            "top1_distance": 0.1,
            "top2_file_name": "a",  # duplicate
            "top2_embed": [1],
            "top2_distance": 0.15,
            "top3_file_name": "c",
            "top3_embed": [3],
            "top3_distance": 0.3,
        }
        out = ensure_top_n_unique(row, total_n=3, keep_n=2)
        assert out["top1_file_name"] == "a"
        assert out["top2_file_name"] == "c"

    def test_empty_keep_raises_via_index(self):
        # keep_n=0 is degenerate; ensure function does not crash on empty unique lists
        row = {
            "top1_file_name": "a",
            "top1_embed": [1],
            "top1_distance": 0.1,
        }
        out = ensure_top_n_unique(row, total_n=1, keep_n=0)
        assert out["top1_file_name"] == "a"


class TestEnsureGlobalTopKUnique:
    def test_global_dedup_prefers_closer_top1(self):
        df = pd.DataFrame(
            [
                {
                    "top1_file_name": "shared",
                    "top1_embed": [1],
                    "top1_distance": 0.05,
                    "top2_file_name": "x",
                    "top2_embed": [2],
                    "top2_distance": 0.5,
                },
                {
                    "top1_file_name": "shared",
                    "top1_embed": [1],
                    "top1_distance": 0.2,
                    "top2_file_name": "y",
                    "top2_embed": [3],
                    "top2_distance": 0.4,
                },
            ]
        )
        out = ensure_global_top_k_unique(df, total_n=2, keep_k=1)
        # First row (closer) keeps shared; second should swap to backup y
        assert out.iloc[0]["top1_file_name"] == "shared"
        assert out.iloc[1]["top1_file_name"] == "y"

    def test_keep_k_must_be_less_than_total_n(self):
        df = pd.DataFrame([{"top1_file_name": "a", "top1_embed": [1], "top1_distance": 0.1}])
        with pytest.raises(AssertionError, match="keep_k must be less than total_n"):
            ensure_global_top_k_unique(df, total_n=1, keep_k=1)


class TestUniqueSourceIds:
    def test_drop_duplicates(self):
        df = pd.DataFrame(
            {
                "top1_file_name": ["a", "b"],
                "top2_file_name": ["b", "c"],
            }
        )
        ids = unique_source_ids_from_frame(df, keep_n=2)
        assert ids == ["a", "b", "c"]
