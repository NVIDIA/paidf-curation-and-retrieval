# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Embedding analytics subpackage."""

from packages.analytics.embeddings.vectors import (
    cosine_scores,
    l2_normalize,
    load_embeddings_parquet,
    stack_embeddings,
)

__all__ = [
    "cosine_scores",
    "l2_normalize",
    "load_embeddings_parquet",
    "stack_embeddings",
]
