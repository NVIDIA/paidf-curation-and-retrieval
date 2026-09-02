# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Embedding vector helpers shared by text-match and distribution analysis."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

PathLike = str | Path


def l2_normalize(x: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalize; zeros become near-zero rows."""
    if x.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {x.shape}")
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return np.asarray(x / np.maximum(norms, 1e-12), dtype=np.float64)


def stack_embeddings(series: Sequence) -> np.ndarray:
    """Stack a sequence of embedding lists into a float64 matrix."""
    if len(series) == 0:
        raise ValueError("Cannot stack empty embedding series")
    return np.stack([np.asarray(v, dtype=np.float64) for v in series])


def load_embeddings_parquet(
    path: PathLike,
    file_column: str = "file_name",
    embed_column: str = "embedding",
) -> tuple[list[str], np.ndarray, pd.DataFrame]:
    """Load TAO-style embeddings.parquet.

    Returns ``(file_names, L2-normalized matrix, full dataframe)``.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Embeddings parquet not found: {path}")
    df = pd.read_parquet(path)
    for col in (file_column, embed_column):
        if col not in df.columns:
            raise ValueError(f"Missing column {col!r} in {path}; have {list(df.columns)}")
    names = df[file_column].astype(str).tolist()
    matrix = l2_normalize(stack_embeddings(df[embed_column].to_numpy()))
    return names, matrix, df


def cosine_scores(queries: np.ndarray, gallery: np.ndarray) -> np.ndarray:
    """Cosine similarity matrix (queries already L2-normalized).

    ``queries`` shape ``(Q, D)``, ``gallery`` shape ``(N, D)`` → ``(Q, N)``.
    """
    q = l2_normalize(np.asarray(queries, dtype=np.float64))
    g = l2_normalize(np.asarray(gallery, dtype=np.float64))
    if q.ndim == 1:
        q = q[None, :]
    if g.ndim == 1:
        g = g[None, :]
    if q.shape[1] != g.shape[1]:
        raise ValueError(f"Dim mismatch: queries {q.shape} vs gallery {g.shape}")
    return np.asarray(q @ g.T, dtype=np.float64)
