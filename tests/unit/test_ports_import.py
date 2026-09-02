# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Smoke import of ports (Protocols are structural; keep coverage honest)."""

from __future__ import annotations

from packages.ports import (
    BatchSimilarityPort,
    ClusterAnalyticsPort,
    EmbeddingStorePort,
    IngestPort,
    RetrievalPort,
    SubsetSelectionPort,
)


def test_ports_are_importable():
    assert RetrievalPort is not None
    assert BatchSimilarityPort is not None
    assert ClusterAnalyticsPort is not None
    assert SubsetSelectionPort is not None
    assert EmbeddingStorePort is not None
    assert IngestPort is not None
