# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dataset Search adapters — online retrieval (CDS / CVDS HTTP API)."""

from adapters.dataset_search.retrieval_adapter import (
    DatasetSearchAdapter,
    normalize_cds_base_url,
    sanitize_document_metadata,
)

__all__ = [
    "DatasetSearchAdapter",
    "normalize_cds_base_url",
    "sanitize_document_metadata",
]
