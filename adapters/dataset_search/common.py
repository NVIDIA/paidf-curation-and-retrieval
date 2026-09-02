# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared Dataset Search adapter utilities."""

from __future__ import annotations

from urllib.parse import urlparse

SUCCESS_JOB_STATES = frozenset({"completed", "finished", "succeeded", "success"})
FAILED_JOB_STATES = frozenset({"cancelled", "canceled", "failed", "error"})
TERMINAL_JOB_STATES = SUCCESS_JOB_STATES | FAILED_JOB_STATES
CDS_COMPATIBLE_EMBEDDING_FAMILY = "ce1"


class BulkIngestUnavailableError(RuntimeError):
    """The required CDS bulk-ingest capability is unavailable."""


class BulkJobPollingTimeout(TimeoutError):
    """CDS did not report a terminal bulk-job state before the deadline."""


def validate_cds_embedding_family(embedding_family: str) -> str:
    """Validate the declared producer family for CDS-bound vectors.

    Vector values do not identify their producer model. CDS compatibility is
    therefore an explicit operator/artifact declaration, not inferred data.
    """
    normalized = str(embedding_family or "").strip().lower()
    if normalized != CDS_COMPATIBLE_EMBEDDING_FAMILY:
        declared = normalized or "missing"
        raise ValueError(
            "CDS bulk insert requires embedding_family='ce1'; "
            f"received {declared!r}. IV2, CLIP, and SigLIP vectors are not CDS-compatible."
        )
    return normalized


def normalize_cds_base_url(base_url: str) -> str:
    """Normalize a CDS/CVDS base URL so API calls land under ``/v1``.

    Accepts ``http://host:8888`` or ``http://host:8888/v1`` (with or without
    trailing slash). Raises if empty.
    """
    if not base_url or not str(base_url).strip():
        raise ValueError("base_url is required")
    cleaned = str(base_url).strip().rstrip("/")

    parsed = urlparse(cleaned)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(f"base_url must be an absolute HTTP(S) URL, got: {base_url!r}")
    if parsed.scheme not in {"http", "https"}:
        raise ValueError(f"base_url must be an absolute HTTP(S) URL, got: {base_url!r}")
    path = (parsed.path or "").rstrip("/")
    root = f"{parsed.scheme}://{parsed.netloc}"
    if path.endswith("/v1"):
        return f"{root}{path}"
    if path in ("", "/"):
        return f"{root}/v1"
    return f"{root}{path}/v1"
