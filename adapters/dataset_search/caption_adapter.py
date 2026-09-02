# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused HTTP adapter for CDS EA external-caption ingestion."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import requests

from adapters.dataset_search.retrieval_adapter import normalize_cds_base_url
from adapters.object_store import validate_s3_endpoint


class CaptionCapabilityError(RuntimeError):
    """The target CDS deployment does not expose a usable caption capability."""


class CaptionAdapter:
    """Call the verified EA caption ingestion and keyword-search contracts."""

    def __init__(self, base_url: str, session: requests.Session | None = None) -> None:
        self._base_url = normalize_cds_base_url(base_url)
        self._session = session or requests.Session()

    def _raise_for_capability(self, response: requests.Response, operation: str) -> None:
        if response.status_code == 404:
            raise CaptionCapabilityError(
                f"CDS caption {operation} is unavailable: "
                "the deployment does not expose the required /captions endpoint"
            )
        if response.status_code == 503:
            raise CaptionCapabilityError(
                f"CDS caption {operation} is unavailable: the EA caption store is not configured"
            )
        response.raise_for_status()

    def stats(self) -> Mapping[str, Any]:
        """Read caption-store statistics as endpoint and metadata-store evidence."""
        response = self._session.request(
            "GET",
            f"{self._base_url}/captions/stats",
            timeout=30,
        )
        self._raise_for_capability(response, "statistics")
        body = response.json()
        if not isinstance(body, dict):
            raise CaptionCapabilityError(
                "CDS caption readiness is unavailable: /captions/stats returned invalid JSON"
            )
        return dict(body)

    def upload_parquet(
        self,
        parquet_path: str | Path,
        *,
        model_name: str = "default",
        data_source: str = "",
    ) -> Mapping[str, Any]:
        """POST one local parquet to `/captions/upload`; submission is not retried."""
        path = Path(parquet_path)
        if not path.is_file():
            raise FileNotFoundError(f"Caption parquet not found: {path}")
        if not model_name.strip():
            raise ValueError("model_name must be non-empty")
        with path.open("rb") as stream:
            response = self._session.request(
                "POST",
                f"{self._base_url}/captions/upload",
                files={"file": (path.name, stream, "application/octet-stream")},
                params={"model_name": model_name, "data_source": data_source},
                timeout=600,
            )
        self._raise_for_capability(response, "upload")
        body = response.json()
        return dict(body) if isinstance(body, dict) else {"result": body}

    def bulk_insert(
        self,
        parquet_paths: Sequence[str],
        *,
        access_key: str | None = None,
        secret_key: str | None = None,
        endpoint_url: str | None = None,
        allow_insecure_endpoint: bool = False,
        model_name_override: str | None = None,
        data_source_override: str | None = None,
    ) -> Mapping[str, Any]:
        """POST S3 parquet references to `/captions/bulk-insert` without retries."""
        if not parquet_paths or any(not str(path).strip() for path in parquet_paths):
            raise ValueError("parquet_paths must contain non-empty values")
        if any(not str(path).startswith("s3://") for path in parquet_paths):
            raise ValueError("caption bulk-insert parquet_paths must use s3:// URIs")
        if bool(access_key) != bool(secret_key):
            raise ValueError("access_key and secret_key must be provided together")
        validate_s3_endpoint(endpoint_url, allow_insecure=allow_insecure_endpoint)

        payload: dict[str, Any] = {"parquet_paths": list(parquet_paths)}
        if access_key:
            payload["access_key"] = access_key
            payload["secret_key"] = secret_key
        if endpoint_url:
            payload["endpoint_url"] = endpoint_url
        if model_name_override:
            payload["model_name_override"] = model_name_override
        if data_source_override:
            payload["data_source_override"] = data_source_override

        response = self._session.request(
            "POST",
            f"{self._base_url}/captions/bulk-insert",
            json=payload,
            timeout=600,
        )
        self._raise_for_capability(response, "bulk insert")
        body = response.json()
        return dict(body) if isinstance(body, dict) else {"result": body}

    def search(
        self,
        query: str,
        *,
        limit: int = 5000,
        data_sources: Sequence[str] | None = None,
    ) -> Mapping[str, Any]:
        """POST the verified `/captions/search` keyword-search contract."""
        normalized_query = str(query or "").strip()
        if not normalized_query:
            raise ValueError("caption search query must be non-empty")
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 50000:
            raise ValueError("caption search limit must be between 1 and 50000")
        if data_sources is not None:
            if isinstance(data_sources, (str, bytes)) or any(
                not str(source).strip() for source in data_sources
            ):
                raise ValueError("caption search data_sources must contain non-empty values")

        payload: dict[str, Any] = {"query": normalized_query, "limit": limit}
        if data_sources:
            payload["data_sources"] = [str(source).strip() for source in data_sources]
        response = self._session.request(
            "POST",
            f"{self._base_url}/captions/search",
            json=payload,
            timeout=60,
        )
        self._raise_for_capability(response, "search")
        body = response.json()
        if not isinstance(body, dict):
            raise CaptionCapabilityError(
                "CDS caption search is unavailable: /captions/search returned invalid JSON"
            )
        clip_ids = body.get("clip_ids")
        count = body.get("count")
        if (
            not isinstance(clip_ids, list)
            or any(not isinstance(clip_id, str) for clip_id in clip_ids)
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count != len(clip_ids)
        ):
            raise CaptionCapabilityError(
                "CDS caption search is unavailable: /captions/search returned an invalid response"
            )
        return {"clip_ids": list(clip_ids), "count": count}
