# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cosmos-Embed NIM HTTP client — live text/video embeddings only.

Query-time use (e.g. text-match against a Cosmos-Embed gallery parquet).
Batch embedding at scale belongs in the Data Mining image or Cosmos Curator
pipeline, not this integration layer.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import requests


class CosmosEmbedClientError(RuntimeError):
    """Raised when the NIM returns an error or unexpected payload."""


class CosmosEmbedClient:
    """Thin client for ``POST {base}/v1/embeddings``."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:9000",
        model: str = "nvidia/cosmos-embed1",
        timeout_s: float = 120.0,
        session: requests.Session | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s
        self._session = session or requests.Session()

    def health_ready(self) -> bool:
        try:
            r = self._session.get(f"{self.base_url}/v1/health/ready", timeout=10)
            return r.status_code == 200
        except requests.RequestException:
            return False

    def embed(
        self,
        inputs: Sequence[str],
        request_type: str = "query",
    ) -> list[np.ndarray]:
        """Embed text strings or ``data:video/...;base64,...`` payloads."""
        if not inputs:
            raise ValueError("inputs must be non-empty")
        payload = {
            "input": list(inputs),
            "request_type": request_type,
            "encoding_format": "float",
            "model": self.model,
        }
        try:
            response = self._session.post(
                f"{self.base_url}/v1/embeddings",
                json=payload,
                timeout=self.timeout_s,
            )
        except requests.RequestException as exc:
            raise CosmosEmbedClientError(f"Request failed: {exc}") from exc
        if not response.ok:
            raise CosmosEmbedClientError(f"HTTP {response.status_code}: {response.text[:500]}")
        data = response.json().get("data")
        if not data:
            raise CosmosEmbedClientError(f"Unexpected response: {response.text[:500]}")
        out: list[np.ndarray] = []
        for item in data:
            emb = item.get("embedding")
            if emb is None:
                raise CosmosEmbedClientError("Missing embedding in response item")
            out.append(np.asarray(emb, dtype=np.float64))
        return out

    def embed_texts(self, texts: Sequence[str]) -> list[np.ndarray]:
        """Embed one or more text queries (``request_type=query``).

        Cosmos-Embed1 query mode accepts a single item per request, so
        multi-text inputs are sent sequentially.
        """
        if not texts:
            raise ValueError("inputs must be non-empty")
        if len(texts) == 1:
            return self.embed(list(texts), request_type="query")
        out: list[np.ndarray] = []
        for text in texts:
            out.extend(self.embed([text], request_type="query"))
        return out
