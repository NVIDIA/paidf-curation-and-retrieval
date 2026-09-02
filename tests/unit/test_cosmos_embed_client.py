# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for Cosmos embed HTTP client (mocked)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from adapters.cosmos_embed.embed_client import CosmosEmbedClient, CosmosEmbedClientError


class TestCosmosEmbedClient:
    def test_embed_texts_ok(self):
        session = MagicMock()
        response = MagicMock()
        response.ok = True
        response.json.return_value = {"data": [{"embedding": [0.1, 0.2, 0.3]}]}
        session.post.return_value = response
        client = CosmosEmbedClient(base_url="http://example:9000", session=session)
        out = client.embed_texts(["a", "b"])
        assert len(out) == 2
        assert out[0].shape == (3,)
        assert session.post.call_count == 2  # query mode: one item per request
        args, kwargs = session.post.call_args
        assert args[0].endswith("/v1/embeddings")
        assert kwargs["json"]["request_type"] == "query"
        assert kwargs["json"]["input"] == ["b"]

    def test_embed_http_error(self):
        session = MagicMock()
        response = MagicMock()
        response.ok = False
        response.status_code = 500
        response.text = "boom"
        session.post.return_value = response
        client = CosmosEmbedClient(session=session)
        with pytest.raises(CosmosEmbedClientError, match="500"):
            client.embed_texts(["x"])

    def test_empty_inputs(self):
        client = CosmosEmbedClient(session=MagicMock())
        with pytest.raises(ValueError):
            client.embed_texts([])

    def test_health_ready(self):
        session = MagicMock()
        response = MagicMock()
        response.status_code = 200
        session.get.return_value = response
        client = CosmosEmbedClient(session=session)
        assert client.health_ready() is True
