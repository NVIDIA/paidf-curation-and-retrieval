# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cosmos-Embed NIM adapter — live text/video embeddings only."""

from adapters.cosmos_embed.embed_client import CosmosEmbedClient, CosmosEmbedClientError

__all__ = ["CosmosEmbedClient", "CosmosEmbedClientError"]
