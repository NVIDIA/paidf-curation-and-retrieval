<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Curation and Retrieval — User Guide (v1.1.0)

How to **run** Physical AI Data Factory — Curation and Retrieval: install it,
pick a cookbook, pull images, and read the outputs.

This guide is for **external customers**. If you are changing the code, follow
this guide once so you can run a cookbook, then switch to
[Developer Docs](../developer/README.md).

Typical commands:

- `make pull` / `make pull-data-mining` — pull runtime images
- `make run-pipeline` — run Cosmos Curator
- `make run-data-mining-select` — run nearest neighbors

Read the pages in this order the first time. After that, use the table as a
lookup.

1. [Installation](installation.md) — clone, host requirements, pull images
2. [Samples and Cookbooks](samples-and-cookbooks.md) — download VSS clips
3. [Getting Started](getting-started.md) — Path A Curator, Path B mining
4. [VLM and LLM Endpoints](vlm-llm-endpoints.md) — only for full caption recipes

If something fails, use [Troubleshooting](troubleshooting.md).

## Start here by goal

| Goal | Page |
|------|------|
| First end-to-end run | [Getting Started](getting-started.md) |
| Clone, sync, host requirements, and limitations | [Installation](installation.md) |
| Download sample clips and pick a cookbook | [Samples and Cookbooks](samples-and-cookbooks.md) |
| Run Cosmos Curator | [Operations: Curator](operations-curator.md) |
| Run embeddings and Data Mining | [Operations: Data Mining](operations-tao-mining.md) |
| Wire VLM and LLM endpoints | [VLM and LLM Endpoints](vlm-llm-endpoints.md) |
| Fix a failed run | [Troubleshooting](troubleshooting.md) |

## What you can run from this guide

- Cosmos Curator video and image curation
- Image and text embeddings
- Data Mining nearest-neighbor selection
- Data Mining unique-neighbor matching

Make is the public interface. An internal helper CLI exists for glue; you do
not need it for the customer path.

## Related material

- Repository hub: [Project README](../../README.md)
- Changing the code: [Developer Docs](../developer/README.md)
- Cookbook index: [Cookbooks](../../cookbook/README.md)
- Release history: [Changelog](../../CHANGELOG.md)
