<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Installation

Install Curation and Retrieval so you can run a cookbook. This is the product
install path for customers.

If you are changing the code, do this install first, then continue with
[Developer Docs](../developer/README.md).

## Requirements

Install only what the chosen path needs.

| Item | Minimum |
|------|---------|
| OS | Ubuntu 22.04 or 24.04 LTS (Linux) |
| Docker | 20.10+ |
| NVIDIA Container Toolkit | Installed and working with Docker |
| GPU | NVIDIA data-center GPU (A40 / A100 / H100 class recommended) |
| CPU | 16+ cores |
| RAM | 64GB system RAM |
| Storage | 500GB SSD; add ~90GB for Curator models |
| Python | `>=3.10,<3.13` via [`uv`](https://github.com/astral.sh/uv); 3.12 recommended |
| Make | GNU Make |

| Path | Extra requirements |
|------|-------------------|
| Path B — Data Mining | One GPU is enough for smoke runs; no Curator weights or FFmpeg |
| Path A — Curator | FFmpeg sidecar (`make ffmpeg-install`); `MODELS_DIR` (~90GB+) |
| Full Curator video recipes | 48GB+ VRAM; VLM/LLM endpoints for caption cookbooks |
| Sample clips | NGC CLI 4.10.0+ and an API key — [Samples and Cookbooks](samples-and-cookbooks.md) |

TAO Toolkit DS documents driver `595.45.04` as a floor. Qualify the selected
Curator image independently.

Never put credentials on Make command lines, in cookbook YAML, or in logs.

## Clone and sync

```bash
git clone https://github.com/NVIDIA/paidf-curation-and-retrieval.git
cd paidf-curation-and-retrieval
cp .env.example .env
uv sync --extra dev
docker login nvcr.io
```

This repository **pulls** published vendor images. It does not build or push
vendor images as part of the supported path.

### Cosmos Curator (Path A)

```bash
make pull
make ffmpeg-install           # default: $HOME/cosmos-curator-ffmpeg
make check-setup
make download-models MODELS_DIR=/path/to/models
```

Model tokens and optional SeedVR2 / SAM3 notes:
[Operations: Curator](operations-curator.md).

**Success signals**

- `make check-setup` validates Docker, GPU visibility, and the FFmpeg sidecar
- Models directory populated under `MODELS_DIR`

### Data Mining (Path B)

```bash
make pull-data-mining
make check-data-mining-image
```

**Success signals**

- `docker pull` completes without auth errors
- `make check-data-mining-image` exits 0

### Verify GPU visibility

```bash
docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi
```

If installation fails, see [Troubleshooting](troubleshooting.md).

## Image pins

Authoritative defaults live in the root `Makefile`. Override locally via
`.env` or command-line Make variables.

| Runtime | Makefile variables | Default pattern |
|---------|--------------------|-----------------|
| Cosmos Curator | `COSMOS_CURATOR_REGISTRY`, `COSMOS_CURATOR_TAG` | `nvcr.io/nvidia/cosmos/cosmos-curator:2.3.0` |
| Data Mining | `DATA_MINING_REGISTRY`, `DATA_MINING_TAG` | `nvcr.io/nvidia/tao/tao-toolkit:7.2.0-data-services` |

Product version `1.1.0` (`pyproject.toml`) describes this glue repository.
Container tags are separate pins.

## Credentials

| Need | When |
|------|------|
| NGC / `nvcr.io` Docker login | Pulling Curator and TAO images |
| NGC CLI API key | Downloading VSS sample clips |
| Hugging Face token | Gated Curator weights (for example SAM3) |
| OpenAI-compatible VLM/LLM keys | Caption cookbooks; see [VLM and LLM Endpoints](vlm-llm-endpoints.md) |

Keep secrets in the environment or gitignored `.env` / `local.env`. Do not
put API keys on Make argv.

Hugging Face token file (Curator):

```bash
mkdir -p ~/.config/cosmos_curator
chmod 600 ~/.config/cosmos_curator/hf_token.txt
```

## Limitations

- Cosmos Dataset Search (CDS) is out of scope for this guide.
- Building or pushing vendor images is not supported; pull only.
- There is no in-repo scheduler. Sequence Make targets yourself.
- Make is the public interface; the internal CLI is an implementation detail.
- TMM does not generate embeddings. Run Curator or TAO `embedding` first.
- Metric `l2` is not accepted; use `euclidean`.
- `coco` detections must be a single JSON file; `kitti` must be a label
  directory.
- The repo does not publish a performance SLA. Time a short clip or a few
  parquet rows on your host before you scale.

Optional Curator source builds (`make clone-curator` / `make build`) are
developer-only. See [Developer Docs](../developer/README.md).
