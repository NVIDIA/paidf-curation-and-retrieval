<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Getting Started

Three paths. The public interface is **Make-only**.

| Path | What | Notes |
|------|------|-------|
| **A — Curator** | Video/image curation | Needs models + FFmpeg. First-run: `split-minimal.yaml` |
| **B — Data Mining** | Nearest neighbors on S/B parquets | No Curator models; three still images are enough |
| **C — Curation + Mining** | Path A embeddings → Path B mining | Same dimension + matching `TDM_EMBEDDING_BACKEND` |

Prerequisites and image pulls: [Installation](installation.md). Sample clips:
[Samples and Cookbooks](samples-and-cookbooks.md).

## Path A — Cosmos Curator

Download the NGC VSS sample pack and copy clips into each cookbook `videos/`
folder first. First-run uses `split-minimal.yaml` on the first clip in
`videos/` (`limit: 1`). It does **not** enable captioning, classifier, SAM3,
or enhance.

```bash
make pull
make ffmpeg-install
make check-setup
make download-models MODELS_DIR=/path/to/models

make run-pipeline \
  CONFIG_FILE=cookbook/traffic-video-analytics/split-minimal.yaml \
  DATA_DIR=$PWD/cookbook/traffic-video-analytics \
  MODELS_DIR=/path/to/models \
  FFMPEG_DIR=$HOME/cosmos-curator-ffmpeg
```

**Success:** `cookbook/traffic-video-analytics/output/split-minimal/ce1_embd_224p_parquet/`
exists. On a shared GPU host, pass `GPUS='"device=N"'`.

Full caption recipes: [VLM and LLM Endpoints](vlm-llm-endpoints.md). Next:
[Operations: Curator](operations-curator.md).

## Path B — Data Mining

| File | Role |
|------|------|
| `S.parquet` | Targets (queries) |
| `B.parquet` | Candidates (pool) |

Copy any three JPEGs and keep the filenames in the JSON below.

```bash
make pull-data-mining
make check-data-mining-image

mkdir -p data/tao-mine/images
cp /path/to/query.jpg      data/tao-mine/images/query.jpg
cp /path/to/candidate-1.jpg data/tao-mine/images/candidate-1.jpg
cp /path/to/candidate-2.jpg data/tao-mine/images/candidate-2.jpg

printf '%s\n' '[{"filepath":"images/query.jpg"}]' \
  > data/tao-mine/S.json
printf '%s\n' \
  '[{"filepath":"images/candidate-1.jpg"},{"filepath":"images/candidate-2.jpg"}]' \
  > data/tao-mine/B.json
```

Create S (queries):

```bash
make image-embeddings-build \
  DATA_DIR=$PWD/data/tao-mine \
  IMAGE_EMBEDDING_JSON=$PWD/data/tao-mine/S.json \
  IMAGE_EMBEDDING_INPUT=$PWD/data/tao-mine/S-input.parquet
make run-image-embeddings \
  DATA_DIR=$PWD/data/tao-mine \
  IMAGE_EMBEDDING_INPUT=$PWD/data/tao-mine/S-input.parquet \
  IMAGE_EMBEDDING_OUTPUT=$PWD/data/tao-mine/S.parquet \
  IMAGE_EMBEDDING_MODEL_TYPE=clip \
  IMAGE_EMBEDDING_MODEL=openai/clip-vit-base-patch32
```

Create B (candidate pool):

```bash
make image-embeddings-build \
  DATA_DIR=$PWD/data/tao-mine \
  IMAGE_EMBEDDING_JSON=$PWD/data/tao-mine/B.json \
  IMAGE_EMBEDDING_INPUT=$PWD/data/tao-mine/B-input.parquet
make run-image-embeddings \
  DATA_DIR=$PWD/data/tao-mine \
  IMAGE_EMBEDDING_INPUT=$PWD/data/tao-mine/B-input.parquet \
  IMAGE_EMBEDDING_OUTPUT=$PWD/data/tao-mine/B.parquet \
  IMAGE_EMBEDDING_MODEL_TYPE=clip \
  IMAGE_EMBEDDING_MODEL=openai/clip-vit-base-patch32
```

Mine:

```bash
make run-data-mining-select \
  DATA_DIR=$PWD/data/tao-mine \
  TARGET_SUBDIR=S.parquet \
  SOURCE_SUBDIR=B.parquet \
  TOPN=1 \
  TDM_EMBEDDING_BACKEND=clip \
  OUTPUT_SUBDIR=out
```

**Success:** `data/tao-mine/out/mined.parquet` exists.

Next: [Operations: Data Mining](operations-tao-mining.md).

## Path C — Curation + Mining

Curator generates video embedding parquets (IV2 / CE1). Stage them as
`S.parquet` / `B.parquet` before mining.

```bash
mkdir -p data/curated-mine
cp /path/to/ce1_embd_224p_parquet/query.parquet data/curated-mine/S.parquet
cp /path/to/ce1_embd_224p_parquet/pool.parquet  data/curated-mine/B.parquet

make run-data-mining-select \
  DATA_DIR=$PWD/data/curated-mine \
  TARGET_SUBDIR=S.parquet \
  SOURCE_SUBDIR=B.parquet \
  TOPN=5 \
  TDM_EMBEDDING_BACKEND=ce1 \
  OUTPUT_SUBDIR=out
```

Use `iv2` when Curator used InternVideo2.

**Success:** `data/curated-mine/out/mined.parquet` exists. Dimension mismatches:
[Troubleshooting](troubleshooting.md).
