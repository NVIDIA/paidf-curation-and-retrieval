<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Troubleshooting

Glue validates inputs before expensive GPU work when it can. Treat every
nonzero exit as failure.

## Preflight failures

| Symptom | Likely cause | What to do |
|---------|--------------|------------|
| `DATA_DIR= required` | Missing Make variable | Pass `DATA_DIR=$PWD/...` |
| Path not contained in `DATA_DIR` | File outside mount root | Move data under `DATA_DIR` |
| Namespace / parquet empty | Empty S or B | Check row counts |
| Non-finite embedding / NaN | Corrupt vectors | Regenerate embeddings |
| Dimension mismatch between S and B | Different model families mixed | Rebuild both sides; set matching `TDM_EMBEDDING_BACKEND` |
| Reserved `embedding` column on embedding **input** | Input already has embeddings | Use a fresh input parquet |
| Missing `text` column | Text-embeddings input wrong schema | Add `text` |
| `class_stratified` without detections | Incomplete UNM knobs | Set detection files, format, and `RARE_CLASS_LIST` |
| COCO detection is a directory | Wrong shape for `DETECTION_FORMAT=coco` | Pass a single JSON **file** |
| KITTI detection is a file / empty dir | Wrong shape for `kitti` | Pass a directory of `.txt` labels |
| Invalid metric `l2` | Unsupported alias | Use `euclidean` |
| Image not found locally | Pull skipped | `make pull-data-mining` |

## Docker and GPU

**Docker permission denied**

```bash
sudo usermod -aG docker $USER && newgrp docker
```

**GPU not visible in container**

```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi
```

Install or repair NVIDIA Container Toolkit if the second command fails.

**Invalid GPU device** — `GPUS='"device=9999"'` fails. Use `GPUS=all` or a
valid device index.

## Cosmos Curator

**Curator skips already-processed videos** — delete or change
`output_clip_path`.

**VLM / model load failures** — check HF token file permissions, `nvidia-smi`,
and `MODELS_DIR`. For remote endpoints, confirm `/v1/models` health and
`EXTRA_DOCKER_ARGS` host-gateway mapping; see
[VLM and LLM Endpoints](vlm-llm-endpoints.md).

**Endpoint unreachable from Curator** — set
`EXTRA_DOCKER_ARGS="--add-host host.docker.internal:host-gateway"` when URLs
use `host.docker.internal`.

**VLM HTTP 500 / EngineDeadError** — endpoint failure, not Curator. Relaunch
the VLM with `--max-num-seqs 1` and re-check `/v1/models`.

**OOM** — lower `captioning_window_size` / batch sizes, or restrict `GPUS`.

**`ffmpeg: command not found` in container** — `make ffmpeg-install` and
`make check-setup`.

**Ray “Deadline Exceeded”** — prefer `DOCKER_NETWORK=bridge`. If Docker disk
is nearly full, set `CURATOR_TMP` to a large filesystem.

## TAO embeddings and mining

**`make check-data-mining-image` fails** — run `make pull-data-mining` after
`docker login nvcr.io`.

**Nearest neighbors dimension mismatch** — regenerate S and B with the same
embedding model; declare `TDM_EMBEDDING_BACKEND` correctly.

**Custom column names ignored** — nearest neighbors uses
`SOURCE_EMBED_COLUMN_NAME` / `TARGET_EMBED_COLUMN_NAME`. UNM uses
`SOURCE_EMBEDDING_COLUMN` / `TARGET_EMBEDDING_COLUMN`.

**UNM `class_stratified` rejected** — supply both detection paths,
`DETECTION_FORMAT`, and non-empty `RARE_CLASS_LIST`.

**UNM output missing `final_unique_files.parquet`** — the job did not complete;
inspect logs and `summary.json` if present.

## Registry authentication

Use Docker interactive login. Never embed tokens in scripts or logs.

## Still stuck

1. `make help`
2. Re-run with a minimal fixture ([Getting Started](getting-started.md))
3. Confirm image tags with `make check-image` / `make check-data-mining-image`
