<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# VLM and LLM Endpoints

Some Curator cookbooks caption, filter, classify, or enhance text through
**OpenAI-compatible HTTP endpoints** (vLLM, NIM, or a hosted API). Other runs
use **in-process** models inside the Curator container and do not need these
services.

Model files under `MODELS_DIR` are not enough for endpoint-backed runs. You
must have live HTTP services reachable from the Curator container.

## When you need endpoints

| Path | Endpoints required? |
|------|---------------------|
| Local in-process Curator | No |
| Full-split cookbooks with remote captioning | Yes — caption/filter/classifier VLM and enhance LLM |
| Embeddings / mining only (Path B) | No |

Cosmos Curator does **not** take endpoint URLs from cookbook YAML. Make builds
`/cosmos_curator/config/cosmos_curator.yaml` from environment variables via
`adapters.cosmos_curator.openai_config_env` before each Curator Docker run.

## Roles and environment variables

| Role | Curator stage | Base URL env | API key env |
|------|---------------|--------------|-------------|
| `caption` | Window / event caption VLM | `COSMOS_CURATOR_OPENAI_CAPTION_BASE_URL` | `COSMOS_CURATOR_OPENAI_CAPTION_API_KEY` |
| `filter` | VLM filter | `COSMOS_CURATOR_OPENAI_FILTER_BASE_URL` | `COSMOS_CURATOR_OPENAI_FILTER_API_KEY` |
| `classifier` | Video classifier | `COSMOS_CURATOR_OPENAI_CLASSIFIER_BASE_URL` | `COSMOS_CURATOR_OPENAI_CLASSIFIER_API_KEY` |
| `enhance` | Caption enhance LLM | `COSMOS_CURATOR_OPENAI_ENHANCE_BASE_URL` | `COSMOS_CURATOR_OPENAI_ENHANCE_API_KEY` |
| `embedding` | Optional remote embed API | `COSMOS_CURATOR_OPENAI_EMBEDDING_BASE_URL` | `COSMOS_CURATOR_OPENAI_EMBEDDING_API_KEY` |

Notes:

- Put values in gitignored `local.env` or export them in the shell. Never commit
  keys. See [Installation](installation.md).
- If unset, filter/classifier default to the caption/VLM URL.
- Missing API keys default to `EMPTY` (typical for local open endpoints).
- Optional aliases: `SQA_VLM_BASE_URL` → caption (and filter/classifier when
  unset); `SQA_LLM_BASE_URL` → enhance.
- Template comments: `.env.example`.

You set the env vars above (or in gitignored `local.env`) and run
`make run-pipeline`. Make generates the Curator OpenAI config mount
automatically.

## Scenario A — local endpoint containers

Use when you host vLLM (or equivalent) on the same machine. Keep endpoint GPUs
**separate** from Curator and TAO GPUs.

### Example 4-GPU map

| Host GPU | Owner | Use |
|----------|-------|-----|
| 0 | VLM | Caption / filter / classifier |
| 1 | LLM | Text enhancement |
| 2 | Curator | Split, SAM3, IV2, orchestration |
| 3 | TAO | Image/text embeddings, TMM NN/UNM |

Confirm free memory before launch:

```bash
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv
```

### Launch (tested stable VLM profile)

Requires: NGC login for the vLLM image, models present under `MODELS_DIR`
(for example after `make download-models`), and an approved `HF_TOKEN` when
gated weights need it.

```bash
# Adjust MODELS_DIR and GPU indices for your host.
export MODELS_DIR=${MODELS_DIR:-$HOME/models}

docker rm -f paidf-vlm paidf-llm 2>/dev/null || true

docker run -d --name paidf-vlm --gpus '"device=0"' --ipc=host \
  -p 8000:8000 \
  -e HF_TOKEN="$HF_TOKEN" \
  -v "$MODELS_DIR:/models:ro" \
  nvcr.io/nvidia/vllm:26.04-py3 \
  vllm serve /models/Qwen/Qwen3-VL-30B-A3B-Instruct-FP8 \
    --served-model-name Qwen/Qwen3-VL-30B-A3B-Instruct-FP8 \
    --trust-remote-code \
    --host 0.0.0.0 --port 8000 \
    --tensor-parallel-size 1 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.85 \
    --max-num-seqs 1 \
    --enforce-eager

docker run -d --name paidf-llm --gpus '"device=1"' --ipc=host \
  -p 8002:8002 \
  -e HF_TOKEN="$HF_TOKEN" \
  -v "$MODELS_DIR:/models:ro" \
  nvcr.io/nvidia/vllm:26.04-py3 \
  vllm serve /models/Qwen/Qwen2.5-14B-Instruct \
    --served-model-name Qwen/Qwen2.5-14B-Instruct \
    --host 0.0.0.0 --port 8002 --max-model-len 16384
```

Keep `--max-num-seqs 1` on the VLM so long video caption requests stay
serialized. Endpoint-backed runs are expected to be slower with this setting.

### Health check on the host

```bash
curl --fail --silent --show-error "http://localhost:8000/v1/models"
curl --fail --silent --show-error "http://localhost:8002/v1/models"
```

### Point Curator at the endpoints

With Make’s default Docker **bridge** network, use `host.docker.internal` and
pass the Linux host-gateway mapping:

```bash
export COSMOS_CURATOR_OPENAI_CAPTION_BASE_URL="http://host.docker.internal:8000/v1"
export COSMOS_CURATOR_OPENAI_FILTER_BASE_URL="$COSMOS_CURATOR_OPENAI_CAPTION_BASE_URL"
export COSMOS_CURATOR_OPENAI_CLASSIFIER_BASE_URL="$COSMOS_CURATOR_OPENAI_CAPTION_BASE_URL"
export COSMOS_CURATOR_OPENAI_ENHANCE_BASE_URL="http://host.docker.internal:8002/v1"
export COSMOS_CURATOR_OPENAI_CAPTION_API_KEY="EMPTY"
export COSMOS_CURATOR_OPENAI_FILTER_API_KEY="EMPTY"
export COSMOS_CURATOR_OPENAI_CLASSIFIER_API_KEY="EMPTY"
export COSMOS_CURATOR_OPENAI_ENHANCE_API_KEY="EMPTY"

export EXTRA_DOCKER_ARGS="--add-host host.docker.internal:host-gateway"
export GPUS='"device=2"'   # Curator pool; do not use GPUS=all while endpoints own 0/1
```

Confirm the host health check above succeeds, then run Make:

```bash
make run-pipeline \
  CONFIG_FILE=cookbook/traffic-video-analytics/split.yaml \
  MODELS_DIR="$MODELS_DIR" \
  DATA_DIR=/path/to/data \
  FFMPEG_DIR=$HOME/cosmos-curator-ffmpeg \
  GPUS='"device=2"' \
  EXTRA_DOCKER_ARGS="--add-host host.docker.internal:host-gateway"
```

If Curator still cannot reach the endpoints, confirm `EXTRA_DOCKER_ARGS` includes
the host-gateway mapping (see [Failure modes](#failure-modes)).

### Tear down

```bash
docker rm -f paidf-vlm paidf-llm
```

## Scenario B — hosted endpoints

When a platform team provides OpenAI-compatible URLs and model names:

```bash
export COSMOS_CURATOR_OPENAI_CAPTION_BASE_URL="<provided-vlm-endpoint>/v1"
export COSMOS_CURATOR_OPENAI_FILTER_BASE_URL="$COSMOS_CURATOR_OPENAI_CAPTION_BASE_URL"
export COSMOS_CURATOR_OPENAI_CLASSIFIER_BASE_URL="$COSMOS_CURATOR_OPENAI_CAPTION_BASE_URL"
export COSMOS_CURATOR_OPENAI_ENHANCE_BASE_URL="<provided-llm-endpoint>/v1"
export COSMOS_CURATOR_OPENAI_CAPTION_API_KEY="$NVIDIA_API_KEY"
export COSMOS_CURATOR_OPENAI_FILTER_API_KEY="$NVIDIA_API_KEY"
export COSMOS_CURATOR_OPENAI_CLASSIFIER_API_KEY="$NVIDIA_API_KEY"
export COSMOS_CURATOR_OPENAI_ENHANCE_API_KEY="$NVIDIA_API_KEY"
```

Health check without printing the key in logs you will archive:

```bash
curl --fail --silent --show-error \
  --oauth2-bearer "$COSMOS_CURATOR_OPENAI_CAPTION_API_KEY" \
  "$COSMOS_CURATOR_OPENAI_CAPTION_BASE_URL/models" >/dev/null
curl --fail --silent --show-error \
  --oauth2-bearer "$COSMOS_CURATOR_OPENAI_ENHANCE_API_KEY" \
  "$COSMOS_CURATOR_OPENAI_ENHANCE_BASE_URL/models" >/dev/null
```

Record URLs and model names in evidence; never record API key values.
`EXTRA_DOCKER_ARGS` is needed only if the URL uses `host.docker.internal`.

## GPU quoting

Preserve Docker’s quoted device expression:

```bash
export GPUS='"device=2"'
# Wrong: GPUS=device=2   (Docker rejects this form)
```

While local VLM/LLM containers occupy GPUs, do not set `GPUS=all` for Curator
or TAO.

## Failure modes

| Symptom | Likely cause | Action |
|---------|--------------|--------|
| Curator cannot reach endpoint | Missing host-gateway / wrong URL | Set `EXTRA_DOCKER_ARGS`; re-check host `/v1/models` |
| `/v1/models` fails on host | Endpoint not ready or wrong port | `docker logs paidf-vlm` / `paidf-llm`; wait for ready |
| VLM HTTP 500, `EngineDeadError`, deepstack token errors | Endpoint overload / bad profile | Relaunch VLM with the stable flags above (`--max-num-seqs 1`) |
| Job appears hung | Serialized VLM queue | Check VLM logs for queue progress before killing |
| Auth errors on hosted API | Bad or missing key | Fix key in `local.env`; do not put it on Make argv |

More Curator tips: [Troubleshooting](troubleshooting.md).

## Related

- [Operations: Curator](operations-curator.md)
- [Samples and Cookbooks](samples-and-cookbooks.md)
