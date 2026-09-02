<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Operations: Cosmos Curator

Run video and image curation with Make. Stage sample clips first:
[Samples and Cookbooks](samples-and-cookbooks.md). Curator pipeline diagram:
[Architecture](../developer/architecture.md).

## Before you run

1. Complete [Installation](installation.md) Path A checks.
2. Put weights in `MODELS_DIR` (`make download-models`). See [Models](#models).
3. Ensure `FFMPEG_DIR` passes `make check-setup`.
4. Copy NGC clips into `cookbook/*/videos/` before the first-run YAMLs.

## Video pipeline (split → dedup → shard)

```bash
MODELS=/path/to/models
DATA=$PWD/cookbook/traffic-video-analytics

make run-pipeline \
  CONFIG_FILE=cookbook/traffic-video-analytics/split-minimal.yaml \
  MODELS_DIR=$MODELS \
  DATA_DIR=$DATA

make run-pipeline \
  CONFIG_FILE=cookbook/traffic-video-analytics/split.yaml \
  MODELS_DIR=$MODELS \
  DATA_DIR=$DATA

make run-pipeline \
  CONFIG_FILE=cookbook/traffic-video-analytics/dedup.yaml \
  MODELS_DIR=$MODELS \
  DATA_DIR=$DATA

make run-pipeline \
  CONFIG_FILE=cookbook/traffic-video-analytics/shard.yaml \
  MODELS_DIR=$MODELS \
  DATA_DIR=$DATA
```

| Directory | Focus |
|-----------|--------|
| `cookbook/traffic-video-analytics/` | Two VSS traffic clips; `split-minimal.yaml` first-run (`limit: 1`) |
| `cookbook/warehouse-safety/` | Four VSS warehouse clips; `split-minimal.yaml` first-run |

## Image annotate

```bash
make run_image_pipeline \
  IMAGE_CONFIG_FILE=configs/image.yaml \
  MODELS_DIR=/path/to/models \
  DATA_DIR=/path/to/images
```

## Make variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `DATA_DIR` | required | Host directory mounted at `/data` |
| `MODELS_DIR` | `$HOME/models` | Curator weights |
| `FFMPEG_DIR` | `$HOME/cosmos-curator-ffmpeg` | FFmpeg sidecar |
| `CONFIG_FILE` | `configs/split.yaml` | Video pipeline YAML |
| `IMAGE_CONFIG_FILE` | `configs/image.yaml` | Image annotate YAML |
| `GPUS` | `all` | Docker `--gpus` value |
| `SHM_SIZE` | `24gb` | Container `/dev/shm` |
| `CURATOR_TMP` | unset | Large scratch for `/tmp` |
| `EXTRA_DOCKER_ARGS` | empty | Extra `docker run` args |

Reference templates: `configs/split.yaml`, `configs/dedup.yaml`,
`configs/shard.yaml`, `configs/image.yaml`. Cookbook YAML uses native
cosmos-curator `snake_case` keys. Omitted keys receive parser defaults.

## Embedding producers for TMM

| Producer | How you get it | `TDM_EMBEDDING_BACKEND` |
|----------|----------------|-------------------------|
| InternVideo2 (IV2) | Curator video embed stage | `iv2` |
| Cosmos-Embed1 (CE1) | Curator video embed stage | `ce1` |
| Image annotate | Curator image pipeline | follow image contract |

After Curator writes embedding parquets, stage S/B and continue with
[Operations: Data Mining](operations-tao-mining.md).

## Remote caption endpoints

Full-split cookbooks need live OpenAI-compatible VLM/LLM HTTP services.
[VLM and LLM Endpoints](vlm-llm-endpoints.md). Never put API keys on Make argv.

## Models

Models Cosmos Curator **2.3.0** (this repo pin) can load. Set algorithms in
YAML. Put weights in `MODELS_DIR`. Allowed strings also appear as comments on
`configs/split.yaml` and `configs/image.yaml`.

Remote OpenAI-compatible or Gemini APIs do not need local VLM weights.

### Default download

`make download-models` pulls the `MODEL_LIST` default from the Makefile
(~90GB+). Override with `MODEL_LIST=...` if you need a subset.

| `MODEL_LIST` token | Used for |
|--------------------|----------|
| `transnetv2` | Scene-cut splitting |
| `qwen3_vl_30b` | Qwen3-VL-30B captions / classifier (BF16) |
| `qwen3_vl_30b_fp8` | Same family, FP8 (traffic cookbook) |
| `qwen2.5_lm` | Caption enhance (`qwen_lm`) |
| `qwen2.5_vl` | Qwen2-VL-7B (`captioning_algorithm: qwen`) |
| `internvideo2_mm` | InternVideo2 video weights (also download `bert`) |
| `clip_vit` | Image CLIP embeddings / aesthetic |
| `aesthetic_scorer` | Aesthetic filter |
| `cosmos_embed1_224p` | Cosmos-Embed1 224p (traffic cookbook) |
| `sam3` | SAM3 tracking (`PIXI_ENV=sam3`) |

SeedVR2 is **not** in `MODEL_LIST`. Run `make download-seedvr2` when a cookbook
sets `super_resolution: true`. SAM3 and some HF weights need an `HF_TOKEN`.

### Extra tokens (opt-in)

Default `MODEL_LIST` does not include these. Pass them when the YAML enables
the matching feature:

```bash
make download-models MODELS_DIR=/path/to/models \
  MODEL_LIST=bert,paddle_ocr_det,paddle_ocr_rec
make download-models MODELS_DIR=/path/to/models MODEL_LIST=t5_xxl
```

| `MODEL_LIST` token | Required when | Notes |
|--------------------|---------------|-------|
| `bert` | `embedding_algorithm: internvideo2` | BERT sidecar. Needed for warehouse Path A. |
| `paddle_ocr_det`, `paddle_ocr_rec` | `artificial_text_filter: enable` | One-process prefetch. Product OCR is off. |
| `t5_xxl` | `pipeline: shard` | Reduced T5 encoder. Confirm `pytorch_model.bin.reduced`. |

`internvideo2_mm` does not vendor BERT (`google-bert/bert-large-uncased`).
Paddle may still fetch an angle classifier at runtime. `t5_xxl` is the
reduced `google-t5/t5-11b` encoder. If the T5 reduce step fails with
missing `torch`, rerun it in the Curator `default` pixi environment
(`model-download` may lack `torch`).

### Captioning (VLM)

YAML key: `captioning_algorithm`. The same family also drives video classifier,
VLM filter, and event captions unless you point those stages at an endpoint.

| Algorithm | Backend | Approx. GPU |
|-----------|---------|-------------|
| `qwen` | Qwen2-VL-7B, local vLLM | ~15 GB |
| `qwen3_5_27b` | Qwen3.5-27B, local | ~20 GB |
| `qwen3_6_27b` | Qwen3.6-27B, local | ~35 GB |
| `qwen3_6_27b_fp8` | Qwen3.6-27B FP8 | ~20 GB |
| `qwen3_vl_30b` | Qwen3-VL-30B, local | ~35 GB |
| `qwen3_vl_30b_fp8` | Qwen3-VL-30B FP8 | ~20 GB |
| `qwen3_vl_235b` / `qwen3_vl_235b_fp8` | Qwen3-VL-235B | multi-GPU |
| `cosmos_r1` / `cosmos_r2` | Cosmos-Reason VL | ~35 GB |
| `nemotron` | Nemotron-Nano-12B-v2-VL | ~15 GB |
| `openai` | OpenAI-compatible HTTP | none (remote) |
| `gemini` | Gemini API | none (remote) |
| `vllm_async` | Auto-scaled vLLM | video only |

Traffic and warehouse cookbooks use `qwen3_vl_30b_fp8` on the full split.

### Caption enhance (LLM)

YAML key: `enhance_captions_lm_variant` (video only, with
`enhance_captions: true`).

| Variant | Backend | Approx. GPU |
|---------|---------|-------------|
| `qwen_lm` | Qwen2.5-14B, local | ~15 GB |
| `gpt_oss_20b` | GPT-OSS-20B, local | ~20 GB |
| `openai` | OpenAI-compatible HTTP | none (remote) |

### Embeddings

YAML key: `embedding_algorithm`. Mine with a matching
`TDM_EMBEDDING_BACKEND` (Make default is `ce1`).

| Algorithm | Modality | `TDM_EMBEDDING_BACKEND` |
|-----------|----------|-------------------------|
| `internvideo2` | Video + image | `iv2` |
| `cosmos-embed1-224p` | Video + image | `ce1` |
| `cosmos-embed1-336p` | Video + image | `ce1` |
| `cosmos-embed1-448p` | Video + image | `ce1` |
| `clip` | Image annotate only | `clip` |
| `openai` | Video + image, remote | follow the API |

Cookbook defaults: traffic `cosmos-embed1-224p`; warehouse `internvideo2`
(download `bert` with `internvideo2_mm`).

### Split, track, quality

| Feature | YAML | Model |
|---------|------|-------|
| Scene cuts | `splitting_algorithm: transnetv2` | TransNetV2 |
| Fixed stride | `splitting_algorithm: fixed-stride` | none |
| Object tracks | `sam3: true` | SAM3 |
| Track overlay | `sam3_region: box` or `contour` | SAM3 (`box` = rectangles) |
| Super-resolution | `super_resolution: true`, `sr_variant` | SeedVR2 `seedvr2_3b`, `seedvr2_7b`, `seedvr2_7b_sharp` |
| Aesthetic filter | `aesthetic_threshold` | CLIP aesthetic scorer |
| Artificial text | `artificial_text_filter: enable` | PaddleOCR (`paddle_ocr_det`, `paddle_ocr_rec`) |
| Motion score | `motion_filter` | optical flow (no HF download) |

Event captions (`event_captioning: true`) require SAM3. Backend:
`event_caption_backend: qwen` or `gemini`.

### How to select weights

1. Pick algorithms in the cookbook YAML.
2. Confirm tokens exist under `MODELS_DIR` (`make download-models`, extra
   tokens from [Extra tokens (opt-in)](#extra-tokens-opt-in), plus
   `make download-seedvr2` if you enable SR).
3. For `openai` / `gemini`, export `COSMOS_CURATOR_OPENAI_*` (or Gemini keys)
   instead of expecting those VLMs in `MODELS_DIR`.

## Interactive shell

```bash
make shell MODELS_DIR=/path/to/models DATA_DIR=/path/to/data
```
