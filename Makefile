# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# =============================================================================
# PAIDF Curation and Retrieval - Makefile (local/operator façade)
# =============================================================================
# Make delegates finite operations to the CLI. External orchestration integrates
# through the CLI/container/API contracts, not through Make.
#
# Primary path:
#   make pull-all              (pull Curator + Dataset Search + Data Mining)
#   make run-pipeline          (Curator)
#   make ds-health / search    (CDS)
#   make run-data-mining-select
#
# Optional developer path:
#   make clone-curator build   (build Curator from an ignored external checkout)
# =============================================================================

# Use bash for recipes so bash builtins like `read -p` work reliably.
SHELL := /bin/bash

-include .env
-include local.env

.PHONY: help setup build build-dry-run clone-curator pull pull-dataset-search pull-data-mining \
	pull-all tag-local-images download-models download-seedvr-ckpts download-seedvr2 \
	download-all-models check-build check-setup check-image \
	check-dataset-search-image check-data-mining-image check-curator-runtime clean format \
	run-pipeline run_image_pipeline shell ffmpeg-install \
	up-dataset-search down-dataset-search ingest-curator search \
	ds-health ds-pipelines ds-collections ds-create-collection ds-get-collection \
	ds-bulk-insert ds-job-status stage-cds-artifact caption-build caption-validate \
	caption-readiness caption-search caption-upload caption-bulk-insert \
	prepare-cds-ce1-for-tdm run-data-mining-select run-data-mining-unique-match \
	image-embeddings-build image-embeddings-validate-input run-image-embeddings \
	image-embeddings-validate-output text-embeddings-validate-input \
	run-text-embeddings text-embeddings-validate-output

# =============================================================================
# Configuration
# =============================================================================

COSMOS_REPO ?= .external/cosmos-curator
COSMOS_REPO_URL ?= https://github.com/NVIDIA/cosmos-curator.git

# NGC image pins (single source of truth — edit here for version drops).
# Make exports these so CLI/adapters pick them up from the environment.
COSMOS_CURATOR_REGISTRY ?= nvcr.io/nvidia/cosmos/cosmos-curator
COSMOS_CURATOR_TAG ?= 2.3.0
COSMOS_CURATOR_IMAGE ?= cosmos-curator
DATASET_SEARCH_REGISTRY ?= nvcr.io/nvidia/blueprint/cosmos-dataset-search
DATASET_SEARCH_TAG ?= 1.0.0
DATASET_SEARCH_IMAGE ?= cosmos-dataset-search
DATA_MINING_REGISTRY ?= nvcr.io/nvidia/tao/tao-toolkit
DATA_MINING_TAG ?= 7.2.0-data-services
DATA_MINING_IMAGE ?= tao-toolkit

export COSMOS_CURATOR_REGISTRY COSMOS_CURATOR_TAG COSMOS_CURATOR_IMAGE
export DATASET_SEARCH_REGISTRY DATASET_SEARCH_TAG DATASET_SEARCH_IMAGE
export DATA_MINING_REGISTRY DATA_MINING_TAG DATA_MINING_IMAGE

IMAGE_NAME ?= $(COSMOS_CURATOR_IMAGE)
IMAGE_TAG ?= $(COSMOS_CURATOR_TAG)
MODELS_DIR ?= $(HOME)/models
MODEL_LIST ?= transnetv2,qwen3_vl_30b,qwen3_vl_30b_fp8,qwen2.5_lm,qwen2.5_vl,internvideo2_mm,clip_vit,aesthetic_scorer,cosmos_embed1_224p,sam3
COSMOS_IMAGE_ENVS ?= cuml,default,legacy-transformers,seedvr,model-download,sam3
DATA_DIR ?=
GPUS ?= all
EXTRA_DOCKER_ARGS ?=
PAIDF_CLI ?= paidf_curation_and_retrieval

# Cosmos Curator OpenAI/HF config mount (container: /cosmos_curator/config/cosmos_curator.yaml).
# Prefer env-generated config (Option A): SQA_VLM_BASE_URL / SQA_LLM_BASE_URL or
# COSMOS_CURATOR_OPENAI_<ROLE>_BASE_URL / _API_KEY. Falls back to host dir when env unset.
# See: uv run python -m adapters.cosmos_curator.openai_config_env prepare --help
COSMOS_CURATOR_CONFIG_DIR ?= $(HOME)/.config/cosmos_curator
COSMOS_CURATOR_CONFIG_GEN_DIR ?= $(if $(CURATOR_TMP),$(CURATOR_TMP)/cosmos_curator_config_env,/tmp/cosmos_curator_config_env)

# ---------------------------------------------------------------------------
# FFmpeg sidecar (host-built, mounted into the container)
# ---------------------------------------------------------------------------
# As of cosmos-curator 2026-05-12, FFmpeg is NO LONGER baked into the Docker
# image. The pipeline expects an FFmpeg sidecar mounted at /opt/ffmpeg with
# /opt/ffmpeg/bin/ffmpeg + ffprobe and /opt/ffmpeg/lib/* present. Pre-built
# GPU-aware FFmpeg installs (e.g. `cosmos-curator-ffmpeg/` produced by
# pip-building ffmpeg-python with CUDA codecs) work out of the box.
#
# Override the host path via the CLI:  make run-pipeline FFMPEG_DIR=/opt/ffmpeg
FFMPEG_DIR ?= $(HOME)/cosmos-curator-ffmpeg

# Pin used by `make ffmpeg-install`. Matches the currently deployed sidecar
# (package ffmpeg-8.1.1-lgpl_ha074a71_801 from conda-forge): LGPL build with
# NVENC, NVDEC, libopenh264 (default H.264 encoder), libdav1d, libsvtav1,
# libvpx, libaom, libplacebo, vaapi. License-clean; no libx264.
FFMPEG_PACKAGE_SPEC ?= ffmpeg=8.1.1=lgpl*
FFMPEG_CHANNEL      ?= conda-forge

# Reusable mount fragment. Use `$(FFMPEG_MOUNT)` inside every recipe that
# starts a container expected to call ffmpeg/ffprobe (split, image annotate,
# interactive shell). PATH is prepended so /opt/ffmpeg/bin shadows anything
# that might have been baked in upstream; LD_LIBRARY_PATH points the dynamic
# loader at /opt/ffmpeg/lib for the GPU-codec shared objects.
FFMPEG_MOUNT = -v "$(FFMPEG_DIR):/opt/ffmpeg:ro" \
	-e PATH=/opt/ffmpeg/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
	-e LD_LIBRARY_PATH=/opt/ffmpeg/lib

# Registry for pulling pre-built Curator images (override as needed).
REGISTRY ?= $(COSMOS_CURATOR_REGISTRY)

# ---------------------------------------------------------------------------
# Dataset Search (CDS) — pull-only; CDS_PROFILE selects public vs internal pin
# ---------------------------------------------------------------------------
CDS_PROFILE ?= public
CDS_IMAGE ?= $(DATASET_SEARCH_IMAGE)
CDS_TAG ?= $(DATASET_SEARCH_TAG)
DATASET_SEARCH_URL ?= http://localhost:8888
CDS_URL ?= $(DATASET_SEARCH_URL)
CURATOR_DIR ?=
EMBEDDING_BACKEND ?= auto
OUT ?= ./data/dataset_search_ingest.parquet
COLLECTION ?=
QUERY ?=
PARQUET_URI ?=
CDS_EMBEDDING_FAMILY ?=
JOB_ID ?=
WAIT ?= 0
POLL_TIMEOUT_SECONDS ?= 900
POLL_INTERVAL_SECONDS ?= 5
STAGE_SOURCE ?=
STAGE_DESTINATION ?=
OBJECT_STORE_ENDPOINT ?=
OBJECT_STORE_ACCESS_KEY_ENV ?= AWS_ACCESS_KEY_ID
OBJECT_STORE_SECRET_KEY_ENV ?= AWS_SECRET_ACCESS_KEY
ALLOW_LAB_HTTP_ENDPOINT ?= 0
ALLOW_LAB_DOCUMENT_FALLBACK ?= 0
CAPTION_JSON ?=
CAPTION_PARQUET ?=
INDEXED_ID_FILE ?=
CAPTION_MODEL_NAME ?= default
CAPTION_DATA_SOURCE ?=
CAPTION_QUERY ?=
CAPTION_SEARCH_LIMIT ?= 5000
CAPTION_SEARCH_DATA_SOURCE ?=
DATASET_SEARCH_COMPOSE ?= deploy/compose/docker-compose.yml

# ---------------------------------------------------------------------------
# Data Mining — TAO Toolkit Data-Services (tmm nearest_neighbors +
# unique_neighbor_matching). Every invocation requires an experiment spec.
# TMM_CONFIG_FILE / TMM_UNM_CONFIG_FILE mount operator-owned specs read-only;
# direct Make variables are validated and converted to a temporary read-only
# spec by the Python runner.
# Supported metrics: cosine, euclidean, manhattan.
# nearest_neighbors also accepts optional distance_threshold (float; -1 disables)
# and source/target embed column names. unique_neighbor_matching accepts
# class_stratified (+ detection files/format/rare classes), exclude_path,
# custom column names, save_embeddings, and visualize.
# ---------------------------------------------------------------------------
DATA_MINING_SHM_SIZE ?= 16g
TMM_CONFIG_FILE ?=
TMM_UNM_CONFIG_FILE ?=
TARGET_SUBDIR ?= S
SOURCE_SUBDIR ?= B
OUTPUT_SUBDIR ?= divknn_out
UNM_OUTPUT_SUBDIR ?= unm_out
TOPN ?= 5
DESIRED_UNIQUE_COUNT ?= 100
ALLOCATION_POLICY ?= global
CANDIDATE_EXPANSION_FACTOR ?= 5
DISTANCE_THRESHOLD ?= -1.0
FILTER_BY_LABEL ?= 0
SOURCE_EMBED_COLUMN_NAME ?= embedding
TARGET_EMBED_COLUMN_NAME ?= embedding
SOURCE_EMBEDDING_COLUMN ?= embedding
TARGET_EMBEDDING_COLUMN ?= embedding
SOURCE_FILEPATH_COLUMN ?= filepath
TARGET_FILEPATH_COLUMN ?= filepath
EXCLUDE_PATH ?=
SOURCE_DETECTION_FILE ?=
TARGET_DETECTION_FILE ?=
DETECTION_FORMAT ?=
RARE_CLASS_LIST ?=
SAVE_EMBEDDINGS ?= 0
VISUALIZE ?= 0
DATA_MINING_METRIC ?= cosine
TDM_EMBEDDING_BACKEND ?= ce1
CDS_CE1_TARGET_SELECTION ?=
CDS_CE1_SOURCE_SELECTION ?=
TMM_PREP_SUBDIR ?= _tmm_prep
IMAGE_EMBEDDING_JSON ?=
IMAGE_EMBEDDING_INPUT ?=
IMAGE_EMBEDDING_OUTPUT ?=
IMAGE_EMBEDDING_CONFIG ?=
IMAGE_EMBEDDING_MODEL_TYPE ?=
IMAGE_EMBEDDING_MODEL ?=
IMAGE_EMBEDDING_MODEL_CONFIG ?=
IMAGE_EMBEDDING_BATCH_SIZE ?= 64
IMAGE_EMBEDDING_DRY_RUN ?= 0
TEXT_EMBEDDING_INPUT ?=
TEXT_EMBEDDING_OUTPUT ?=
TEXT_EMBEDDING_CONFIG ?=
TEXT_EMBEDDING_MODEL ?=
TEXT_EMBEDDING_MODEL_PATH ?=
TEXT_EMBEDDING_BATCH_SIZE ?= 64
TEXT_EMBEDDING_DRY_RUN ?= 0

# Local retag sources (override in untracked local.env)
LOCAL_CDS_SOURCE_IMAGE ?= cds
LOCAL_CDS_SOURCE_TAG ?= local
LOCAL_DATA_MINING_SOURCE_IMAGE ?= $(DATA_MINING_REGISTRY)
LOCAL_DATA_MINING_SOURCE_TAG ?= $(DATA_MINING_TAG)

# CONFIG_FILE: a flat YAML/JSON config consumed by upstream load_pipeline_config.
# Mount it into the container and pass it to the native upstream runner.
CONFIG_FILE ?= configs/split.yaml

# SHM_SIZE: container /dev/shm size, used by Ray's shared-memory object store.
# Shared memory is allocated from system RAM, so this MUST NOT exceed
# available RAM (see README "System Requirements"). Recommended ranges:
#   - Single-GPU runs:      8gb-16gb
#   - Multi-GPU split/dedup: 24gb-64gb (default 24gb fits the 64GB RAM minimum)
#   - Large multi-GPU hosts: scale up toward ~40% of system RAM
# Override on the CLI: `make run-pipeline SHM_SIZE=32gb`
SHM_SIZE ?= 24gb

# PIXI_ENV: which pre-baked pixi environment inside the cosmos-curator image to
# run under. Staging distributable images (2026-07+) dropped `unified`; use
# `default` for split / annotate. Override with `make run-pipeline PIXI_ENV=...`.
PIXI_ENV ?= default

# Docker networking for Curator containers.
# Default: bridge — avoids Ray double-init metrics-port clashes on --network=host.
# Override: DOCKER_NETWORK=host when you explicitly need host networking.
DOCKER_NETWORK ?= bridge

# Optional host dir for container /tmp and /config/tmp (Ray / HF scratch).
# Use a large filesystem when Docker's graph driver is nearly full.
# Example: CURATOR_TMP=/data/curator-tmp make run-pipeline ...
CURATOR_TMP ?=

# Models volume mode: rw (default; Transformers may write HF modules cache) or
# ro (then HF caches are redirected under /tmp — set CURATOR_TMP for durability).
MODELS_MOUNT_MODE ?= rw

# Minimum free GiB warning thresholds for Curator preflight.
CURATOR_MIN_TMP_GIB ?= 20
CURATOR_MIN_DOCKER_GIB ?= 10

# SeedVR2 checkpoints (pseudo-labeling layout under MODELS_DIR/seedvr2).
# Sources: ByteDance-Seed/SeedVR2-3B and SeedVR2-7B (ema_vae + DiT + pos/neg_emb).
# Preflight mounts seedvr2/ -> SeedVR/ckpts and pos/neg_emb onto
# /config/models/ByteDance-Seed/SeedVR2-{3B|7B}/ (Curator text-embed path).
SEEDVR_VARIANT ?= seedvr2_3b
# auto: download if CONFIG_FILE enables super_resolution; check: fail if missing;
# skip: never; always: ensure even when SR disabled.
ENSURE_SEEDVR_CKPTS ?= auto
SEEDVR_CONTAINER_CKPTS ?= /opt/cosmos-curator/SeedVR/ckpts

# Derived docker run fragments (do not override unless you know why).
ifeq ($(DOCKER_NETWORK),)
DOCKER_NETWORK_ARGS :=
else
DOCKER_NETWORK_ARGS := --network=$(DOCKER_NETWORK)
endif

ifeq ($(MODELS_MOUNT_MODE),ro)
MODELS_VOLUME_ARGS := -v "$(MODELS_DIR):/config/models:ro"
HF_ENV_ARGS := -e HF_HOME=/tmp/hf_home -e HF_MODULES_CACHE=/tmp/hf_home/modules -e TRANSFORMERS_CACHE=/tmp/hf_home
else
MODELS_VOLUME_ARGS := -v "$(MODELS_DIR):/config/models"
HF_ENV_ARGS := -e HF_HOME=/config/models
endif

ifdef CURATOR_TMP
CURATOR_TMP_ARGS := -e TMPDIR=/tmp -e RAY_TMPDIR=/tmp/ray \
	-v "$(CURATOR_TMP):/tmp" -v "$(CURATOR_TMP):/config/tmp"
else
CURATOR_TMP_ARGS :=
endif

# =============================================================================
# Commands
# =============================================================================

help:
	@echo ""
	@echo "PAIDF Curation and Retrieval — Make-only operator UX"
	@echo "===================================================="
	@echo ""
	@echo "Get images:"
	@echo "  make pull-all              Pull Curator + Dataset Search + Data Mining"
	@echo "  make pull                  Pull Curator (REGISTRY=$(REGISTRY) IMAGE_TAG=$(IMAGE_TAG))"
	@echo "  make setup                 Pull Curator, install/check FFmpeg sidecar, verify setup"
	@echo ""
	@echo "Optional Curator source build:"
	@echo "  make clone-curator         Clone upstream source into $(COSMOS_REPO)"
	@echo "  make build                 Build cosmos-curator image from $(COSMOS_REPO)"
	@echo "  make build-dry-run         Test source-build prerequisites without building"
	@echo ""
	@echo "Setup:"
	@echo "  make check-setup           Check prerequisites (Docker, NVIDIA driver, FFmpeg sidecar)"
	@echo "  make check-curator-runtime Preflight for Curator docker run (disk, ffmpeg, models, SeedVR, GPU)"
	@echo "  make check-image           Verify cosmos-curator image exists locally"
	@echo "  make ffmpeg-install        Install conda-forge LGPL FFmpeg sidecar into FFMPEG_DIR"
	@echo "                             FFMPEG_DIR=$(FFMPEG_DIR)"
	@echo "  make download-models       Download Curator models (~90GB; no SeedVR2)"
	@echo "                             MODELS_DIR=$(MODELS_DIR)"
	@echo "  make download-seedvr2      Download SeedVR2 HF weights into MODELS_DIR/seedvr2"
	@echo "                             alias: download-seedvr-ckpts; SEEDVR_VARIANT=$(SEEDVR_VARIANT)"
	@echo "                             (ByteDance-Seed/SeedVR2-3B|7B; needs HF_TOKEN if gated)"
	@echo "  make download-all-models   download-models then download-seedvr2"
	@echo ""
	@echo "Curator:"
	@echo "  make run-pipeline          Video pipeline (split/dedup/shard)"
	@echo "                             CONFIG_FILE=$(CONFIG_FILE)"
	@echo "                             DOCKER_NETWORK=$(DOCKER_NETWORK) MODELS_MOUNT_MODE=$(MODELS_MOUNT_MODE)"
	@echo "                             [CURATOR_TMP=/path/for/scratch] [SEEDVR_VARIANT=$(SEEDVR_VARIANT)]"
	@echo "                             OpenAI endpoints: set SQA_VLM_BASE_URL + SQA_LLM_BASE_URL"
	@echo "                             (Makefile writes task-local /cosmos_curator/config)"
	@echo "  make run_image_pipeline    Image annotate (IMAGE_CONFIG_FILE=$(IMAGE_CONFIG_FILE))"
	@echo "  make shell                 Interactive shell in Curator container"
	@echo ""
	@echo "Dataset Search (CDS_PROFILE=public|internal — see .env.example):"
	@echo "  make pull-dataset-search   Pull CDS image"
	@echo "  make up-dataset-search     Start Milvus + Dataset Search via compose"
	@echo "  make down-dataset-search   Stop Dataset Search compose stack"
	@echo "  make ds-health             GET /health (CDS_URL=$(CDS_URL))"
	@echo "  make ds-pipelines          List embedding pipelines"
	@echo "  make ds-collections        List collections"
	@echo "  make ds-create-collection  NAME=... PIPELINE=cosmos_video_search_milvus"
	@echo "  make ds-get-collection     COLLECTION=..."
	@echo "  make ingest-curator        Curator export → CDS parquet"
	@echo "                             CURATOR_DIR=... OUT=$(OUT) [COLLECTION=...] [EMBEDDING_BACKEND=auto|iv2|ce1]"
	@echo "  make stage-cds-artifact    STAGE_SOURCE=... STAGE_DESTINATION=s3://bucket/key"
	@echo "  make ds-bulk-insert        COLLECTION=... PARQUET_URI=s3://bucket/key CDS_EMBEDDING_FAMILY=ce1"
	@echo "  make ds-job-status         JOB_ID=... [WAIT=1]"
	@echo "  make caption-build         CAPTION_JSON=... CAPTION_PARQUET=..."
	@echo "  make caption-validate      CAPTION_PARQUET=... [INDEXED_ID_FILE=...]"
	@echo "  make caption-readiness     PIPELINE=... CDS_PROFILE=internal CDS_URL=..."
	@echo "  make caption-search        CAPTION_QUERY='...' [CAPTION_SEARCH_LIMIT=5000]"
	@echo "  make caption-upload        CAPTION_PARQUET=... CDS_URL=..."
	@echo "  make caption-bulk-insert   PARQUET_URI=s3://bucket/captions.parquet"
	@echo "  make search                COLLECTION=... QUERY='...'"
	@echo ""
	@echo "Data Mining (TAO Toolkit DS — tmm nearest_neighbors | unique_neighbor_matching):"
	@echo "  make pull-data-mining      Pull TAO Data Services (tag: $(DATA_MINING_TAG))"
	@echo "  make prepare-cds-ce1-for-tdm  DATA_DIR=... CDS_CE1_TARGET_SELECTION=..."
	@echo "                             CDS_CE1_SOURCE_SELECTION=... CDS_EMBEDDING_FAMILY=ce1"
	@echo "  make run-data-mining-select  DATA_DIR=... [TMM_CONFIG_FILE=...]"
	@echo "                             TARGET_SUBDIR=$(TARGET_SUBDIR)  SOURCE_SUBDIR=$(SOURCE_SUBDIR)"
	@echo "                             OUTPUT_SUBDIR=$(OUTPUT_SUBDIR)  TOPN=$(TOPN)"
	@echo "                             DATA_MINING_METRIC=cosine|euclidean|manhattan"
	@echo "                             DISTANCE_THRESHOLD=$(DISTANCE_THRESHOLD)"
	@echo "                             FILTER_BY_LABEL=$(FILTER_BY_LABEL)"
	@echo "                             SOURCE_EMBED_COLUMN_NAME=$(SOURCE_EMBED_COLUMN_NAME)"
	@echo "                             TARGET_EMBED_COLUMN_NAME=$(TARGET_EMBED_COLUMN_NAME)"
	@echo "                             TDM_EMBEDDING_BACKEND=$(TDM_EMBEDDING_BACKEND)"
	@echo "                             GPUS=$(GPUS)  DATA_MINING_SHM_SIZE=$(DATA_MINING_SHM_SIZE)"
	@echo "  make run-data-mining-unique-match  DATA_DIR=... [TMM_UNM_CONFIG_FILE=...]"
	@echo "                             DESIRED_UNIQUE_COUNT=$(DESIRED_UNIQUE_COUNT)"
	@echo "                             ALLOCATION_POLICY=$(ALLOCATION_POLICY)"
	@echo "                             UNM_OUTPUT_SUBDIR=$(UNM_OUTPUT_SUBDIR)"
	@echo "                             CANDIDATE_EXPANSION_FACTOR=$(CANDIDATE_EXPANSION_FACTOR)"
	@echo "                             SOURCE_DETECTION_FILE=... TARGET_DETECTION_FILE=..."
	@echo "                             DETECTION_FORMAT=coco (JSON file) | kitti (label dir)"
	@echo "                             RARE_CLASS_LIST=class1,class2"
	@echo "                             EXCLUDE_PATH=... SAVE_EMBEDDINGS=0|1 VISUALIZE=0|1"
	@echo "  make image-embeddings-build  DATA_DIR=... IMAGE_EMBEDDING_JSON=..."
	@echo "                             IMAGE_EMBEDDING_INPUT=.../input.parquet"
	@echo "  make run-image-embeddings  DATA_DIR=... IMAGE_EMBEDDING_INPUT=..."
	@echo "                             IMAGE_EMBEDDING_OUTPUT=... IMAGE_EMBEDDING_MODEL_TYPE=clip|siglip"
	@echo "                             IMAGE_EMBEDDING_MODEL=<HF model or TAO checkpoint>"
	@echo "                             or IMAGE_EMBEDDING_CONFIG=<vendor YAML>"
	@echo "                             (all TAO 7.1 runs use a read-only experiment spec)"
	@echo "  make image-embeddings-validate-input   Validate image rows before a GPU run"
	@echo "  make image-embeddings-validate-output  Validate vectors and metadata"
	@echo "  make text-embeddings-validate-input    Validate text rows before a GPU run"
	@echo "  make run-text-embeddings   DATA_DIR=... TEXT_EMBEDDING_INPUT=<parquet with text column>"
	@echo "                             TEXT_EMBEDDING_OUTPUT=... TEXT_EMBEDDING_MODEL=clip|siglip|siglip2"
	@echo "                             TEXT_EMBEDDING_MODEL_PATH=<HF model id or DATA_DIR path>"
	@echo "                             or TEXT_EMBEDDING_CONFIG=<vendor YAML>"
	@echo "  make text-embeddings-validate-output  Validate text vectors and metadata"
	@echo ""
	@echo "Cookbooks: cookbook/README.md"
	@echo "  Curator: make run-pipeline CONFIG_FILE=cookbook/traffic-video-analytics/split.yaml"
	@echo "  CDS:     make ds-health && make search COLLECTION=... QUERY='...'"
	@echo "  Data mining: make run-data-mining-select DATA_DIR=\$$PWD/data TMM_CONFIG_FILE=cookbook/nearest-neighbor-mining/tmm.yaml"
	@echo ""
	@echo "Linting:"
	@echo "  make format                Format code using ruff"
	@echo ""
	@echo "Key variables: CDS_URL=$(CDS_URL)  FFMPEG_DIR=$(FFMPEG_DIR)  MODELS_DIR=$(MODELS_DIR)"
	@echo ""
	@echo "Examples:"
	@echo "  make pull-all"
	@echo "  make run-pipeline CONFIG_FILE=configs/split.yaml MODELS_DIR=/models DATA_DIR=/data"
	@echo "  make ds-health"
	@echo "  make ds-create-collection NAME=traffic PIPELINE=cosmos_video_search_milvus"
	@echo "  make ingest-curator CURATOR_DIR=/path/to/curator/out OUT=./data/ingest.parquet"
	@echo "  make search COLLECTION=traffic QUERY='person crossing'"
	@echo "  make run-data-mining-select DATA_DIR=\$$PWD/data TMM_CONFIG_FILE=cookbook/nearest-neighbor-mining/tmm.yaml"
	@echo ""
	@echo "FFmpeg sidecar (required for Curator distributable images):"
	@echo "  Host build at \$$(FFMPEG_DIR) mounted at /opt/ffmpeg:ro"
	@echo "    \$$(FFMPEG_DIR)/bin/{ffmpeg,ffprobe}"
	@echo "    \$$(FFMPEG_DIR)/lib/*.so"
	@echo ""

# ---------------------------------------------------------------------------
# Prerequisites
# ---------------------------------------------------------------------------

check-build:
	@echo "Checking build prerequisites..."
	@if ! command -v docker >/dev/null 2>&1; then \
		echo "ERROR: Docker not found. Install: https://docs.docker.com/engine/install/"; \
		exit 1; \
	fi
	@echo "OK: Docker: $(shell docker --version)"
	@if ! command -v poetry >/dev/null 2>&1; then \
		echo "WARNING: Poetry not found (required for this source-build path)"; \
		echo "  Install: curl -sSL https://install.python-poetry.org | python3 -"; \
	else \
		echo "OK: Poetry: $(shell poetry --version)"; \
	fi

# Full runtime check (adds GPU + FFmpeg sidecar). The FFmpeg sidecar is a
# RUNTIME dependency (bind-mounted at run time), so it must not gate `make build`.
check-setup:
	@echo "Checking prerequisites..."
	@echo ""
	@if ! command -v docker >/dev/null 2>&1; then \
		echo "ERROR: Docker not found. Install: https://docs.docker.com/engine/install/"; \
		exit 1; \
	fi
	@echo "OK: Docker: $(shell docker --version)"
	@if ! command -v nvidia-smi >/dev/null 2>&1; then \
		echo "WARNING: nvidia-smi not found (GPU may not be available)"; \
	else \
		echo "OK: NVIDIA driver: $(shell nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"; \
	fi
	@if [ ! -x "$(FFMPEG_DIR)/bin/ffmpeg" ] || [ ! -x "$(FFMPEG_DIR)/bin/ffprobe" ]; then \
		echo "ERROR: FFmpeg sidecar not found at $(FFMPEG_DIR)/bin"; \
		echo "       cosmos-curator's Docker image does NOT bundle FFmpeg."; \
		echo "       Build / install a GPU-enabled FFmpeg under $(FFMPEG_DIR)/"; \
		echo "       so the layout is:"; \
		echo "         $(FFMPEG_DIR)/bin/ffmpeg"; \
		echo "         $(FFMPEG_DIR)/bin/ffprobe"; \
		echo "         $(FFMPEG_DIR)/lib/*.so"; \
		echo "       Override the location via:  make ... FFMPEG_DIR=/path"; \
		exit 1; \
	fi
	@echo "OK: FFmpeg sidecar: $$($(FFMPEG_DIR)/bin/ffmpeg -version 2>/dev/null | head -1)"
	@echo ""
	@echo "OK: All prerequisites met!"

# ---------------------------------------------------------------------------
# FFmpeg sidecar installer
# ---------------------------------------------------------------------------
# Drops a conda-forge LGPL FFmpeg into $(FFMPEG_DIR). This mirrors the
# currently deployed install (package ffmpeg-8.1.1-lgpl_ha074a71_801) so the
# Docker bind-mount at /opt/ffmpeg:ro picks up NVENC/NVDEC + libopenh264.
#
# Autodetects micromamba > mamba > conda. Fails fast with an install hint
# if none is available.

ffmpeg-install:
	@echo "Installing FFmpeg sidecar..."
	@echo "  Prefix:   $(FFMPEG_DIR)"
	@echo "  Channel:  $(FFMPEG_CHANNEL)"
	@echo "  Package:  $(FFMPEG_PACKAGE_SPEC)"
	@echo ""
	@# Single shell block so the existence-check / manager-detect / install
	@# all share state and an early-exit short-circuits the whole recipe
	@# (otherwise the `read -p` answer of N still falls through to install).
	@set -e; \
	if [ -x "$(FFMPEG_DIR)/bin/ffmpeg" ]; then \
		echo "WARNING: $(FFMPEG_DIR)/bin/ffmpeg already exists"; \
		echo "         Version: $$($(FFMPEG_DIR)/bin/ffmpeg -version 2>/dev/null | head -1)"; \
		read -p "Re-install over existing prefix? [y/N] " confirm; \
		if [ "$$confirm" != "y" ] && [ "$$confirm" != "Y" ]; then \
			echo "Cancelled (existing sidecar kept). Use 'make check-setup' to validate."; \
			exit 0; \
		fi; \
		case "$(FFMPEG_DIR)" in \
			/*) : ;; \
			*) echo "ERROR: FFMPEG_DIR must be an absolute path, got '$(FFMPEG_DIR)'"; exit 1 ;; \
		esac; \
		case "$(FFMPEG_DIR)" in \
			""|"/"|"$$HOME"|"$$HOME/") \
				echo "ERROR: refusing to 'rm -rf' unsafe FFMPEG_DIR='$(FFMPEG_DIR)'"; exit 1 ;; \
		esac; \
		echo "Removing existing prefix $(FFMPEG_DIR)..."; \
		rm -rf -- "$(FFMPEG_DIR)"; \
	fi; \
	if command -v micromamba >/dev/null 2>&1; then \
		mgr=micromamba; \
	elif command -v mamba >/dev/null 2>&1; then \
		mgr=mamba; \
	elif command -v conda >/dev/null 2>&1; then \
		mgr=conda; \
	else \
		echo "ERROR: none of micromamba / mamba / conda found on PATH."; \
		echo "       Install one of them before running 'make ffmpeg-install',"; \
		echo "       or install FFmpeg some other way and point FFMPEG_DIR at it."; \
		echo "  micromamba (recommended): curl -Ls https://micro.mamba.pm/install.sh | bash"; \
		exit 1; \
	fi; \
	echo "Using $$mgr"; \
	"$$mgr" create -y -p "$(FFMPEG_DIR)" -c "$(FFMPEG_CHANNEL)" "$(FFMPEG_PACKAGE_SPEC)"; \
	echo ""; \
	echo "FFmpeg sidecar installed."; \
	echo ""; \
	echo "Verify:"; \
	echo "  $(FFMPEG_DIR)/bin/ffmpeg -version | head -1"; \
	echo "  $(FFMPEG_DIR)/bin/ffmpeg -encoders | grep -E 'h264_nvenc|libopenh264'"; \
	echo ""; \
	echo "Next:"; \
	echo "  make check-setup"; \
	echo "  make run-pipeline CONFIG_FILE=configs/split.yaml MODELS_DIR=/path/to/models"

check-image:
	@if docker images $(IMAGE_NAME):$(IMAGE_TAG) --format "{{.Repository}}:{{.Tag}}" | grep -q "^$(IMAGE_NAME):$(IMAGE_TAG)$$"; then \
		echo "OK: $(IMAGE_NAME):$(IMAGE_TAG) found locally"; \
		echo "  Size: $$(docker images $(IMAGE_NAME):$(IMAGE_TAG) --format '{{.Size}}')"; \
		echo "  Created: $$(docker images $(IMAGE_NAME):$(IMAGE_TAG) --format '{{.CreatedSince}}')"; \
	else \
		echo "ERROR: $(IMAGE_NAME):$(IMAGE_TAG) not found locally."; \
		echo ""; \
		echo "Get it via one of:"; \
		echo "  make pull                                          # pull from $(REGISTRY)"; \
		echo "  make pull IMAGE_TAG=$(IMAGE_TAG)                   # pull the pinned tag"; \
		exit 1; \
	fi

# Soft preflight for Curator docker runs (warns on low disk; fails on missing deps).
check-curator-runtime:
	@echo "=== Curator runtime preflight ==="
	@echo "  DOCKER_NETWORK=$(DOCKER_NETWORK)"
	@echo "  MODELS_MOUNT_MODE=$(MODELS_MOUNT_MODE)"
	@echo "  MODELS_DIR=$(MODELS_DIR)"
	@echo "  CURATOR_TMP=$(if $(CURATOR_TMP),$(CURATOR_TMP),(unset — using container defaults))"
	@if ! command -v docker >/dev/null 2>&1; then \
		echo "ERROR: docker not found in PATH"; exit 1; \
	fi
	@if ! docker info >/dev/null 2>&1; then \
		echo "ERROR: cannot talk to Docker daemon (permission or daemon down)"; exit 1; \
	fi
	@if [ ! -x "$(FFMPEG_DIR)/bin/ffmpeg" ]; then \
		echo "ERROR: $(FFMPEG_DIR)/bin/ffmpeg not found."; \
		echo "       Run 'make ffmpeg-install' or 'make check-setup'."; \
		exit 1; \
	fi
	@if [ ! -d "$(MODELS_DIR)" ]; then \
		echo "ERROR: MODELS_DIR does not exist: $(MODELS_DIR)"; \
		echo "       Run: make download-models MODELS_DIR=$(MODELS_DIR)"; \
		exit 1; \
	fi
	@if ! command -v nvidia-smi >/dev/null 2>&1; then \
		echo "WARNING: nvidia-smi not found — GPU Curator stages may fail"; \
	else \
		nvidia-smi -L >/dev/null || echo "WARNING: nvidia-smi -L failed"; \
	fi
	@if [ "$(DOCKER_NETWORK)" = "host" ]; then \
		echo "WARNING: DOCKER_NETWORK=host can break Ray (metrics port clash on double init)."; \
		echo "         Prefer DOCKER_NETWORK=bridge (default) unless you need host networking."; \
	fi
	@if [ -n "$(CURATOR_TMP)" ]; then \
		mkdir -p "$(CURATOR_TMP)"; \
		avail=$$(df -BG --output=avail "$(CURATOR_TMP)" 2>/dev/null | tail -1 | tr -dc '0-9'); \
		if [ -n "$$avail" ] && [ "$$avail" -lt "$(CURATOR_MIN_TMP_GIB)" ]; then \
			echo "WARNING: CURATOR_TMP has only $${avail}GiB free (recommend >= $(CURATOR_MIN_TMP_GIB)GiB)"; \
		else \
			echo "OK: CURATOR_TMP free space check ($${avail:-?}GiB)"; \
		fi; \
	else \
		echo "NOTE: CURATOR_TMP unset — Ray/HF scratch uses the container filesystem."; \
		echo "      If Docker disk (/var/lib) is nearly full, set CURATOR_TMP to a large path."; \
	fi
	@root_avail=$$(df -BG --output=avail /var/lib/docker 2>/dev/null | tail -1 | tr -dc '0-9'); \
	if [ -z "$$root_avail" ]; then root_avail=$$(df -BG --output=avail /var/lib 2>/dev/null | tail -1 | tr -dc '0-9'); fi; \
	if [ -n "$$root_avail" ] && [ "$$root_avail" -lt "$(CURATOR_MIN_DOCKER_GIB)" ]; then \
		echo "WARNING: Docker storage has only $${root_avail}GiB free (recommend >= $(CURATOR_MIN_DOCKER_GIB)GiB)."; \
		echo "         Set CURATOR_TMP to a large filesystem and/or prune unused Docker data."; \
	fi
	@if [ -n "$(CONFIG_FILE)" ] && [ -f "$(CONFIG_FILE)" ]; then \
		echo "=== SeedVR2 checkpoint preflight (ENSURE_SEEDVR_CKPTS=$(ENSURE_SEEDVR_CKPTS)) ==="; \
		uv run python -m adapters.cosmos_curator.seedvr_ckpts preflight \
			--models-dir "$(MODELS_DIR)" \
			--config "$(CONFIG_FILE)" \
			--variant "$(SEEDVR_VARIANT)" \
			--ensure "$(ENSURE_SEEDVR_CKPTS)"; \
	elif [ "$(ENSURE_SEEDVR_CKPTS)" = "always" ]; then \
		echo "=== SeedVR2 checkpoint preflight (always) ==="; \
		uv run python -m adapters.cosmos_curator.seedvr_ckpts ensure \
			--models-dir "$(MODELS_DIR)" \
			--variant "$(SEEDVR_VARIANT)"; \
	else \
		echo "NOTE: CONFIG_FILE unset — skipping SeedVR2 ensure (set CONFIG_FILE or ENSURE_SEEDVR_CKPTS=always)."; \
	fi
	@echo "OK: curator runtime preflight complete"

# ---------------------------------------------------------------------------
# Primary path: pull pre-built image from a registry
# ---------------------------------------------------------------------------

setup: pull ffmpeg-install check-setup
	@echo ""
	@echo "Setup complete. Next:"
	@echo "  make download-models MODELS_DIR=/path/to/models"
	@echo "  make download-seedvr2 MODELS_DIR=/path/to/models   # if using super_resolution"
	@echo "  # or: make download-all-models MODELS_DIR=/path/to/models"
	@echo "  make run-pipeline CONFIG_FILE=configs/split.yaml MODELS_DIR=/path/to/models DATA_DIR=/path/to/data"

pull:
	@if [ -z "$(REGISTRY)" ]; then \
		echo "ERROR: REGISTRY is not set."; \
		echo ""; \
		echo "Usage:"; \
		echo "  make pull REGISTRY=<registry-url> [IMAGE_TAG=<tag>]"; \
		echo ""; \
		exit 1; \
	fi
	@echo ""
	@echo "=== Pulling cosmos-curator from registry ==="
	@echo ""
	@echo "  Source: $(REGISTRY):$(IMAGE_TAG)"
	@echo "  Local:  $(IMAGE_NAME):$(IMAGE_TAG)"
	@echo ""
	docker pull $(REGISTRY):$(IMAGE_TAG)
	docker tag $(REGISTRY):$(IMAGE_TAG) $(IMAGE_NAME):$(IMAGE_TAG)
	@echo ""
	@echo "Pull complete: $(IMAGE_NAME):$(IMAGE_TAG)"
	@echo "  Size: $$(docker images $(IMAGE_NAME):$(IMAGE_TAG) --format '{{.Size}}')"
	@echo ""
	@echo "Next steps:"
	@echo "  1. Set up HuggingFace authentication:"
	@echo "     echo 'your-hf-token' > $(HOME)/.config/cosmos_curator/hf_token.txt"
	@echo "  2. Download models: make download-models"
	@echo "  3. Run pipeline:    make run-pipeline CONFIG_FILE=my_split.yaml"

# ---------------------------------------------------------------------------
# Dataset Search — pull + compose + glue CLI (no image build in this repo)
# ---------------------------------------------------------------------------

pull-dataset-search:
	@profile="$(CDS_PROFILE)"; \
	registry="$(DATASET_SEARCH_REGISTRY)"; \
	tag="$(DATASET_SEARCH_TAG)"; \
	if [ -z "$$registry" ] || [ -z "$$tag" ]; then \
		if [ "$$profile" = "public" ]; then \
			registry="$${registry:-nvcr.io/nvidia/blueprint/cosmos-dataset-search}"; \
			tag="$${tag:-1.0.0}"; \
		fi; \
	fi; \
	if [ -z "$$registry" ] || [ -z "$$tag" ]; then \
		echo ""; \
		echo "=== Dataset Search pull (CDS_PROFILE=$$profile) ==="; \
		echo ""; \
		echo "DATASET_SEARCH_REGISTRY and DATASET_SEARCH_TAG are not set."; \
		echo "Set CDS runtime in .env (see .env.example):"; \
		echo "  CDS_PROFILE=public"; \
		echo "  DATASET_SEARCH_REGISTRY=nvcr.io/nvidia/blueprint/cosmos-dataset-search"; \
		echo "  DATASET_SEARCH_TAG=1.0.0"; \
		echo "  DATASET_SEARCH_IMAGE=cosmos-dataset-search"; \
		echo ""; \
		echo "Internal EA: CDS_PROFILE=internal + private/local tag (gitignored local.env)."; \
		echo "Then re-run: make pull-dataset-search"; \
		echo ""; \
		exit 1; \
	fi; \
	echo ""; \
	echo "=== Pulling Dataset Search (CDS_PROFILE=$$profile) ==="; \
	echo ""; \
	echo "  Source: $$registry:$$tag"; \
	echo "  Local:  $(DATASET_SEARCH_IMAGE):$$tag"; \
	echo ""; \
	docker pull "$$registry:$$tag"; \
	docker tag "$$registry:$$tag" "$(DATASET_SEARCH_IMAGE):$$tag"; \
	echo ""; \
	echo "Pull complete: $(DATASET_SEARCH_IMAGE):$$tag"; \
	echo "  Next: make up-dataset-search  (or point CDS_URL at an upstream CDS stack)"

pull-all: pull pull-dataset-search pull-data-mining
	@echo ""
	@echo "All three images pulled (Curator + Dataset Search + Data Mining)."
	@echo "Profiles: CDS_PROFILE=$(CDS_PROFILE) — see .env.example"
	@echo "Next: make run-pipeline / make ds-health / make run-data-mining-select"

tag-local-images:
	@if [ -z "$(DATASET_SEARCH_TAG)" ] || [ -z "$(DATA_MINING_TAG)" ]; then \
		echo "ERROR: DATASET_SEARCH_TAG and DATA_MINING_TAG must be set (see local.env)."; \
		exit 1; \
	fi
	@set -e; \
	src_cds="$(LOCAL_CDS_SOURCE_IMAGE):$(LOCAL_CDS_SOURCE_TAG)"; \
	dst_cds="$(DATASET_SEARCH_IMAGE):$(DATASET_SEARCH_TAG)"; \
	src_dm="$(LOCAL_DATA_MINING_SOURCE_IMAGE):$(LOCAL_DATA_MINING_SOURCE_TAG)"; \
	dst_dm="$(DATA_MINING_IMAGE):$(DATA_MINING_TAG)"; \
	for pair in "$$src_cds $$dst_cds" "$$src_dm $$dst_dm"; do \
		src=$${pair%% *}; dst=$${pair#* }; \
		if ! docker image inspect "$$src" >/dev/null 2>&1; then \
			echo "ERROR: source image not found: $$src"; \
			echo "       Pull tao-toolkit / build CDS, or fix LOCAL_* in local.env"; \
			exit 1; \
		fi; \
		docker tag "$$src" "$$dst"; \
		echo "Tagged $$src → $$dst"; \
	done
	@echo "OK: local integration tags ready ($(DATASET_SEARCH_IMAGE):$(DATASET_SEARCH_TAG), $(DATA_MINING_IMAGE):$(DATA_MINING_TAG))"

check-dataset-search-image:
	@if [ -z "$(DATASET_SEARCH_TAG)" ]; then \
		echo "ERROR: DATASET_SEARCH_TAG is not set (see .env.example)."; \
		echo "       Run: make pull-dataset-search after setting DATASET_SEARCH_REGISTRY"; \
		exit 1; \
	fi
	@if ! docker image inspect "$(DATASET_SEARCH_IMAGE):$(DATASET_SEARCH_TAG)" >/dev/null 2>&1; then \
		echo "ERROR: $(DATASET_SEARCH_IMAGE):$(DATASET_SEARCH_TAG) not found locally."; \
		echo "       Set DATASET_SEARCH_REGISTRY + DATASET_SEARCH_TAG in .env, then:"; \
		echo "       make pull-dataset-search"; \
		exit 1; \
	fi
	@echo "OK: $(DATASET_SEARCH_IMAGE):$(DATASET_SEARCH_TAG)"

up-dataset-search: check-dataset-search-image
	@echo "Starting Dataset Search stack (Milvus + pulled image only)..."
	DATASET_SEARCH_IMAGE="$(DATASET_SEARCH_IMAGE)" \
	DATASET_SEARCH_TAG="$(DATASET_SEARCH_TAG)" \
	COSMOS_EMBED_NIM_URI="$(COSMOS_EMBED_NIM_URI)" \
	CRADIO_NIM_URI="$(CRADIO_NIM_URI)" \
	ALLOWED_PIPELINES="$(ALLOWED_PIPELINES)" \
		docker compose -f "$(DATASET_SEARCH_COMPOSE)" up -d
	@echo "Dataset Search API: $(DATASET_SEARCH_URL)"

down-dataset-search:
	docker compose -f "$(DATASET_SEARCH_COMPOSE)" down

# CDS / glue helpers call the internal Python entrypoint (not operator-facing).
PIPELINE ?= cosmos_video_search_milvus
NAME ?=

ds-health:
	DATASET_SEARCH_URL="$(DATASET_SEARCH_URL)" CDS_URL="$(CDS_URL)" \
		uv run $(PAIDF_CLI) ds health --cds-url "$(CDS_URL)"

ds-pipelines:
	DATASET_SEARCH_URL="$(DATASET_SEARCH_URL)" CDS_URL="$(CDS_URL)" \
		uv run $(PAIDF_CLI) ds pipelines --cds-url "$(CDS_URL)"

ds-collections:
	DATASET_SEARCH_URL="$(DATASET_SEARCH_URL)" CDS_URL="$(CDS_URL)" \
		uv run $(PAIDF_CLI) ds collections --cds-url "$(CDS_URL)"

ds-create-collection:
	@if [ -z "$(NAME)" ]; then \
		echo "ERROR: NAME= required (e.g. NAME=traffic)"; \
		exit 1; \
	fi
	DATASET_SEARCH_URL="$(DATASET_SEARCH_URL)" CDS_URL="$(CDS_URL)" \
		uv run $(PAIDF_CLI) ds create-collection \
		--name "$(NAME)" \
		--pipeline "$(PIPELINE)" \
		$(if $(COLLECTION_ID),--id "$(COLLECTION_ID)",) \
		--cds-url "$(CDS_URL)"

ds-get-collection:
	@if [ -z "$(COLLECTION)" ]; then \
		echo "ERROR: COLLECTION= required (name or id)"; \
		exit 1; \
	fi
	DATASET_SEARCH_URL="$(DATASET_SEARCH_URL)" CDS_URL="$(CDS_URL)" \
		uv run $(PAIDF_CLI) ds get-collection "$(COLLECTION)" --cds-url "$(CDS_URL)"

ingest-curator:
	@if [ -z "$(CURATOR_DIR)" ]; then \
		echo "ERROR: CURATOR_DIR= required (iv2_embd_parquet/ or ce1_embd*_parquet/)"; \
		exit 1; \
	fi
	uv run $(PAIDF_CLI) ingest-curator \
		--curator-dir "$(CURATOR_DIR)" \
		--output-parquet "$(OUT)" \
		--embedding-backend "$(EMBEDDING_BACKEND)" \
		$(if $(COLLECTION),--collection "$(COLLECTION)" --ingest,--convert-only) \
		$(if $(filter 1 true yes,$(ALLOW_LAB_DOCUMENT_FALLBACK)),--allow-lab-document-fallback,)

stage-cds-artifact:
	@test -n "$(STAGE_SOURCE)" || { echo "ERROR: STAGE_SOURCE= required"; exit 1; }
	@test -n "$(STAGE_DESTINATION)" || { echo "ERROR: STAGE_DESTINATION= required"; exit 1; }
	uv run $(PAIDF_CLI) integration stage-artifact \
		--source "$(STAGE_SOURCE)" \
		--destination "$(STAGE_DESTINATION)" \
		--access-key-env "$(OBJECT_STORE_ACCESS_KEY_ENV)" \
		--secret-key-env "$(OBJECT_STORE_SECRET_KEY_ENV)" \
		$(if $(OBJECT_STORE_ENDPOINT),--endpoint-url "$(OBJECT_STORE_ENDPOINT)",) \
		$(if $(filter 1 true yes,$(ALLOW_LAB_HTTP_ENDPOINT)),--allow-lab-http-endpoint,)

ds-bulk-insert:
	@test -n "$(COLLECTION)" || { echo "ERROR: COLLECTION= required"; exit 1; }
	@test -n "$(PARQUET_URI)" || { echo "ERROR: PARQUET_URI= required"; exit 1; }
	@test -n "$(CDS_EMBEDDING_FAMILY)" || { echo "ERROR: CDS_EMBEDDING_FAMILY=ce1 required"; exit 1; }
	uv run $(PAIDF_CLI) ds bulk-insert \
		--collection "$(COLLECTION)" \
		--embedding-family "$(CDS_EMBEDDING_FAMILY)" \
		--parquet "$(PARQUET_URI)" \
		--access-key-env "$(OBJECT_STORE_ACCESS_KEY_ENV)" \
		--secret-key-env "$(OBJECT_STORE_SECRET_KEY_ENV)" \
		$(if $(OBJECT_STORE_ENDPOINT),--endpoint-url "$(OBJECT_STORE_ENDPOINT)",) \
		$(if $(filter 1 true yes,$(ALLOW_LAB_HTTP_ENDPOINT)),--allow-lab-http-endpoint,) \
		--cds-url "$(CDS_URL)"

ds-job-status:
	@test -n "$(JOB_ID)" || { echo "ERROR: JOB_ID= required"; exit 1; }
	uv run $(PAIDF_CLI) ds job-status "$(JOB_ID)" \
		$(if $(filter 1 true yes,$(WAIT)),--wait,--no-wait) \
		--timeout-seconds "$(POLL_TIMEOUT_SECONDS)" \
		--poll-interval-seconds "$(POLL_INTERVAL_SECONDS)" \
		--cds-url "$(CDS_URL)"

caption-build:
	@test -n "$(CAPTION_JSON)" || { echo "ERROR: CAPTION_JSON= required"; exit 1; }
	@test -n "$(CAPTION_PARQUET)" || { echo "ERROR: CAPTION_PARQUET= required"; exit 1; }
	uv run $(PAIDF_CLI) integration captions build \
		--input-json "$(CAPTION_JSON)" \
		--output-parquet "$(CAPTION_PARQUET)" \
		$(if $(INDEXED_ID_FILE),--indexed-id-file "$(INDEXED_ID_FILE)",)

caption-validate:
	@test -n "$(CAPTION_PARQUET)" || { echo "ERROR: CAPTION_PARQUET= required"; exit 1; }
	uv run $(PAIDF_CLI) integration captions validate \
		--parquet "$(CAPTION_PARQUET)" \
		$(if $(INDEXED_ID_FILE),--indexed-id-file "$(INDEXED_ID_FILE)",)

caption-readiness:
	@test -n "$(PIPELINE)" || { echo "ERROR: PIPELINE= required"; exit 1; }
	uv run $(PAIDF_CLI) integration captions readiness \
		--cds-url "$(CDS_URL)" \
		--cds-profile "$(CDS_PROFILE)" \
		--pipeline "$(PIPELINE)"

caption-search:
	@test -n "$(CAPTION_QUERY)" || { echo "ERROR: CAPTION_QUERY= required"; exit 1; }
	uv run $(PAIDF_CLI) integration captions search \
		--query "$(CAPTION_QUERY)" \
		--limit "$(CAPTION_SEARCH_LIMIT)" \
		$(if $(CAPTION_SEARCH_DATA_SOURCE),--data-source "$(CAPTION_SEARCH_DATA_SOURCE)",) \
		--cds-url "$(CDS_URL)"

caption-upload:
	@test -n "$(CAPTION_PARQUET)" || { echo "ERROR: CAPTION_PARQUET= required"; exit 1; }
	uv run $(PAIDF_CLI) integration captions upload \
		--parquet "$(CAPTION_PARQUET)" \
		--model-name "$(CAPTION_MODEL_NAME)" \
		--data-source "$(CAPTION_DATA_SOURCE)" \
		$(if $(INDEXED_ID_FILE),--indexed-id-file "$(INDEXED_ID_FILE)",) \
		--cds-url "$(CDS_URL)"

caption-bulk-insert:
	@test -n "$(PARQUET_URI)" || { echo "ERROR: PARQUET_URI= required"; exit 1; }
	uv run $(PAIDF_CLI) integration captions bulk-insert \
		--parquet "$(PARQUET_URI)" \
		--access-key-env "$(OBJECT_STORE_ACCESS_KEY_ENV)" \
		--secret-key-env "$(OBJECT_STORE_SECRET_KEY_ENV)" \
		$(if $(OBJECT_STORE_ENDPOINT),--endpoint-url "$(OBJECT_STORE_ENDPOINT)",) \
		$(if $(filter 1 true yes,$(ALLOW_LAB_HTTP_ENDPOINT)),--allow-lab-http-endpoint,) \
		--cds-url "$(CDS_URL)"

search:
	@if [ -z "$(COLLECTION)" ] || [ -z "$(QUERY)" ]; then \
		echo "ERROR: COLLECTION= and QUERY= required"; \
		exit 1; \
	fi
	DATASET_SEARCH_URL="$(DATASET_SEARCH_URL)" CDS_URL="$(CDS_URL)" \
		uv run $(PAIDF_CLI) search --collection "$(COLLECTION)" --query "$(QUERY)"

# ---------------------------------------------------------------------------
# Data Mining — pull + DivKNN job (no image build in this repo)
# ---------------------------------------------------------------------------

prepare-cds-ce1-for-tdm:
	@test -n "$(DATA_DIR)" || { echo "ERROR: DATA_DIR= required"; exit 1; }
	@test -n "$(CDS_CE1_TARGET_SELECTION)" || { echo "ERROR: CDS_CE1_TARGET_SELECTION= required"; exit 1; }
	@test -n "$(CDS_CE1_SOURCE_SELECTION)" || { echo "ERROR: CDS_CE1_SOURCE_SELECTION= required"; exit 1; }
	@test -n "$(CDS_EMBEDDING_FAMILY)" || { echo "ERROR: CDS_EMBEDDING_FAMILY=ce1 required"; exit 1; }
	uv run $(PAIDF_CLI) integration prepare-cds-ce1-for-tdm \
		--data-dir "$(DATA_DIR)" \
		--target-selection "$(CDS_CE1_TARGET_SELECTION)" \
		--source-selection "$(CDS_CE1_SOURCE_SELECTION)" \
		--output-subdir "$(TMM_PREP_SUBDIR)" \
		--embedding-family "$(CDS_EMBEDDING_FAMILY)"

pull-data-mining:
	@if [ -z "$(DATA_MINING_REGISTRY)" ] || [ -z "$(DATA_MINING_TAG)" ]; then \
		echo ""; \
		echo "=== Data Mining pull ==="; \
		echo ""; \
		echo "DATA_MINING_REGISTRY and DATA_MINING_TAG are not set."; \
		echo "This repo pulls the published TAO Toolkit DS image only — it does not build it."; \
		echo ""; \
		echo "Add to .env (see .env.example):"; \
		echo "  DATA_MINING_REGISTRY=nvcr.io/nvidia/tao/tao-toolkit"; \
		echo "  DATA_MINING_TAG=7.2.0-data-services"; \
		echo "  DATA_MINING_IMAGE=tao-toolkit"; \
		echo ""; \
		echo "Then re-run: make pull-data-mining"; \
		echo ""; \
		exit 1; \
	fi
	@echo ""
	@echo "=== Pulling Data Mining (TAO Toolkit DS) from registry ==="
	@echo ""
	@echo "  Source: $(DATA_MINING_REGISTRY):$(DATA_MINING_TAG)"
	@echo "  Local:  $(DATA_MINING_IMAGE):$(DATA_MINING_TAG)"
	@echo ""
	docker pull "$(DATA_MINING_REGISTRY):$(DATA_MINING_TAG)"
	docker tag "$(DATA_MINING_REGISTRY):$(DATA_MINING_TAG)" \
		"$(DATA_MINING_IMAGE):$(DATA_MINING_TAG)"
	@echo ""
	@echo "Pull complete: $(DATA_MINING_IMAGE):$(DATA_MINING_TAG)"
	@echo "  Mining CLI: tmm nearest_neighbors | tmm unique_neighbor_matching"
	@echo "  Next: make run-data-mining-select DATA_DIR=..."
	@echo "        make run-data-mining-unique-match DATA_DIR=... DESIRED_UNIQUE_COUNT=..."

check-data-mining-image:
	@if [ -z "$(DATA_MINING_TAG)" ]; then \
		echo "ERROR: DATA_MINING_TAG is not set (see .env.example)."; \
		echo "       Run: make pull-data-mining after setting DATA_MINING_REGISTRY"; \
		exit 1; \
	fi
	@if ! docker image inspect "$(DATA_MINING_IMAGE):$(DATA_MINING_TAG)" >/dev/null 2>&1; then \
		if docker image inspect "$(DATA_MINING_REGISTRY):$(DATA_MINING_TAG)" >/dev/null 2>&1; then \
			docker tag "$(DATA_MINING_REGISTRY):$(DATA_MINING_TAG)" \
				"$(DATA_MINING_IMAGE):$(DATA_MINING_TAG)"; \
		else \
			echo "ERROR: $(DATA_MINING_IMAGE):$(DATA_MINING_TAG) not found locally."; \
			echo "       Set DATA_MINING_REGISTRY + DATA_MINING_TAG in .env, then:"; \
			echo "       make pull-data-mining"; \
			exit 1; \
		fi; \
	fi
	@echo "OK: $(DATA_MINING_IMAGE):$(DATA_MINING_TAG)"

run-data-mining-select: check-data-mining-image
	@if [ -z "$(DATA_DIR)" ]; then \
		echo "ERROR: DATA_DIR= required (mounted at /data; use dirs or .parquet under TARGET/SOURCE_SUBDIR)"; \
		exit 1; \
	fi
	@echo "Running tmm nearest_neighbors in $(DATA_MINING_IMAGE):$(DATA_MINING_TAG)..."
	@echo "  docker-equivalent: --gpus $(GPUS) --shm-size=$(DATA_MINING_SHM_SIZE) -v $(DATA_DIR):/data --entrypoint tmm"
	DATA_MINING_IMAGE="$(DATA_MINING_IMAGE)" DATA_MINING_TAG="$(DATA_MINING_TAG)" \
		uv run $(PAIDF_CLI) data-mining-select \
		--data-dir "$(DATA_DIR)" \
		$(if $(TMM_CONFIG_FILE),--config-file "$(TMM_CONFIG_FILE)",) \
		--target-subdir "$(TARGET_SUBDIR)" \
		--source-subdir "$(SOURCE_SUBDIR)" \
		--output-subdir "$(OUTPUT_SUBDIR)" \
		--topn "$(TOPN)" \
		--metric "$(DATA_MINING_METRIC)" \
		--distance-threshold "$(DISTANCE_THRESHOLD)" \
		$(if $(filter 1 true TRUE yes YES,$(FILTER_BY_LABEL)),--filter-by-label,--no-filter-by-label) \
		--source-embed-column-name "$(SOURCE_EMBED_COLUMN_NAME)" \
		--target-embed-column-name "$(TARGET_EMBED_COLUMN_NAME)" \
		--embedding-backend "$(TDM_EMBEDDING_BACKEND)" \
		--gpus "$(GPUS)" \
		--shm-size "$(DATA_MINING_SHM_SIZE)" \
		--image "$(DATA_MINING_IMAGE):$(DATA_MINING_TAG)" \
		--no-dry-run

run-data-mining-unique-match: check-data-mining-image
	@if [ -z "$(DATA_DIR)" ]; then \
		echo "ERROR: DATA_DIR= required (mounted at /data; use dirs or .parquet under TARGET/SOURCE_SUBDIR)"; \
		exit 1; \
	fi
	@echo "Running tmm unique_neighbor_matching in $(DATA_MINING_IMAGE):$(DATA_MINING_TAG)..."
	@echo "  docker-equivalent: --gpus $(GPUS) --shm-size=$(DATA_MINING_SHM_SIZE) -v $(DATA_DIR):/data --entrypoint tmm"
	DATA_MINING_IMAGE="$(DATA_MINING_IMAGE)" DATA_MINING_TAG="$(DATA_MINING_TAG)" \
		uv run $(PAIDF_CLI) data-mining-unique-match \
		--data-dir "$(DATA_DIR)" \
		$(if $(TMM_UNM_CONFIG_FILE),--config-file "$(TMM_UNM_CONFIG_FILE)",) \
		--target-subdir "$(TARGET_SUBDIR)" \
		--source-subdir "$(SOURCE_SUBDIR)" \
		--output-subdir "$(UNM_OUTPUT_SUBDIR)" \
		--desired-unique-count "$(DESIRED_UNIQUE_COUNT)" \
		--allocation-policy "$(ALLOCATION_POLICY)" \
		--metric "$(DATA_MINING_METRIC)" \
		--candidate-expansion-factor "$(CANDIDATE_EXPANSION_FACTOR)" \
		--source-embedding-column "$(SOURCE_EMBEDDING_COLUMN)" \
		--target-embedding-column "$(TARGET_EMBEDDING_COLUMN)" \
		--source-filepath-column "$(SOURCE_FILEPATH_COLUMN)" \
		--target-filepath-column "$(TARGET_FILEPATH_COLUMN)" \
		$(if $(EXCLUDE_PATH),--exclude-path "$(EXCLUDE_PATH)",) \
		$(if $(SOURCE_DETECTION_FILE),--source-detection-file "$(SOURCE_DETECTION_FILE)",) \
		$(if $(TARGET_DETECTION_FILE),--target-detection-file "$(TARGET_DETECTION_FILE)",) \
		$(if $(DETECTION_FORMAT),--detection-format "$(DETECTION_FORMAT)",) \
		--rare-class-list "$(RARE_CLASS_LIST)" \
		$(if $(filter 1 true TRUE yes YES,$(SAVE_EMBEDDINGS)),--save-embeddings,--no-save-embeddings) \
		$(if $(filter 1 true TRUE yes YES,$(VISUALIZE)),--visualize,--no-visualize) \
		--embedding-backend "$(TDM_EMBEDDING_BACKEND)" \
		--gpus "$(GPUS)" \
		--shm-size "$(DATA_MINING_SHM_SIZE)" \
		--image "$(DATA_MINING_IMAGE):$(DATA_MINING_TAG)" \
		--no-dry-run

image-embeddings-build:
	@test -n "$(DATA_DIR)" || { echo "ERROR: DATA_DIR= required"; exit 1; }
	@test -n "$(IMAGE_EMBEDDING_JSON)" || { echo "ERROR: IMAGE_EMBEDDING_JSON= required"; exit 1; }
	@test -n "$(IMAGE_EMBEDDING_INPUT)" || { echo "ERROR: IMAGE_EMBEDDING_INPUT= required"; exit 1; }
	uv run $(PAIDF_CLI) integration image-embeddings build \
		--input-json "$(IMAGE_EMBEDDING_JSON)" \
		--data-dir "$(DATA_DIR)" \
		--output-parquet "$(IMAGE_EMBEDDING_INPUT)"

image-embeddings-validate-input:
	@test -n "$(DATA_DIR)" || { echo "ERROR: DATA_DIR= required"; exit 1; }
	@test -n "$(IMAGE_EMBEDDING_INPUT)" || { echo "ERROR: IMAGE_EMBEDDING_INPUT= required"; exit 1; }
	uv run $(PAIDF_CLI) integration image-embeddings validate-input \
		--parquet "$(IMAGE_EMBEDDING_INPUT)" \
		--data-dir "$(DATA_DIR)"

run-image-embeddings: check-data-mining-image
	@test -n "$(DATA_DIR)" || { echo "ERROR: DATA_DIR= required"; exit 1; }
	uv run $(PAIDF_CLI) integration image-embeddings run \
		--data-dir "$(DATA_DIR)" \
		$(if $(IMAGE_EMBEDDING_CONFIG),--config-file "$(IMAGE_EMBEDDING_CONFIG)",\
			--input-parquet "$(IMAGE_EMBEDDING_INPUT)" \
			--output-parquet "$(IMAGE_EMBEDDING_OUTPUT)" \
			--model-type "$(IMAGE_EMBEDDING_MODEL_TYPE)" \
			--model-name-or-path "$(IMAGE_EMBEDDING_MODEL)" \
			$(if $(IMAGE_EMBEDDING_MODEL_CONFIG),--model-config-path "$(IMAGE_EMBEDDING_MODEL_CONFIG)",)) \
		--batch-size "$(IMAGE_EMBEDDING_BATCH_SIZE)" \
		--gpus "$(GPUS)" \
		--shm-size "$(DATA_MINING_SHM_SIZE)" \
		--image "$(DATA_MINING_IMAGE):$(DATA_MINING_TAG)" \
		$(if $(filter 1 true yes,$(IMAGE_EMBEDDING_DRY_RUN)),--dry-run,--no-dry-run)

image-embeddings-validate-output:
	@test -n "$(DATA_DIR)" || { echo "ERROR: DATA_DIR= required"; exit 1; }
	@test -n "$(IMAGE_EMBEDDING_INPUT)" || { echo "ERROR: IMAGE_EMBEDDING_INPUT= required"; exit 1; }
	@test -n "$(IMAGE_EMBEDDING_OUTPUT)" || { echo "ERROR: IMAGE_EMBEDDING_OUTPUT= required"; exit 1; }
	uv run $(PAIDF_CLI) integration image-embeddings validate-output \
		--input-parquet "$(IMAGE_EMBEDDING_INPUT)" \
		--output-parquet "$(IMAGE_EMBEDDING_OUTPUT)" \
		--data-dir "$(DATA_DIR)"

text-embeddings-validate-input:
	@test -n "$(DATA_DIR)" || { echo "ERROR: DATA_DIR= required"; exit 1; }
	@test -n "$(TEXT_EMBEDDING_INPUT)" || { echo "ERROR: TEXT_EMBEDDING_INPUT= required"; exit 1; }
	uv run $(PAIDF_CLI) integration text-embeddings validate-input \
		--parquet "$(TEXT_EMBEDDING_INPUT)" \
		--data-dir "$(DATA_DIR)"

run-text-embeddings: check-data-mining-image
	@test -n "$(DATA_DIR)" || { echo "ERROR: DATA_DIR= required"; exit 1; }
	uv run $(PAIDF_CLI) integration text-embeddings run \
		--data-dir "$(DATA_DIR)" \
		$(if $(TEXT_EMBEDDING_CONFIG),--config-file "$(TEXT_EMBEDDING_CONFIG)",\
			--input-parquet "$(TEXT_EMBEDDING_INPUT)" \
			--output-parquet "$(TEXT_EMBEDDING_OUTPUT)" \
			--model "$(TEXT_EMBEDDING_MODEL)" \
			--model-path "$(TEXT_EMBEDDING_MODEL_PATH)") \
		--batch-size "$(TEXT_EMBEDDING_BATCH_SIZE)" \
		--gpus "$(GPUS)" \
		--shm-size "$(DATA_MINING_SHM_SIZE)" \
		--image "$(DATA_MINING_IMAGE):$(DATA_MINING_TAG)" \
		$(if $(filter 1 true yes,$(TEXT_EMBEDDING_DRY_RUN)),--dry-run,--no-dry-run)

text-embeddings-validate-output:
	@test -n "$(DATA_DIR)" || { echo "ERROR: DATA_DIR= required"; exit 1; }
	@test -n "$(TEXT_EMBEDDING_INPUT)" || { echo "ERROR: TEXT_EMBEDDING_INPUT= required"; exit 1; }
	@test -n "$(TEXT_EMBEDDING_OUTPUT)" || { echo "ERROR: TEXT_EMBEDDING_OUTPUT= required"; exit 1; }
	uv run $(PAIDF_CLI) integration text-embeddings validate-output \
		--input-parquet "$(TEXT_EMBEDDING_INPUT)" \
		--output-parquet "$(TEXT_EMBEDDING_OUTPUT)" \
		--data-dir "$(DATA_DIR)"

# ---------------------------------------------------------------------------
# Optional developer path: build from upstream source outside the git repo
# ---------------------------------------------------------------------------

clone-curator:
	@if [ -d "$(COSMOS_REPO)/.git" ]; then \
		echo "OK: Cosmos-Curator source already exists at $(COSMOS_REPO)"; \
	else \
		echo "Cloning Cosmos-Curator into ignored path: $(COSMOS_REPO)"; \
		mkdir -p "$$(dirname "$(COSMOS_REPO)")"; \
		git clone --recurse-submodules "$(COSMOS_REPO_URL)" "$(COSMOS_REPO)"; \
	fi

build-dry-run:
	@echo "Running build dry-run (generating Dockerfile only)..."
	@if ! command -v poetry >/dev/null 2>&1; then \
		echo "ERROR: Poetry not found (required for building from source)"; \
		echo "  Install: curl -sSL https://install.python-poetry.org | python3 -"; \
		exit 1; \
	fi
	@if [ ! -d "$(COSMOS_REPO)" ]; then \
		echo "ERROR: Cosmos-Curator repository not found at: $(COSMOS_REPO)"; \
		echo "  Run: make clone-curator"; \
		exit 1; \
	fi
	@echo "OK: Cosmos-Curator repo: $(COSMOS_REPO)"
	cd $(COSMOS_REPO) && \
		PYTHONNOUSERSITE=1 PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring \
		poetry run cosmos-curator image build \
			--image-name $(IMAGE_NAME) \
			--image-tag $(IMAGE_TAG) \
			--envs $(COSMOS_IMAGE_ENVS) \
			--dry-run --verbose

build: check-build
	@echo ""
	@echo "=== Building cosmos-curator from external source checkout ==="
	@echo ""
	@if ! command -v poetry >/dev/null 2>&1; then \
		echo "ERROR: Poetry not found (required for building from source)"; \
		echo "  Install: curl -sSL https://install.python-poetry.org | python3 -"; \
		exit 1; \
	fi
	@if [ ! -d "$(COSMOS_REPO)" ]; then \
		echo "ERROR: Cosmos-Curator repository not found at: $(COSMOS_REPO)"; \
		echo "       Run: make clone-curator"; \
		exit 1; \
	fi
	@if [ ! -f "$(COSMOS_REPO)/pyproject.toml" ]; then \
		echo "ERROR: Source checkout missing pyproject.toml"; \
		exit 1; \
	fi
	@echo "Building cosmos-curator Docker image..."
	@echo "  Source: $(COSMOS_REPO)"
	@echo "  Image:  $(IMAGE_NAME):$(IMAGE_TAG)"
	@echo "  This will take 15-30 minutes and use ~50GB disk space."
	@echo ""
	@echo "Step 1/2: Installing cosmos-curator CLI (poetry install)..."
	cd $(COSMOS_REPO) && \
		PYTHONNOUSERSITE=1 PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring \
		poetry install --extras=local
	@echo ""
	@echo "Step 2/2: Building Docker image..."
	cd $(COSMOS_REPO) && \
		PYTHONNOUSERSITE=1 PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring \
		poetry run cosmos-curator image build \
			--image-name $(IMAGE_NAME) \
			--image-tag $(IMAGE_TAG) \
			--envs $(COSMOS_IMAGE_ENVS)
	@echo ""
	@echo "Build complete: $(IMAGE_NAME):$(IMAGE_TAG)"
	@echo ""
	@echo "Next steps:"
	@echo "  1. Set up HuggingFace authentication:"
	@echo "     echo 'your-hf-token' > $(HOME)/.config/cosmos_curator/hf_token.txt"
	@echo "  2. Download models: make download-models"
	@echo "  3. Run pipeline:    make run-pipeline CONFIG_FILE=my_split.yaml"

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

run-pipeline: check-image check-curator-runtime
	@if [ ! -r "$(CONFIG_FILE)" ]; then \
		echo "ERROR: Config file '$(CONFIG_FILE)' not found or not readable"; \
		echo "       Pass one via: make run-pipeline CONFIG_FILE=configs/split.yaml"; \
		exit 1; \
	fi
	@echo "Running cosmos-curator pipeline..."
	@echo "  Image:  $(IMAGE_NAME):$(IMAGE_TAG)"
	@echo "  Config: $(CONFIG_FILE)"
	@echo "  Models: $(MODELS_DIR) (mount=$(MODELS_MOUNT_MODE))"
	@echo "  FFmpeg: $(FFMPEG_DIR)"
	@echo "  GPUs:   $(GPUS)"
	@echo "  SHM:    $(SHM_SIZE)"
	@echo "  Net:    $(DOCKER_NETWORK)"
	@if [ -n "$(CURATOR_TMP)" ]; then echo "  Tmp:    $(CURATOR_TMP)"; fi
	@if [ -n "$(DATA_DIR)" ]; then echo "  Data:   $(DATA_DIR)"; mkdir -p "$(DATA_DIR)/output"; fi
	@if [ ! -x "$(FFMPEG_DIR)/bin/ffmpeg" ]; then \
		echo "ERROR: $(FFMPEG_DIR)/bin/ffmpeg not found."; \
		echo "       Run 'make check-setup' for the sidecar installation hint."; \
		exit 1; \
	fi
	@SEEDVR_DOCKER_ARGS=$$(uv run python -m adapters.cosmos_curator.seedvr_ckpts preflight \
		--models-dir "$(MODELS_DIR)" \
		--config "$(CONFIG_FILE)" \
		--variant "$(SEEDVR_VARIANT)" \
		--ensure skip \
		--print-docker-args 2>/dev/null || true); \
	if [ -n "$$SEEDVR_DOCKER_ARGS" ]; then echo "  SeedVR: $(MODELS_DIR)/seedvr2 -> $(SEEDVR_CONTAINER_CKPTS)"; fi; \
	COSMOS_CFG_MOUNT=$$(uv run python -m adapters.cosmos_curator.openai_config_env prepare \
		--output-dir "$(COSMOS_CURATOR_CONFIG_GEN_DIR)" \
		--merge-from "$(COSMOS_CURATOR_CONFIG_DIR)" \
		--print-docker-mount); \
	if [ -z "$$COSMOS_CFG_MOUNT" ] && [ -d "$(COSMOS_CURATOR_CONFIG_DIR)" ]; then \
		COSMOS_CFG_MOUNT="-v $(COSMOS_CURATOR_CONFIG_DIR):/cosmos_curator/config:ro"; \
	fi; \
	if [ -n "$$COSMOS_CFG_MOUNT" ]; then echo "  Curator cfg: env/host → /cosmos_curator/config"; fi; \
	echo ""; \
	docker run --gpus '$(GPUS)' --rm \
		$(MODELS_VOLUME_ARGS) \
		-v "$(abspath $(CONFIG_FILE)):/config/pipeline_config.yaml:ro" \
		$$COSMOS_CFG_MOUNT \
		$(FFMPEG_MOUNT) \
		$(if $(DATA_DIR),-v "$(DATA_DIR):$(DATA_DIR)") \
		$(if $(DATA_DIR),-v "$(DATA_DIR):/data") \
		$(HF_ENV_ARGS) \
		$(CURATOR_TMP_ARGS) \
		$(DOCKER_NETWORK_ARGS) \
		--shm-size=$(SHM_SIZE) \
		$$SEEDVR_DOCKER_ARGS \
		$(EXTRA_DOCKER_ARGS) \
		$(IMAGE_NAME):$(IMAGE_TAG) \
		bash -c "cd /opt/cosmos-curator && pixi run -e $(PIXI_ENV) --as-is python -m cosmos_curator.pipelines.video.run_pipeline /config/pipeline_config.yaml"

# ---------------------------------------------------------------------------
# Run (image annotate pipeline)
# ---------------------------------------------------------------------------
# Upstream image runner supports config-file mode (same as video):
#   python -m cosmos_curator.pipelines.image.run_pipeline /config/pipeline_config.yaml
# Requires pipeline: annotate in the YAML. Override IMAGE_CONFIG_FILE to select
# a different config (defaults to configs/image.yaml).

IMAGE_CONFIG_FILE ?= configs/image.yaml

run_image_pipeline: check-image check-curator-runtime
	@if [ ! -r "$(IMAGE_CONFIG_FILE)" ]; then \
		echo "ERROR: Image config file '$(IMAGE_CONFIG_FILE)' not found or not readable"; \
		echo "       Pass one via: make run_image_pipeline IMAGE_CONFIG_FILE=configs/image.yaml"; \
		exit 1; \
	fi
	@echo "Running cosmos-curator image annotate pipeline..."
	@echo "  Image:  $(IMAGE_NAME):$(IMAGE_TAG)"
	@echo "  Config: $(IMAGE_CONFIG_FILE)"
	@echo "  Models: $(MODELS_DIR) (mount=$(MODELS_MOUNT_MODE))"
	@echo "  FFmpeg: $(FFMPEG_DIR)"
	@echo "  GPUs:   $(GPUS)"
	@echo "  SHM:    $(SHM_SIZE)"
	@echo "  Net:    $(DOCKER_NETWORK)"
	@if [ -n "$(CURATOR_TMP)" ]; then echo "  Tmp:    $(CURATOR_TMP)"; fi
	@if [ -n "$(DATA_DIR)" ]; then echo "  Data:   $(DATA_DIR)"; fi
	@if [ ! -x "$(FFMPEG_DIR)/bin/ffmpeg" ]; then \
		echo "ERROR: $(FFMPEG_DIR)/bin/ffmpeg not found."; \
		echo "       Run 'make check-setup' for the sidecar installation hint."; \
		exit 1; \
	fi
	@COSMOS_CFG_MOUNT=$$(uv run python -m adapters.cosmos_curator.openai_config_env prepare \
		--output-dir "$(COSMOS_CURATOR_CONFIG_GEN_DIR)" \
		--merge-from "$(COSMOS_CURATOR_CONFIG_DIR)" \
		--print-docker-mount); \
	if [ -z "$$COSMOS_CFG_MOUNT" ] && [ -d "$(COSMOS_CURATOR_CONFIG_DIR)" ]; then \
		COSMOS_CFG_MOUNT="-v $(COSMOS_CURATOR_CONFIG_DIR):/cosmos_curator/config:ro"; \
	fi; \
	if [ -n "$$COSMOS_CFG_MOUNT" ]; then echo "  Curator cfg: env/host → /cosmos_curator/config"; fi; \
	echo ""; \
	docker run --gpus '$(GPUS)' --rm \
		$(MODELS_VOLUME_ARGS) \
		-v "$(abspath $(IMAGE_CONFIG_FILE)):/config/pipeline_config.yaml:ro" \
		$$COSMOS_CFG_MOUNT \
		$(FFMPEG_MOUNT) \
		$(if $(DATA_DIR),-v "$(DATA_DIR):$(DATA_DIR)") \
		$(HF_ENV_ARGS) \
		$(CURATOR_TMP_ARGS) \
		$(DOCKER_NETWORK_ARGS) \
		--shm-size=$(SHM_SIZE) \
		$(EXTRA_DOCKER_ARGS) \
		$(IMAGE_NAME):$(IMAGE_TAG) \
		bash -c "cd /opt/cosmos-curator && pixi run -e $(PIXI_ENV) --as-is python -m cosmos_curator.pipelines.image.run_pipeline /config/pipeline_config.yaml"

shell: check-image check-curator-runtime
	@echo "Starting interactive shell in cosmos-curator container..."
	@echo "  Image:  $(IMAGE_NAME):$(IMAGE_TAG)"
	@echo "  Config: $(CONFIG_FILE)"
	@echo "  FFmpeg: $(FFMPEG_DIR)"
	@echo "  Net:    $(DOCKER_NETWORK)"
	@if [ ! -x "$(FFMPEG_DIR)/bin/ffmpeg" ]; then \
		echo "ERROR: $(FFMPEG_DIR)/bin/ffmpeg not found."; \
		echo "       Run 'make check-setup' for the sidecar installation hint."; \
		exit 1; \
	fi
	@COSMOS_CFG_MOUNT=$$(uv run python -m adapters.cosmos_curator.openai_config_env prepare \
		--output-dir "$(COSMOS_CURATOR_CONFIG_GEN_DIR)" \
		--merge-from "$(COSMOS_CURATOR_CONFIG_DIR)" \
		--print-docker-mount); \
	if [ -z "$$COSMOS_CFG_MOUNT" ] && [ -d "$(COSMOS_CURATOR_CONFIG_DIR)" ]; then \
		COSMOS_CFG_MOUNT="-v $(COSMOS_CURATOR_CONFIG_DIR):/cosmos_curator/config:ro"; \
	fi; \
	echo ""; \
	docker run --gpus '$(GPUS)' --rm -it \
		$(MODELS_VOLUME_ARGS) \
		-v "$(abspath $(CONFIG_FILE)):/config/pipeline_config.yaml:ro" \
		$$COSMOS_CFG_MOUNT \
		$(FFMPEG_MOUNT) \
		$(if $(DATA_DIR),-v "$(DATA_DIR):$(DATA_DIR)") \
		$(HF_ENV_ARGS) \
		$(CURATOR_TMP_ARGS) \
		$(DOCKER_NETWORK_ARGS) \
		--shm-size=$(SHM_SIZE) \
		$(EXTRA_DOCKER_ARGS) \
		$(IMAGE_NAME):$(IMAGE_TAG) \
		bash -c "cd /opt/cosmos-curator && exec bash"

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

download-models: check-image
	@echo "Downloading required models (~90GB+, 20-30 minutes)..."
	@echo "  Image:  $(IMAGE_NAME):$(IMAGE_TAG)"
	@echo "  Target: $(MODELS_DIR)"
	@echo "  Models: $(MODEL_LIST)"
	@echo ""
	@mkdir -p "$(MODELS_DIR)"
	@mkdir -p "$(HOME)/.config/cosmos_curator"
	@if [ -f "$(HOME)/.config/cosmos_curator/hf_token.txt" ] && [ ! -f "$(HOME)/.config/cosmos_curator/config.yaml" ]; then \
		echo "Creating config.yaml from hf_token.txt..."; \
		echo "huggingface:" > "$(HOME)/.config/cosmos_curator/config.yaml"; \
		echo "    api_key: \"$$(cat $(HOME)/.config/cosmos_curator/hf_token.txt)\"" >> "$(HOME)/.config/cosmos_curator/config.yaml"; \
	fi
	@if [ -f "$(HOME)/.config/cosmos_curator/config.yaml" ] && [ ! -f "$(HOME)/.config/cosmos_curator/cosmos_curator.yaml" ]; then \
		cp "$(HOME)/.config/cosmos_curator/config.yaml" "$(HOME)/.config/cosmos_curator/cosmos_curator.yaml"; \
	fi
	@if [ -f "$(HOME)/.config/cosmos_curator/cosmos_curator.yaml" ] && [ ! -f "$(HOME)/.config/cosmos_curator/config.yaml" ]; then \
		cp "$(HOME)/.config/cosmos_curator/cosmos_curator.yaml" "$(HOME)/.config/cosmos_curator/config.yaml"; \
	fi
	@echo "OK: Models directory: $(MODELS_DIR)"
	@echo "OK: HuggingFace config: $(HOME)/.config/cosmos_curator"
	@echo ""
	@if [ ! -f "$(HOME)/.config/cosmos_curator/config.yaml" ] && [ ! -f "$(HOME)/.config/cosmos_curator/cosmos_curator.yaml" ] && [ ! -f "$(HOME)/.config/cosmos_curator/hf_token.txt" ] && [ -z "$$HF_TOKEN" ]; then \
		echo "WARNING: HuggingFace token not found"; \
		echo "  Models require authentication. Set HF_TOKEN or create:"; \
		echo "    echo 'your-token' > $(HOME)/.config/cosmos_curator/hf_token.txt"; \
		echo ""; \
		read -p "Continue anyway? [y/N] " confirm && \
		if [ "$$confirm" != "y" ] && [ "$$confirm" != "Y" ]; then \
			echo "Cancelled. Set up HuggingFace authentication first."; \
			exit 1; \
		fi; \
	fi
	@echo "Downloading models using Docker container..."
	docker run --gpus all --rm \
		-v "$(HOME)/.config/cosmos_curator:/cosmos_curator/config:ro" \
		-v "$(MODELS_DIR):/config/models" \
		-e HF_HOME=/config/models \
		--network=host \
		$(IMAGE_NAME):$(IMAGE_TAG) \
		bash -c "cd /opt/cosmos-curator && pixi run -e model-download --as-is python -m cosmos_curator.core.managers.model_cli download \
		--models $(MODEL_LIST)"
	@echo ""
	@echo "Models downloaded to: $(MODELS_DIR)"
	@echo "Verify: ls -lh $(MODELS_DIR)"
	@echo ""
	@echo "NOTE: SeedVR2 is not in MODEL_LIST. For super_resolution cookbooks run:"
	@echo "  make download-seedvr2 MODELS_DIR=$(MODELS_DIR) SEEDVR_VARIANT=$(SEEDVR_VARIANT)"
	@echo "  # or: make download-all-models MODELS_DIR=$(MODELS_DIR)"

# SeedVR2 HF checkpoints (layout shared with pseudo-labeling: MODELS_DIR/seedvr2/).
# Required for Curator cookbooks with super_resolution: true.
# download-seedvr2 is the preferred operator name; download-seedvr-ckpts is an alias.
download-seedvr2 download-seedvr-ckpts:
	@echo "Downloading SeedVR2 checkpoints into $(MODELS_DIR)/seedvr2 ..."
	@echo "  Variant: $(SEEDVR_VARIANT)"
	@echo "  Sources: ByteDance-Seed/SeedVR2-3B and/or SeedVR2-7B"
	@mkdir -p "$(MODELS_DIR)"
	uv run python -m adapters.cosmos_curator.seedvr_ckpts ensure \
		--models-dir "$(MODELS_DIR)" \
		--variant "$(SEEDVR_VARIANT)"
	@echo "OK: $$(ls -lh "$(MODELS_DIR)/seedvr2" 2>/dev/null || true)"

# Curator MODEL_LIST weights, then SeedVR2 ckpts (SQA-03 / SR path).
download-all-models: download-models download-seedvr2
	@echo "OK: Curator models + SeedVR2 ($(SEEDVR_VARIANT)) under $(MODELS_DIR)"

# ---------------------------------------------------------------------------
# Development
# ---------------------------------------------------------------------------

format:
	@echo "Formatting code..."
	@ruff format .
	@echo "Done."

clean:
	@echo "Cleaning up..."
	@echo ""
	@echo "This will NOT delete:"
	@echo "  - Docker images (use: docker rmi $(IMAGE_NAME):$(IMAGE_TAG))"
	@echo "  - Downloaded models (in $(MODELS_DIR))"
	@echo ""
	@read -p "Continue? [y/N] " confirm && \
	if [ "$$confirm" = "y" ] || [ "$$confirm" = "Y" ]; then \
		rm -rf .pytest_cache htmlcov .coverage; \
		echo "Cleaned."; \
	else \
		echo "Cancelled."; \
	fi
