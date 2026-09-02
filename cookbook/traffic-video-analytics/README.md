# Traffic video analytics

Curator recipes for the clips in `videos/`. Source: NGC
`nvidia/vss-developer/dev-profile-sample-data:3.1.0`. These are synthetic
clips for trying the recipe; the classifier taxonomy in `split.yaml` still
applies (nominal flow vs conflict / aftermath / obstruction).

## Video samples

Keep docs **outside** `videos/`. Curator treats every file in that directory
as input media.

| File | Duration | Format | Source clip |
|------|----------|--------|-------------|
| `videos/SDG-Intersection-01.mp4` | ~89 s | H.264 1920×1080 30 fps | `sample-sim-jaywalking.mp4`; Path A (`limit: 1`) |
| `videos/SDG-Intersection-02.mp4` | ~130 s | H.264 1920×1080 30 fps | `sample-sim-traffic.mp4` |

`split.yaml` and `split-minimal.yaml` both read `videos/` (`/data/videos`).
`split-minimal.yaml` sets `limit: 1`, so it processes `SDG-Intersection-01.mp4`
first.

## Run

```bash
DATA=$PWD/cookbook/traffic-video-analytics
MODELS=/path/to/models

# Split + Cosmos-Embed1 only (limit: 1 → SDG-Intersection-01.mp4)
make run-pipeline \
  CONFIG_FILE=cookbook/traffic-video-analytics/split-minimal.yaml \
  DATA_DIR=$DATA \
  MODELS_DIR=$MODELS

# Captions, classifier, SAM3, event captions, Cosmos-Embed1
make run-pipeline \
  CONFIG_FILE=cookbook/traffic-video-analytics/split.yaml \
  DATA_DIR=$DATA \
  MODELS_DIR=$MODELS \
  PIXI_ENV=sam3

make run-pipeline CONFIG_FILE=cookbook/traffic-video-analytics/dedup.yaml DATA_DIR=$DATA MODELS_DIR=$MODELS
make run-pipeline CONFIG_FILE=cookbook/traffic-video-analytics/shard.yaml DATA_DIR=$DATA MODELS_DIR=$MODELS
```

YAML I/O is `/data/...`. Pass `DATA_DIR` as this directory. Full `split.yaml`
needs VLM/LLM endpoints or in-process 30B weights — see
[VLM and LLM Endpoints](../../docs/user-guide/vlm-llm-endpoints.md).

To use other MP4s, replace files under `videos/` (H.264). Do not put a README
in `videos/`; Curator reads every file there. Delete `output/split*` before a
clean re-run.
