# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Drift guards for configs/split.yaml against Curator contracts.

Always-on checks catch YAML/type regressions without Docker.
Optional Docker checks call upstream load_pipeline_config when a Curator
image is present locally (skip otherwise so CI stays offline-friendly).
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SPLIT_YAML = REPO_ROOT / "configs" / "split.yaml"

# Stage / feature toggles that must remain present in the reference catalog.
REQUIRED_KEYS = frozenset(
    {
        "pipeline",
        "input_video_path",
        "output_clip_path",
        "model_weights_path",
        "splitting_algorithm",
        "fixed_stride_split_duration",
        "transnetv2_threshold",
        "super_resolution",
        "sr_variant",
        "motion_filter",
        "aesthetic_threshold",
        "artificial_text_filter",
        "vlm_filter",
        "video_classifier",
        "generate_embeddings",
        "embedding_algorithm",
        "generate_captions",
        "captioning_algorithm",
        "enhance_captions",
        "sam3",
        "sam3_region",
        "sam3_output_format",
        "event_captioning",
        "upload_clips",
        "write_all_caption_json",
        "generate_cosmos_predict_dataset",
        "multi_cam",
        "stage_save",
        "stage_replay",
    }
)

# Keys that must never appear (silent no-ops / historical bugs).
FORBIDDEN_KEYS = frozenset(
    {
        "enable_sam3",
        "enable_event_captioning",
        "enable_super_resolution",
    }
)

BOOL_KEYS = frozenset(
    {
        "sam3",
        "event_captioning",
        "super_resolution",
        "video_classifier",
        "generate_embeddings",
        "generate_captions",
        "enhance_captions",
        "upload_clips",
        "write_all_caption_json",
        "multi_cam",
        "dry_run",
        "verbose",
        "perf_profile",
    }
)

STRING_ENUM_KEYS = {
    "motion_filter": frozenset({"disable", "enable", "score-only"}),
    "vlm_filter": frozenset({"disable", "enable", "score-only"}),
    "artificial_text_filter": frozenset({"disable", "enable"}),
    "splitting_algorithm": frozenset({"fixed-stride", "transnetv2"}),
    "sam3_region": frozenset({"box", "contour"}),
    "sam3_output_format": frozenset({"native", "coco", "mot"}),
    "sam3_annotated_video_label_style": frozenset({"id", "name", "none"}),
}


def _default_curator_image() -> str:
    from adapters.docker_runtime import default_curator_image

    return default_curator_image()


def _docker_image_present(image: str) -> bool:
    if shutil.which("docker") is None:
        return False
    result = subprocess.run(
        ["docker", "image", "inspect", image],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def _load_split_config() -> dict:
    raw = yaml.safe_load(SPLIT_YAML.read_text(encoding="utf-8"))
    assert isinstance(raw, dict)
    return raw


class TestSplitConfigReferenceContract:
    """Host-side contracts — always run in unit CI."""

    def test_split_yaml_exists_and_parses(self) -> None:
        assert SPLIT_YAML.is_file()
        cfg = _load_split_config()
        assert cfg["pipeline"] == "split"

    def test_required_stage_keys_present(self) -> None:
        cfg = _load_split_config()
        missing = sorted(REQUIRED_KEYS - set(cfg))
        assert missing == [], f"configs/split.yaml missing keys: {missing}"

    def test_forbidden_legacy_keys_absent(self) -> None:
        cfg = _load_split_config()
        present = sorted(FORBIDDEN_KEYS & set(cfg))
        assert present == [], f"forbidden keys present: {present}"

    def test_boolean_toggles_are_bool(self) -> None:
        cfg = _load_split_config()
        for key in BOOL_KEYS:
            assert key in cfg, key
            assert isinstance(cfg[key], bool), f"{key} must be bool, got {cfg[key]!r}"

    def test_string_enum_toggles(self) -> None:
        cfg = _load_split_config()
        for key, allowed in STRING_ENUM_KEYS.items():
            assert cfg[key] in allowed, f"{key}={cfg[key]!r} not in {sorted(allowed)}"

    def test_video_classifier_not_truthy_string(self) -> None:
        """Regression: string 'disable' is truthy in Python and would enable classifier."""
        cfg = _load_split_config()
        assert cfg["video_classifier"] is False or cfg["video_classifier"] is True
        assert not isinstance(cfg["video_classifier"], str)

    def test_generate_cosmos_predict_dataset_enum(self) -> None:
        cfg = _load_split_config()
        value = cfg["generate_cosmos_predict_dataset"]
        assert value in {"disable", "predict2", False, True}, value

    def test_embedding_and_caption_algorithms_documented(self) -> None:
        cfg = _load_split_config()
        assert cfg["embedding_algorithm"] in {
            "internvideo2",
            "cosmos-embed1-224p",
            "cosmos-embed1-336p",
            "cosmos-embed1-448p",
            "openai",
        }
        assert isinstance(cfg["captioning_algorithm"], str)
        assert cfg["captioning_algorithm"]

    def test_sam3_region_and_output_format_defaults(self) -> None:
        """Curator 2.3.0 overlay/export defaults must stay explicit in the catalog."""
        cfg = _load_split_config()
        assert cfg["sam3_region"] == "contour"
        assert cfg["sam3_output_format"] == "native"
        assert cfg["sam3_annotated_video_label_style"] == "id"
        assert cfg["sam3_det_nms_thresh"] is None

    def test_reference_key_count_floor(self) -> None:
        """Guard against accidental truncation of the reference catalog."""
        cfg = _load_split_config()
        assert len(cfg) >= 200, f"expected rich reference config, got {len(cfg)} keys"

    def test_placeholders_present_for_io(self) -> None:
        cfg = _load_split_config()
        assert "<YOUR_" in str(cfg["input_video_path"])
        assert "<YOUR_" in str(cfg["output_clip_path"])


@pytest.mark.skipif(
    not _docker_image_present(_default_curator_image()),
    reason=f"Curator image {_default_curator_image()} not present locally",
)
class TestSplitConfigLoadsInCuratorImage:
    """Upstream load_pipeline_config drift check (requires local Curator image)."""

    def test_load_pipeline_config_accepts_split_yaml(self, tmp_path: Path) -> None:
        cfg = _load_split_config()
        # Replace placeholders so loader path checks do not fail on angle-brackets.
        cfg["input_video_path"] = "/tmp/sqa_split_input"
        cfg["output_clip_path"] = "/tmp/sqa_split_output"
        cfg["model_weights_path"] = "/config/models"
        runnable = tmp_path / "split_runnable.yaml"
        runnable.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

        image = _default_curator_image()
        script = (
            "from cosmos_curator.core.utils.config.pipeline_config_loader "
            "import load_pipeline_config\n"
            "cfg = load_pipeline_config('/config/split_runnable.yaml')\n"
            "assert cfg.get('_pipeline') == 'split', cfg.get('_pipeline')\n"
            "print('LOAD_OK', cfg.get('_pipeline'), "
            "cfg.get('splitting_algorithm'), cfg.get('video_classifier'), "
            "type(cfg.get('video_classifier')).__name__)\n"
        )
        result = subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "-v",
                f"{runnable}:/config/split_runnable.yaml:ro",
                "--entrypoint",
                "bash",
                image,
                "-c",
                "cd /opt/cosmos-curator && pixi run -e default --as-is python - <<'PY'\n"
                f"{script}"
                "PY",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=180,
        )
        assert result.returncode == 0, (
            f"load_pipeline_config failed\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
        assert "LOAD_OK" in result.stdout
        assert "bool" in result.stdout
