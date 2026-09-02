# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for Cosmos Curator configuration preflight validation."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import click
import pytest
import yaml
from click.testing import CliRunner

from apps.cli.__main__ import main
from apps.cli.pipeline_config import validate_curator_pipeline_config
from apps.workflows import RunCuratorPipelineRequest, run_curator_pipeline

_WAREHOUSE_SAM3_MASS_NOUNS = frozenset({"caution tape"})


def _warehouse_sam3_prompt(obj: str) -> str:
    """Map a classification_events object name to a SAM3 text prompt."""
    if obj in _WAREHOUSE_SAM3_MASS_NOUNS:
        return obj
    article = "an" if obj[0] in "aeiou" else "a"
    return f"{article} {obj}"


def test_traffic_cookbook_event_taxonomies_stay_aligned() -> None:
    """Keep executable and reference traffic category lists in lock-step."""
    cookbook = Path(__file__).resolve().parents[2] / "cookbook" / "traffic-video-analytics"
    split = yaml.safe_load((cookbook / "split.yaml").read_text(encoding="utf-8"))
    manifest = json.loads((cookbook / "input_config.json").read_text(encoding="utf-8"))
    events = yaml.safe_load((cookbook / "classification_events.yaml").read_text(encoding="utf-8"))

    expected = split["video_classifier_allow"]
    assert expected
    assert manifest["cosmos_curator"]["overrides"]["video_classifier_allow"] == expected
    assert events["events_list"] == expected
    assert "nominal_traffic_flow" in expected
    assert "jaywalking" in expected
    assert "normal_traffic_flow" not in expected
    assert split["sam3_prompts"] == manifest["cosmos_curator"]["overrides"]["sam3_prompts"]
    assert "a police car" in split["sam3_prompts"]
    assert "an ambulance" in split["sam3_prompts"]
    assert "a fire truck" in split["sam3_prompts"]
    assert "a taxi" in split["sam3_prompts"]
    assert "a crosswalk" in split["sam3_prompts"]
    assert "a stop sign" in split["sam3_prompts"]
    assert split["sam3_region"] == "contour"
    assert split["sam3_output_format"] == "native"
    assert split["sam3_det_nms_thresh"] == 0.1
    assert split["sam3_annotated_video_label_style"] == "id"
    assert split["sam3_annotated_video_mask_opacity"] == 0
    overrides = manifest["cosmos_curator"]["overrides"]
    assert overrides["sam3_region"] == split["sam3_region"]
    assert overrides["sam3_output_format"] == split["sam3_output_format"]
    assert overrides["sam3_det_nms_thresh"] == split["sam3_det_nms_thresh"]
    assert split["embedding_algorithm"] == "cosmos-embed1-224p"
    assert split["generate_embeddings"] is True
    assert split["fixed_stride_split_duration"] == 10
    assert (
        manifest["cosmos_curator"]["overrides"]["fixed_stride_split_duration"]
        == split["fixed_stride_split_duration"]
    )
    stats = manifest["video_stats"]
    assert stats["resolution"] == "1920x1080"
    assert stats["fps"] == 30
    assert "352x288" not in json.dumps(stats)
    assert "10-15" not in json.dumps(stats)


def test_traffic_split_minimal_is_first_run_not_kitchen_sink() -> None:
    """Path A first-run YAML must skip VLM/SAM3 and still emit Cosmos-Embed1 embeddings."""
    cookbook = Path(__file__).resolve().parents[2] / "cookbook" / "traffic-video-analytics"
    config_path = cookbook / "split-minimal.yaml"
    split = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert split["pipeline"] == "split"
    assert split["generate_captions"] is False
    assert split["enhance_captions"] is False
    assert split["video_classifier"] is False
    assert split["sam3"] is False
    assert split["event_captioning"] is False
    assert split["super_resolution"] is False
    assert split["artificial_text_filter"] is False
    assert split["generate_embeddings"] is True
    assert split["embedding_algorithm"] == "cosmos-embed1-224p"
    assert split["input_video_path"] == "/data/videos"
    assert split["limit"] == 1
    assert split["output_clip_path"] == "/data/output/split-minimal"
    assert "video_classifier_allow" not in split
    assert "captioning_algorithm" not in split
    validate_curator_pipeline_config(config_path)


def test_traffic_cookbook_ships_samples_and_data_dir_io() -> None:
    """Traffic cookbook documents sample clips and uses /data I/O."""
    cookbook = Path(__file__).resolve().parents[2] / "cookbook" / "traffic-video-analytics"
    videos_dir = cookbook / "videos"
    if videos_dir.is_dir():
        assert all(p.suffix == ".mp4" for p in videos_dir.iterdir() if p.is_file())
    readme = (cookbook / "README.md").read_text(encoding="utf-8")
    assert "SDG-Intersection-01.mp4" in readme
    assert "SDG-Intersection-02.mp4" in readme
    assert not (cookbook / "videos-smoke").exists()
    assert (cookbook / "event_caption_prompt.txt").is_file()

    split = yaml.safe_load((cookbook / "split.yaml").read_text(encoding="utf-8"))
    dedup = yaml.safe_load((cookbook / "dedup.yaml").read_text(encoding="utf-8"))
    shard = yaml.safe_load((cookbook / "shard.yaml").read_text(encoding="utf-8"))
    assert split["input_video_path"] == "/data/videos"
    assert split["output_clip_path"] == "/data/output/split"
    assert split["event_caption_prompt_file"] == "/data/event_caption_prompt.txt"
    assert dedup["input_embeddings_path"] == "/data/output/split"
    assert dedup["output_path"] == "/data/output/dedup"
    assert shard["input_clip_path"] == "/data/output/split"
    assert shard["input_semantic_dedup_path"] == "/data/output/dedup"
    assert shard["output_dataset_path"] == "/data/output/shard"
    assert split["input_video_path"].startswith("/data/")


def test_warehouse_sam3_prompt_articles() -> None:
    """SAM3 prompts use a/an except for mass nouns in the warehouse object list."""
    assert _warehouse_sam3_prompt("worker") == "a worker"
    assert _warehouse_sam3_prompt("emergency exit sign") == "an emergency exit sign"
    assert _warehouse_sam3_prompt("caution tape") == "caution tape"
    assert _warehouse_sam3_prompt("box") == "a box"
    with pytest.raises(IndexError):
        _warehouse_sam3_prompt("")


def test_warehouse_cookbook_event_taxonomies_stay_aligned() -> None:
    """Keep executable and reference warehouse category lists in lock-step."""
    cookbook = Path(__file__).resolve().parents[2] / "cookbook" / "warehouse-safety"
    split = yaml.safe_load((cookbook / "split.yaml").read_text(encoding="utf-8"))
    manifest = json.loads((cookbook / "input_config.json").read_text(encoding="utf-8"))
    events = yaml.safe_load((cookbook / "classification_events.yaml").read_text(encoding="utf-8"))

    expected = split["video_classifier_allow"]
    assert expected
    assert manifest["cosmos_curator"]["overrides"]["video_classifier_allow"] == expected
    assert events["events_list"] == expected
    assert "unauthorized_restricted_area_entry" in expected
    assert "forklift_collision_with_vehicle" in expected
    assert "forklift_traveling_with_raised_load" in expected
    assert "spill_or_obstruction_in_aisle" in expected
    assert "normal_warehouse_operations" in expected
    prompt = (cookbook / "prompt.md").read_text(encoding="utf-8")
    valid_line = next(
        line for line in prompt.splitlines() if line.strip().startswith("Valid labels:")
    )
    for label in expected:
        assert label in valid_line
        assert f"**{label}**" in prompt
    for extra in ("caution tape", "traffic cone", "rolling ladder", "rack"):
        assert extra in events["objects_of_interest"]
    assert "floor markings" not in events["objects_of_interest"]
    assert "floor markings" not in split["sam3_prompts"]
    assert "forklift_collision_with_rack" in events["events_list"]
    assert "rack_collapse_with_falling_inventory" in events["events_list"]
    assert split["sam3"] is True
    assert split["event_captioning"] is True
    assert split["sam3_prompts"] == manifest["cosmos_curator"]["overrides"]["sam3_prompts"]
    assert split["sam3_prompts"] == [
        _warehouse_sam3_prompt(obj) for obj in events["objects_of_interest"]
    ]
    assert split["sam3_region"] == "box"
    assert split["sam3_output_format"] == "native"
    assert split["sam3_det_nms_thresh"] == 0.1
    assert split["sam3_annotated_video_mask_opacity"] == 0
    assert split["sam3_annotated_video_label_style"] == "id"
    assert split["sam3_write_annotated_video"] is True
    assert split["sam3_annotated_video_trails"] is False
    overrides = manifest["cosmos_curator"]["overrides"]
    assert overrides["sam3_region"] == "box"
    assert overrides["sam3_output_format"] == "native"
    assert overrides["sam3_det_nms_thresh"] == 0.1
    assert "a cardboard box" not in split["sam3_prompts"]
    assert "a box" in split["sam3_prompts"]
    assert split["event_caption_prompt_file"] == "/data/event_caption_prompt.txt"
    assert split["embedding_algorithm"] == "internvideo2"
    event_prompt = (cookbook / "event_caption_prompt.txt").read_text(encoding="utf-8")
    for label in expected:
        assert label in event_prompt
    validate_curator_pipeline_config(cookbook / "split.yaml")


def test_warehouse_split_minimal_is_first_run_not_kitchen_sink() -> None:
    """Warehouse Path A YAML must skip VLM/SAM3 and still emit IV2 embeddings."""
    cookbook = Path(__file__).resolve().parents[2] / "cookbook" / "warehouse-safety"
    config_path = cookbook / "split-minimal.yaml"
    split = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert split["pipeline"] == "split"
    assert split["generate_captions"] is False
    assert split["enhance_captions"] is False
    assert split["video_classifier"] is False
    assert split["sam3"] is False
    assert split["event_captioning"] is False
    assert split["super_resolution"] is False
    assert split["artificial_text_filter"] is False
    assert split["generate_embeddings"] is True
    assert split["embedding_algorithm"] == "internvideo2"
    assert split["input_video_path"] == "/data/videos"
    assert split["limit"] == 1
    assert split["output_clip_path"] == "/data/output/split-minimal"
    validate_curator_pipeline_config(config_path)


def test_warehouse_cookbook_ships_samples_and_data_dir_io() -> None:
    """Warehouse cookbook documents sample clips and uses /data I/O."""
    cookbook = Path(__file__).resolve().parents[2] / "cookbook" / "warehouse-safety"
    videos_dir = cookbook / "videos"
    if videos_dir.is_dir():
        assert all(p.suffix == ".mp4" for p in videos_dir.iterdir() if p.is_file())
    readme = (cookbook / "README.md").read_text(encoding="utf-8")
    assert "SDG-Warehouse-04.mp4" in readme
    assert "warehouse_sample.mp4" in readme
    assert not (cookbook / "videos-smoke").exists()
    assert (cookbook / "event_caption_prompt.txt").is_file()

    split = yaml.safe_load((cookbook / "split.yaml").read_text(encoding="utf-8"))
    dedup = yaml.safe_load((cookbook / "dedup.yaml").read_text(encoding="utf-8"))
    shard = yaml.safe_load((cookbook / "shard.yaml").read_text(encoding="utf-8"))
    assert split["input_video_path"] == "/data/videos"
    assert split["output_clip_path"] == "/data/output/split"
    assert split["event_caption_prompt_file"] == "/data/event_caption_prompt.txt"
    assert dedup["input_embeddings_path"] == "/data/output/split"
    assert dedup["output_path"] == "/data/output/dedup"
    assert shard["input_clip_path"] == "/data/output/split"
    assert shard["input_semantic_dedup_path"] == "/data/output/dedup"
    assert shard["output_dataset_path"] == "/data/output/shard"


@pytest.mark.parametrize(
    ("placement", "deprecated_key", "canonical_key"),
    [
        ("top_level", "enable_sam3", "sam3"),
        ("top_level", "enable_event_captioning", "event_captioning"),
        ("args", "enable_sam3", "sam3"),
        ("args", "enable_event_captioning", "event_captioning"),
    ],
)
def test_rejects_deprecated_keys_in_supported_placements(
    tmp_path: Path,
    placement: str,
    deprecated_key: str,
    canonical_key: str,
) -> None:
    config = tmp_path / "split.yaml"
    key_line = f"{deprecated_key}: true\n"
    contents = f"pipeline: split\n{key_line}"
    if placement == "args":
        contents = f"pipeline: split\nargs:\n  {key_line}"
    config.write_text(contents, encoding="utf-8")

    with pytest.raises(click.ClickException) as error:
        validate_curator_pipeline_config(config)

    assert deprecated_key in error.value.message
    assert canonical_key in error.value.message


@pytest.mark.parametrize(
    "contents",
    [
        "pipeline: split\nsam3: true\nevent_captioning: true\n",
        "pipeline: split\nargs:\n  sam3: true\n  event_captioning: true\n",
    ],
)
def test_accepts_canonical_keys_in_supported_placements(
    tmp_path: Path,
    contents: str,
) -> None:
    config = tmp_path / "split.yaml"
    config.write_text(contents, encoding="utf-8")

    validate_curator_pipeline_config(config)


@pytest.mark.parametrize("args_value", ["null", "false", "split", "[]"])
def test_non_mapping_args_follow_curator_top_level_fallback(
    tmp_path: Path,
    args_value: str,
) -> None:
    config = tmp_path / "split.yaml"
    config.write_text(
        f"pipeline: split\nargs: {args_value}\nsam3: true\nevent_captioning: true\n",
        encoding="utf-8",
    )

    validate_curator_pipeline_config(config)


def test_does_not_scan_metadata_overrides(tmp_path: Path) -> None:
    config = tmp_path / "metadata.yaml"
    config.write_text(
        "cosmos_curator:\n  overrides:\n    enable_sam3: true\n    enable_event_captioning: true\n",
        encoding="utf-8",
    )

    validate_curator_pipeline_config(config)


@pytest.mark.parametrize("contents", ["pipeline: [unterminated", "- split\n- sam3"])
def test_rejects_malformed_configuration(tmp_path: Path, contents: str) -> None:
    config = tmp_path / "split.yaml"
    config.write_text(contents, encoding="utf-8")

    with pytest.raises(click.ClickException):
        validate_curator_pipeline_config(config)


def test_reports_unreadable_configuration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tmp_path / "split.yaml"
    config.write_text("pipeline: split\n", encoding="utf-8")
    monkeypatch.setattr(Path, "read_text", MagicMock(side_effect=PermissionError))

    with pytest.raises(click.ClickException, match="could not be read"):
        validate_curator_pipeline_config(config)


def test_cli_validates_before_runner_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tmp_path / "split.yaml"
    config.write_text(
        "pipeline: split\nargs:\n  enable_sam3: true\n",
        encoding="utf-8",
    )
    runner_factory = MagicMock()
    monkeypatch.setattr("apps.cli.__main__.build_curator_runner", runner_factory)

    result = CliRunner().invoke(main, ["curator-run", "--config-file", str(config)])

    assert result.exit_code == 1
    assert "enable_sam3" in result.output
    assert "sam3" in result.output
    runner_factory.assert_not_called()


def test_cli_preserves_canonical_config_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tmp_path / "split.yaml"
    config.write_text(
        "pipeline: split\nsam3: true\nevent_captioning: true\n",
        encoding="utf-8",
    )
    runner = MagicMock()
    runner.run_pipeline.return_value = SimpleNamespace(command=["docker", "run"])
    monkeypatch.setattr("apps.cli.__main__.build_curator_runner", lambda **_: runner)

    result = CliRunner().invoke(
        main,
        ["curator-run", "--config-file", str(config), "--dry-run"],
    )

    assert result.exit_code == 0
    assert json.loads(result.output) == {"dry_run": True, "command": ["docker", "run"]}
    runner.run_pipeline.assert_called_once()


def test_cli_help_remains_click_text() -> None:
    result = CliRunner().invoke(main, ["curator-run", "--help"])

    assert result.exit_code == 0
    assert result.output.startswith("Usage:")


def test_workflow_validates_before_runner_construction() -> None:
    runner_factory = MagicMock()

    def reject_config(config_file: str | Path) -> None:
        raise click.ClickException(f"invalid config: {config_file}")

    with pytest.raises(click.ClickException, match="invalid config"):
        run_curator_pipeline(
            RunCuratorPipelineRequest(config_file="bad.yaml"),
            runner_factory=runner_factory,
            validate_config=reject_config,
        )

    runner_factory.assert_not_called()


def test_workflow_passes_request_to_runner() -> None:
    runner = MagicMock()
    runner.run_pipeline.return_value = SimpleNamespace(command=("docker", "run"))
    validate_config = MagicMock()

    result = run_curator_pipeline(
        RunCuratorPipelineRequest(
            config_file="split.yaml",
            data_dir="data",
            models_dir="models",
            ffmpeg_dir="ffmpeg",
            dry_run=True,
        ),
        runner_factory=lambda: runner,
        validate_config=validate_config,
    )

    validate_config.assert_called_once_with("split.yaml")
    runner.run_pipeline.assert_called_once_with(
        "split.yaml",
        data_dir="data",
        models_dir="models",
        ffmpeg_dir="ffmpeg",
        dry_run=True,
    )
    assert result.dry_run is True
    assert result.command == ["docker", "run"]
