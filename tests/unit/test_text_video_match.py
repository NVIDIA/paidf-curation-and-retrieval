# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for text↔video matching analytics."""

from __future__ import annotations

import numpy as np
import pytest
from click import ClickException

from apps.cli.analytics_cmds import _copy_selected, run_divknn_select_cli
from packages.analytics.text_video_match import (
    match_text_to_videos,
    mean_std_threshold,
    score_gallery_against_texts,
    select_by_threshold,
    select_top_k,
)


def _unit(v):
    a = np.asarray(v, dtype=np.float64)
    return a / np.linalg.norm(a)


class TestScoreGalleryAgainstTexts:
    def test_max_reduce_picks_best_query(self):
        gallery = np.stack([_unit([1, 0, 0]), _unit([0, 1, 0]), _unit([0, 0, 1])])
        t1 = _unit([1, 0, 0])
        t2 = _unit([0, 1, 0])
        scores, per = score_gallery_against_texts(gallery, [t1, t2], reduce="max")
        assert scores.shape == (3,)
        assert scores[0] == pytest.approx(1.0, abs=1e-6)
        assert scores[1] == pytest.approx(1.0, abs=1e-6)
        assert scores[2] < 0.1
        assert "q1" in per and "q2" in per

    def test_empty_texts_raises(self):
        with pytest.raises(ValueError, match="At least one"):
            score_gallery_against_texts(np.eye(2), [], reduce="max")

    def test_mean_reduce(self):
        gallery = np.stack([_unit([1, 0])])
        t1 = _unit([1, 0])
        t2 = _unit([-1, 0])
        scores, _ = score_gallery_against_texts(gallery, [t1, t2], reduce="mean")
        assert scores[0] == pytest.approx(0.0, abs=1e-6)


class TestThresholdAndTopK:
    def test_mean_std_threshold(self):
        scores = np.array([0.0, 0.0, 0.0, 1.0])
        thr = mean_std_threshold(scores, k_std=1.0)
        assert thr > scores.mean()

    def test_select_by_threshold_sorted(self):
        names = ["a", "b", "c"]
        scores = np.array([0.2, 0.9, 0.5])
        idx = select_by_threshold(names, scores, threshold=0.4)
        assert [names[i] for i in idx] == ["b", "c"]

    def test_select_top_k(self):
        names = ["a", "b", "c", "d"]
        scores = np.array([0.1, 0.4, 0.3, 0.2])
        idx = select_top_k(names, scores, top_k=2)
        assert [names[i] for i in idx] == ["b", "c"]

    def test_top_k_invalid(self):
        with pytest.raises(ValueError):
            select_top_k(["a"], np.array([1.0]), top_k=0)


class TestMatchTextToVideos:
    def test_threshold_mode(self):
        names = ["v0", "v1", "v2"]
        gallery = np.stack([_unit([1, 0]), _unit([0.9, 0.1]), _unit([0, 1])])
        text = [_unit([1, 0])]
        result = match_text_to_videos(names, gallery, text, mode="threshold", k_std=0.5)
        assert result.threshold is not None
        assert "v0" in result.matched_file_names
        assert len(result.scores) == 3

    def test_top_k_mode(self):
        names = ["v0", "v1", "v2"]
        gallery = np.stack([_unit([1, 0]), _unit([0.5, 0.5]), _unit([0, 1])])
        text = [_unit([1, 0])]
        result = match_text_to_videos(names, gallery, text, mode="top_k", top_k=1)
        assert result.matched_file_names == ["v0"]
        assert result.threshold is None


class TestCopySelected:
    def test_copy_selected_keeps_recursive_basename_lookup(self, tmp_path):
        video_dir = tmp_path / "videos"
        nested = video_dir / "nested"
        nested.mkdir(parents=True)
        clip = nested / "safe.mp4"
        clip.write_bytes(b"clip")

        copied = _copy_selected(["safe.mp4"], video_dir, tmp_path / "selected_clips")

        assert copied == [
            {
                "rank": 1,
                "file_name": "safe.mp4",
                "source_path": str(clip),
                "dest_path": str(tmp_path / "selected_clips" / "safe.mp4"),
            }
        ]
        assert (tmp_path / "selected_clips" / "safe.mp4").read_bytes() == b"clip"

    def test_copy_selected_empty_input_creates_empty_output(self, tmp_path):
        output_dir = tmp_path / "selected_clips"

        assert _copy_selected([], tmp_path / "videos", output_dir) == []
        assert output_dir.is_dir()
        assert not any(output_dir.iterdir())

    def test_copy_selected_rejects_missing_later_clip_without_partial_output(self, tmp_path):
        video_dir = tmp_path / "videos"
        video_dir.mkdir()
        (video_dir / "present.mp4").write_bytes(b"clip")
        output_dir = tmp_path / "selected_clips"

        with pytest.raises(ClickException, match="Selected clip not found: missing.mp4"):
            _copy_selected(["present.mp4", "missing.mp4"], video_dir, output_dir)

        assert output_dir.is_dir()
        assert not any(output_dir.iterdir())

    def test_copy_selected_rejects_ambiguous_recursive_matches(self, tmp_path):
        video_dir = tmp_path / "videos"
        for directory in ("first", "second"):
            nested = video_dir / directory
            nested.mkdir(parents=True)
            (nested / "duplicate.mp4").write_bytes(directory.encode())

        with pytest.raises(ClickException, match="ambiguous.*matched 2"):
            _copy_selected(
                ["duplicate.mp4"],
                video_dir,
                tmp_path / "selected_clips",
            )

    @pytest.mark.parametrize(
        "name",
        [
            "/tmp/evil.mp4",
            "../evil.mp4",
            "nested/evil.mp4",
            r"nested\evil.mp4",
            r"C:\temp\evil.mp4",
        ],
    )
    def test_copy_selected_rejects_pathful_metadata_before_copy(self, tmp_path, name):
        video_dir = tmp_path / "videos"
        video_dir.mkdir()
        output_dir = tmp_path / "selected_clips"

        with pytest.raises(ClickException, match="Unsafe clip file_name"):
            _copy_selected([name], video_dir, output_dir)

        assert not any(output_dir.iterdir())


class TestDivKnnClipCopy:
    @staticmethod
    def _write_embeddings(path, names, embeddings):
        import pandas as pd

        pd.DataFrame({"file_name": names, "embedding": embeddings}).to_parquet(path, index=False)

    def test_divknn_selection_copies_every_selected_clip(self, tmp_path):
        target = tmp_path / "target.parquet"
        source = tmp_path / "source.parquet"
        self._write_embeddings(target, ["target.mp4"], [[1.0, 0.0]])
        self._write_embeddings(
            source,
            ["first.mp4", "second.mp4"],
            [[1.0, 0.0], [0.0, 1.0]],
        )
        video_dir = tmp_path / "videos"
        video_dir.mkdir()
        (video_dir / "first.mp4").write_bytes(b"first")
        (video_dir / "second.mp4").write_bytes(b"second")

        manifest = run_divknn_select_cli(
            target,
            source,
            tmp_path / "output",
            video_dir,
            top_n=1,
            backup=1,
            target_count=2,
        )

        assert manifest["copied"] == 2
        assert [item["file_name"] for item in manifest["files"]] == [
            "first.mp4",
            "second.mp4",
        ]

    def test_divknn_selection_rejects_missing_selected_clip(self, tmp_path):
        target = tmp_path / "target.parquet"
        source = tmp_path / "source.parquet"
        self._write_embeddings(target, ["target.mp4"], [[1.0, 0.0]])
        self._write_embeddings(source, ["missing.mp4"], [[1.0, 0.0]])
        video_dir = tmp_path / "videos"
        video_dir.mkdir()
        output_dir = tmp_path / "output"

        with pytest.raises(ClickException, match="Selected clip not found: missing.mp4"):
            run_divknn_select_cli(
                target,
                source,
                output_dir,
                video_dir,
                top_n=1,
                backup=1,
                target_count=1,
            )

        assert not (output_dir / "selection_manifest.json").exists()
        assert not (output_dir / "unique_selected_files.parquet").exists()
