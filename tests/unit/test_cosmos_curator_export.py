# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Curator export → CDS parquet conversion."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from adapters.cosmos_curator.export_adapter import (
    CuratorExportError,
    convert_curator_dir_to_cds_parquet,
    find_embedding_parquet_dir,
    load_curator_export,
    process_single_file,
)

_FIRST_UUID = "11111111-1111-1111-1111-111111111111"
_SECOND_UUID = "22222222-2222-2222-2222-222222222222"


def _write_curator_export(
    root: Path,
    rows: list[dict[str, object]],
    metadata_by_uuid: dict[str, dict[str, str]],
    *,
    directory: str = "iv2_embd_parquet",
) -> Path:
    emb_dir = root / directory
    meta_dir = root / "metas" / "v0"
    emb_dir.mkdir(parents=True)
    meta_dir.mkdir(parents=True)
    parquet_path = emb_dir / "span.parquet"
    frame = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["id", "embedding"])
    frame.to_parquet(parquet_path, index=False)
    for uid, metadata in metadata_by_uuid.items():
        (meta_dir / f"{uid}.json").write_text(json.dumps(metadata), encoding="utf-8")
    return parquet_path


@pytest.fixture
def curator_dir(tmp_path: Path) -> Path:
    _write_curator_export(
        tmp_path,
        [{"id": _FIRST_UUID, "embedding": [0.1, 0.2, 0.3]}],
        {_FIRST_UUID: {"span_uuid": _FIRST_UUID, "camera": "front"}},
    )
    return tmp_path


class TestCuratorExport:
    def test_one_row_file_preserves_load_and_convert_behavior(
        self, curator_dir: Path, tmp_path: Path
    ):
        records = load_curator_export(curator_dir)
        assert len(records) == 1
        assert records[0].record_id == _FIRST_UUID
        assert list(records[0].embedding) == [0.1, 0.2, 0.3]
        assert records[0].metadata["camera"] == "front"

        out = tmp_path / "cds.parquet"
        convert_curator_dir_to_cds_parquet(curator_dir, out)
        df = pd.read_parquet(out)
        assert list(df.columns) == ["id", "embedding", "$meta"]
        assert json.loads(df.iloc[0]["$meta"])["camera"] == "front"

    def test_grouped_file_returns_and_writes_every_row_in_order(self, tmp_path: Path):
        _write_curator_export(
            tmp_path,
            [
                {"id": _FIRST_UUID, "embedding": [0.1, 0.2]},
                {"id": _SECOND_UUID, "embedding": [0.3, 0.4]},
            ],
            {
                _FIRST_UUID: {"span_uuid": _FIRST_UUID, "camera": "front"},
                _SECOND_UUID: {"span_uuid": _SECOND_UUID, "camera": "rear"},
            },
        )

        records = load_curator_export(tmp_path)

        assert [record.record_id for record in records] == [_FIRST_UUID, _SECOND_UUID]
        assert [record.metadata["camera"] for record in records] == ["front", "rear"]

        output = tmp_path / "cds.parquet"
        convert_curator_dir_to_cds_parquet(tmp_path, output)
        output_frame = pd.read_parquet(output)
        assert list(output_frame["id"]) == [_FIRST_UUID, _SECOND_UUID]
        assert [json.loads(value)["camera"] for value in output_frame["$meta"]] == [
            "front",
            "rear",
        ]

    def test_empty_parquet_is_rejected(self, tmp_path: Path):
        parquet_path = _write_curator_export(tmp_path, [], {})

        with pytest.raises(CuratorExportError, match="must contain at least one row"):
            process_single_file(parquet_path, tmp_path)

    def test_missing_sidecar_for_later_row_preserves_empty_metadata_contract(self, tmp_path: Path):
        _write_curator_export(
            tmp_path,
            [
                {"id": _FIRST_UUID, "embedding": [0.1, 0.2]},
                {"id": _SECOND_UUID, "embedding": [0.3, 0.4]},
            ],
            {_FIRST_UUID: {"span_uuid": _FIRST_UUID, "camera": "front"}},
        )

        records = load_curator_export(tmp_path)

        assert [record.record_id for record in records] == [_FIRST_UUID, _SECOND_UUID]
        assert records[1].metadata == {}

    def test_uuid_mismatch_on_later_row_is_rejected(self, tmp_path: Path):
        parquet_path = _write_curator_export(
            tmp_path,
            [
                {"id": _FIRST_UUID, "embedding": [0.1, 0.2]},
                {"id": _SECOND_UUID, "embedding": [0.3, 0.4]},
            ],
            {
                _FIRST_UUID: {"span_uuid": _FIRST_UUID},
                _SECOND_UUID: {"span_uuid": "other"},
            },
        )

        with pytest.raises(
            CuratorExportError,
            match=f"UUID mismatch: parquet={_SECOND_UUID} meta.span_uuid=other",
        ):
            process_single_file(parquet_path, tmp_path)

    @pytest.mark.parametrize("missing_column", ["id", "embedding"])
    def test_missing_required_column_is_rejected(self, tmp_path: Path, missing_column: str):
        row: dict[str, object] = {"id": _FIRST_UUID, "embedding": [0.1, 0.2]}
        del row[missing_column]
        parquet_path = _write_curator_export(tmp_path, [row], {})

        with pytest.raises(CuratorExportError, match="'id' and 'embedding' columns"):
            process_single_file(parquet_path, tmp_path)

    def test_load_cosmos_embed1_export(self, tmp_path: Path):
        _write_curator_export(
            tmp_path,
            [{"id": _SECOND_UUID, "embedding": [0.4, 0.5]}],
            {},
            directory="ce1_embd_224p_parquet",
        )
        found = find_embedding_parquet_dir(tmp_path, backend="ce1")
        assert found.name == "ce1_embd_224p_parquet"
        records = load_curator_export(tmp_path, backend="ce1")
        assert len(records) == 1
        assert records[0].record_id == _SECOND_UUID

    def test_uuid_mismatch(self, curator_dir: Path):
        bad = curator_dir / "metas" / "v0" / f"{_FIRST_UUID}.json"
        bad.write_text(json.dumps({"span_uuid": "other"}), encoding="utf-8")
        with pytest.raises(CuratorExportError, match="UUID mismatch"):
            process_single_file(curator_dir / "iv2_embd_parquet" / "span.parquet", curator_dir)

    def test_missing_parquet_dir(self, tmp_path: Path):
        with pytest.raises(CuratorExportError, match="No iv2_embd_parquet"):
            load_curator_export(tmp_path)
