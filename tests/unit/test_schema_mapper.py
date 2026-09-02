# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for Dataset Search ↔ Data Mining schema mapping."""

from __future__ import annotations

import json

from adapters.schema.mapper import (
    data_mining_row_to_record,
    dataset_search_row_to_record,
    enrich_dataset_search_meta,
    record_to_data_mining_row,
    record_to_dataset_search_row,
    record_to_tmm_row,
    records_to_data_mining_frame_dicts,
    records_to_dataset_search_frame_dicts,
)
from packages.domain.types import EmbeddingRecord


class TestSchemaMapper:
    def test_data_mining_roundtrip(self):
        row = {"file_name": "img1.jpg", "embedding": [0.1, 0.2], "camera": "front"}
        rec = data_mining_row_to_record(row)
        assert rec.record_id == "img1.jpg"
        assert list(rec.embedding) == [0.1, 0.2]
        assert rec.metadata["camera"] == "front"
        back = record_to_data_mining_row(rec)
        assert back["file_name"] == "img1.jpg"
        assert back["embedding"] == [0.1, 0.2]

    def test_tmm_filepath_roundtrip(self):
        row = {"filepath": "clip.mp4", "embedding": [0.3, 0.4], "label": "car"}
        rec = data_mining_row_to_record(row)
        assert rec.record_id == "clip.mp4"
        assert rec.metadata["label"] == "car"
        back = record_to_tmm_row(rec)
        assert back["filepath"] == "clip.mp4"
        assert "file_name" not in back
        assert back["embedding"] == [0.3, 0.4]

    def test_dataset_search_roundtrip_with_json_meta(self):
        row = {
            "id": "uuid-1",
            "embedding": [1.0, 0.0],
            "$meta": json.dumps({"span": 3}),
        }
        rec = dataset_search_row_to_record(row)
        assert rec.record_id == "uuid-1"
        assert rec.metadata["span"] == 3
        back = record_to_dataset_search_row(rec)
        assert back["id"] == "uuid-1"
        assert json.loads(back["$meta"])["span"] == 3

    def test_enrich_meta(self):
        rec = EmbeddingRecord("a", [1.0], {})
        enriched = enrich_dataset_search_meta(rec, cluster_id=2, diversity_rank=0, selected=True)
        assert enriched.metadata["cluster_id"] == 2
        assert enriched.metadata["diversity_rank"] == 0
        assert enriched.metadata["selected"] is True

    def test_dataset_search_row_rejects_missing_id(self):
        try:
            dataset_search_row_to_record({"embedding": [1.0]})
            raise AssertionError("expected KeyError")
        except KeyError:
            pass

    def test_dataset_search_dict_meta_and_batch_helpers(self):
        rec = dataset_search_row_to_record(
            {"id": "z", "embedding": [0.5], "$meta": {"ok": True}, "extra": 1}
        )
        assert rec.metadata["ok"] is True
        assert rec.metadata["extra"] == 1
        assert len(records_to_data_mining_frame_dicts([rec])) == 1
        assert len(records_to_dataset_search_frame_dicts([rec])) == 1

    def test_empty_string_meta(self):
        rec = dataset_search_row_to_record({"id": "z", "embedding": [0.5], "$meta": ""})
        assert rec.metadata == {}
