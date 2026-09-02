# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Image embedding workflow tests."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from apps.workflows import (
    BuildImageEmbeddingInputRequest,
    ValidateImageEmbeddingInputRequest,
    ValidateImageEmbeddingOutputRequest,
    build_image_embedding_input_handoff,
    validate_image_embedding_input_handoff,
    validate_image_embedding_output_handoff,
)


def test_build_image_embedding_input_uses_request_data_dir(tmp_path: Path) -> None:
    input_json = tmp_path / "rows.json"
    output_parquet = tmp_path / "input.parquet"
    input_json.write_text(json.dumps([{"image": "frame.jpg"}]), encoding="utf-8")
    seen: dict[str, Any] = {}

    def build_input(
        rows: Sequence[dict[str, Any]],
        output_path: str,
        *,
        data_dir: str,
    ) -> Path:
        seen["build"] = (rows, output_path, data_dir)
        return Path(output_path)

    def validate_input(parquet_path: str | Path, *, data_dir: str) -> dict[str, Any]:
        seen["validate"] = (parquet_path, data_dir)
        return {"rows": 1}

    payload = build_image_embedding_input_handoff(
        BuildImageEmbeddingInputRequest(
            input_json=str(input_json),
            data_dir="/request/data",
            output_parquet=str(output_parquet),
        ),
        build_input=build_input,
        validate_input=validate_input,
    )

    assert seen["build"] == ([{"image": "frame.jpg"}], str(output_parquet), "/request/data")
    assert seen["validate"] == (output_parquet, "/request/data")
    assert payload == {"status": "built", "rows": 1}


def test_validate_image_embedding_input_uses_request_data_dir() -> None:
    seen: dict[str, Any] = {}

    def validate_input(parquet_path: str | Path, *, data_dir: str) -> dict[str, Any]:
        seen["validate"] = (parquet_path, data_dir)
        return {"rows": 2}

    payload = validate_image_embedding_input_handoff(
        ValidateImageEmbeddingInputRequest(
            parquet_path="/tmp/input.parquet",
            data_dir="/request/data",
        ),
        validate_input=validate_input,
    )

    assert seen["validate"] == ("/tmp/input.parquet", "/request/data")
    assert payload == {"status": "valid", "rows": 2}


def test_validate_image_embedding_output_uses_request_paths() -> None:
    seen: dict[str, Any] = {}

    def validate_output(
        output_path: str | Path,
        *,
        data_dir: str,
        input_path: str | Path,
    ) -> dict[str, Any]:
        seen["validate"] = (output_path, data_dir, input_path)
        return {"rows": 3}

    payload = validate_image_embedding_output_handoff(
        ValidateImageEmbeddingOutputRequest(
            input_parquet="/tmp/input.parquet",
            output_parquet="/tmp/output.parquet",
            data_dir="/request/data",
        ),
        validate_output=validate_output,
    )

    assert seen["validate"] == ("/tmp/output.parquet", "/request/data", "/tmp/input.parquet")
    assert payload == {"status": "valid", "rows": 3}
