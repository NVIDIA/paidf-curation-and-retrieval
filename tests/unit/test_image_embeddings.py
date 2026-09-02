# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the TAO DS image-embedding parquet handoff."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from adapters.data_mining.image_embeddings import (
    ImageEmbeddingParquetError,
    build_image_embedding_input,
    validate_image_embedding_config,
    validate_image_embedding_input,
    validate_image_embedding_model,
    validate_image_embedding_output,
)


def _image(data_dir: Path, name: str = "images/a.jpg") -> Path:
    image = data_dir / name
    image.parent.mkdir(parents=True, exist_ok=True)
    image.write_bytes(b"image")
    return image


def _input_parquet(data_dir: Path) -> Path:
    _image(data_dir)
    return build_image_embedding_input(
        [{"filepath": "images/a.jpg", "label": "car", "score": 0.8}],
        data_dir / "manifests/input.parquet",
        data_dir=data_dir,
    )


def test_build_input_normalizes_paths_and_preserves_metadata(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    input_path = _input_parquet(data_dir)

    frame = pd.read_parquet(input_path)
    summary = validate_image_embedding_input(input_path, data_dir=data_dir)

    assert frame.to_dict("records") == [
        {
            "filepath": "/data/images/a.jpg",
            "label": "car",
            "score": 0.8,
        }
    ]
    assert summary["rows"] == 1


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ([{"label": "car"}], "required columns"),
        ([{"filepath": ""}], "non-empty string"),
        ([{"filepath": "../outside.jpg"}], "contained"),
        ([{"filepath": "images/missing.jpg"}], "not found"),
        ([{"filepath": "images/a.jpg", "embedding": [1.0]}], "reserved"),
    ],
)
def test_build_input_rejects_invalid_schema_and_paths(
    tmp_path: Path,
    rows: list[dict[str, object]],
    message: str,
) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _image(data_dir)

    with pytest.raises(ImageEmbeddingParquetError, match=message):
        build_image_embedding_input(
            rows,
            data_dir / "input.parquet",
            data_dir=data_dir,
        )


def test_build_input_rejects_empty_and_duplicate_images(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    empty = data_dir / "empty.jpg"
    empty.write_bytes(b"")

    with pytest.raises(ImageEmbeddingParquetError, match="non-empty file"):
        build_image_embedding_input(
            [{"filepath": "empty.jpg"}],
            data_dir / "empty.parquet",
            data_dir=data_dir,
        )

    _image(data_dir)
    with pytest.raises(ImageEmbeddingParquetError, match="unique"):
        build_image_embedding_input(
            [{"filepath": "images/a.jpg"}, {"filepath": "images/a.jpg"}],
            data_dir / "duplicate.parquet",
            data_dir=data_dir,
        )


def test_validate_output_checks_vectors_and_metadata(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    input_path = _input_parquet(data_dir)
    output_path = data_dir / "outputs/embeddings.parquet"
    output_path.parent.mkdir()
    pd.DataFrame(
        [
            {
                "filepath": "/data/images/a.jpg",
                "embedding": [0.1, 0.2, 0.3],
                "label": "car",
                "score": 0.8,
            }
        ]
    ).to_parquet(output_path, index=False)

    summary = validate_image_embedding_output(
        output_path,
        data_dir=data_dir,
        input_path=input_path,
    )

    assert summary["dimension"] == 3
    assert summary["metadata_columns"] == ["label", "score"]


@pytest.mark.parametrize(
    ("embeddings", "message"),
    [
        ([], "at least one row"),
        ([[]], "non-empty numeric vector"),
        ([[1.0, float("nan")]], "finite numbers"),
        ([[1.0], [1.0, 2.0]], "mixed embedding dimensions"),
    ],
)
def test_validate_output_rejects_invalid_vectors(
    tmp_path: Path,
    embeddings: list[list[float]],
    message: str,
) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    output = data_dir / "output.parquet"
    pd.DataFrame(
        {
            "filepath": [f"/data/{index}.jpg" for index in range(len(embeddings))],
            "embedding": embeddings,
        }
    ).to_parquet(output, index=False)

    with pytest.raises(ImageEmbeddingParquetError, match=message):
        validate_image_embedding_output(output, data_dir=data_dir)


def test_validate_output_rejects_missing_columns_metadata_and_output(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    input_path = _input_parquet(data_dir)
    output = data_dir / "output.parquet"
    pd.DataFrame({"filepath": ["/data/images/a.jpg"]}).to_parquet(output, index=False)

    with pytest.raises(ImageEmbeddingParquetError, match="required columns"):
        validate_image_embedding_output(output, data_dir=data_dir)
    with pytest.raises(ImageEmbeddingParquetError, match="not found"):
        validate_image_embedding_output(data_dir / "missing.parquet", data_dir=data_dir)

    pd.DataFrame({"filepath": ["/data/images/a.jpg"], "embedding": [[1.0, 2.0]]}).to_parquet(
        output, index=False
    )
    with pytest.raises(ImageEmbeddingParquetError, match="metadata columns"):
        validate_image_embedding_output(output, data_dir=data_dir, input_path=input_path)

    pd.DataFrame({"filepath": [""], "embedding": [[1.0, 2.0]]}).to_parquet(
        output,
        index=False,
    )
    with pytest.raises(ImageEmbeddingParquetError, match="non-empty string"):
        validate_image_embedding_output(output, data_dir=data_dir)


def test_validate_output_rejects_changed_metadata_and_identity(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    input_path = _input_parquet(data_dir)
    output = data_dir / "output.parquet"
    pd.DataFrame(
        {
            "filepath": ["/data/images/a.jpg"],
            "embedding": [[1.0]],
            "label": ["truck"],
            "score": [0.8],
        }
    ).to_parquet(output, index=False)
    with pytest.raises(ImageEmbeddingParquetError, match="metadata values"):
        validate_image_embedding_output(output, data_dir=data_dir, input_path=input_path)

    pd.DataFrame(
        {
            "filepath": ["/data/images/different.jpg"],
            "embedding": [[1.0]],
            "label": ["car"],
            "score": [0.8],
        }
    ).to_parquet(output, index=False)
    with pytest.raises(ImageEmbeddingParquetError, match="do not match"):
        validate_image_embedding_output(output, data_dir=data_dir, input_path=input_path)


def test_model_validation_supports_hf_and_contained_tao_checkpoint(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    checkpoint = _image(data_dir, "models/model.ckpt")
    config = _image(data_dir, "models/model.yaml")

    assert validate_image_embedding_model(
        "siglip",
        "google/siglip-base-patch16-224",
        data_dir=data_dir,
    ) == {
        "model": "SigLIP",
        "model_path": "google/siglip-base-patch16-224",
    }
    assert validate_image_embedding_model(
        "clip",
        str(checkpoint),
        data_dir=data_dir,
        model_config_path=str(config),
    ) == {
        "model": "CLIP",
        "model_path": "/data/models/model.ckpt",
        "model_config_path": "/data/models/model.yaml",
    }


def test_model_validation_rejects_unsupported_and_unsafe_options(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    checkpoint = _image(data_dir, "model.ckpt")

    with pytest.raises(ImageEmbeddingParquetError, match="clip, siglip"):
        validate_image_embedding_model("unknown", "model", data_dir=data_dir)
    with pytest.raises(ImageEmbeddingParquetError, match="only for model type clip"):
        validate_image_embedding_model(
            "siglip",
            str(checkpoint),
            data_dir=data_dir,
            model_config_path=str(checkpoint),
        )
    with pytest.raises(ImageEmbeddingParquetError, match="config path is required"):
        validate_image_embedding_model("clip", str(checkpoint), data_dir=data_dir)
    with pytest.raises(ImageEmbeddingParquetError, match="contained"):
        validate_image_embedding_model(
            "clip",
            str(tmp_path / "outside.ckpt"),
            data_dir=data_dir,
            model_config_path=str(checkpoint),
        )


def test_validate_engine_config_checks_contract(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _input_parquet(data_dir)
    config = tmp_path / "embedding.yaml"
    config.write_text(
        "\n".join(
            [
                "input_parquet: /data/manifests/input.parquet",
                "output_parquet: /data/outputs/embeddings.parquet",
                "model: CLIP",
                "model_path: openai/clip-vit-base-patch32",
                "batch_size: 8",
            ]
        ),
        encoding="utf-8",
    )

    result = validate_image_embedding_config(config, data_dir=data_dir)

    assert result["batch_size"] == 8
    assert result["model"]["model"] == "CLIP"

    config.write_text(
        "input_parquet: /tmp/input.parquet\n"
        "output_parquet: /data/output.parquet\n"
        "model: CLIP\n"
        "model_path: openai/clip-vit-base-patch32\n",
        encoding="utf-8",
    )
    with pytest.raises(ImageEmbeddingParquetError, match="contained"):
        validate_image_embedding_config(config, data_dir=data_dir)
