# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Safety checks for the local/operator Make façade."""

from __future__ import annotations

from pathlib import Path

MAKEFILE = Path(__file__).resolve().parents[2] / "Makefile"


def _recipe(target: str) -> str:
    text = MAKEFILE.read_text(encoding="utf-8")
    start = text.index(f"{target}:")
    next_target = text.find("\n\n", start)
    return text[start : next_target if next_target >= 0 else len(text)]


def test_integration_make_targets_are_noninteractive() -> None:
    for target in (
        "stage-cds-artifact",
        "ds-bulk-insert",
        "ds-job-status",
        "caption-readiness",
        "caption-search",
        "caption-upload",
        "caption-bulk-insert",
        "prepare-cds-ce1-for-tdm",
        "run-image-embeddings",
    ):
        assert "read -p" not in _recipe(target)


def test_makefile_uses_component_scoped_internal_cli() -> None:
    text = MAKEFILE.read_text(encoding="utf-8")

    assert "PAIDF_CLI ?= paidf_curation_and_retrieval" in text
    assert "uv run paidf " not in text


def test_object_store_make_targets_pass_credential_names_not_values() -> None:
    recipes = "\n".join(
        [
            _recipe("stage-cds-artifact"),
            _recipe("ds-bulk-insert"),
            _recipe("caption-bulk-insert"),
        ]
    )
    assert "--access-key-env" in recipes
    assert "--secret-key-env" in recipes
    assert "--access-key " not in recipes
    assert "--secret-key " not in recipes


def test_cds_bulk_and_tdm_handoff_require_ce1_declaration() -> None:
    recipes = "\n".join([_recipe("ds-bulk-insert"), _recipe("prepare-cds-ce1-for-tdm")])

    assert recipes.count('--embedding-family "$(CDS_EMBEDDING_FAMILY)"') == 2
    assert recipes.count("ERROR: CDS_EMBEDDING_FAMILY=ce1 required") == 2


def test_make_owns_public_tao_image_and_passes_it_explicitly() -> None:
    text = MAKEFILE.read_text(encoding="utf-8")

    assert "DATA_MINING_REGISTRY ?= nvcr.io/nvidia/tao/tao-toolkit" in text
    assert "DATA_MINING_TAG ?= 7.2.0-data-services" in text
    assert "DATA_MINING_IMAGE ?= tao-toolkit" in text
    assert "TDM_EMBEDDING_BACKEND ?= ce1" in text
    for target in (
        "run-data-mining-select",
        "run-data-mining-unique-match",
        "run-image-embeddings",
    ):
        assert '--image "$(DATA_MINING_IMAGE):$(DATA_MINING_TAG)"' in _recipe(target)
        if target != "run-image-embeddings":
            assert '--embedding-backend "$(TDM_EMBEDDING_BACKEND)"' in _recipe(target)


def test_make_documents_tao_ds_experiment_and_metric_contract() -> None:
    text = MAKEFILE.read_text(encoding="utf-8")

    assert "Every invocation requires an experiment spec" in text
    assert "DATA_MINING_METRIC=cosine|euclidean|manhattan" in text
    assert "distance_threshold" in text
    assert "SOURCE_EMBED_COLUMN_NAME" in text
    assert "run-data-mining-unique-match" in text
    assert "unique_neighbor_matching" in text
    assert "SOURCE_DETECTION_FILE" in text
    assert "RARE_CLASS_LIST" in text
    assert "FILTER_BY_LABEL" in text
    assert "DETECTION_FORMAT=coco (JSON file) | kitti (label dir)" in text
    assert "--source-embed-column-name" in _recipe("run-data-mining-select")
    assert "--filter-by-label" in _recipe("run-data-mining-select")
    assert "--source-detection-file" in _recipe("run-data-mining-unique-match")
    assert "--save-embeddings" in _recipe("run-data-mining-unique-match")


def test_make_exposes_text_embedding_subtask() -> None:
    text = MAKEFILE.read_text(encoding="utf-8")
    recipe = _recipe("run-text-embeddings")

    assert "TEXT_EMBEDDING_MODEL=clip|siglip|siglip2" in text
    assert "integration text-embeddings run" in recipe
    assert '--image "$(DATA_MINING_IMAGE):$(DATA_MINING_TAG)"' in recipe
    assert "read -p" not in recipe
    assert "integration text-embeddings validate-output" in _recipe(
        "text-embeddings-validate-output"
    )
