# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host-side handoff helpers for TAO DS ``tmm unique_neighbor_matching``.

This module does not run the UNM algorithm. It prepares and checks the
operator contract before and after ``DataMiningDockerRunner`` invokes
``tmm unique_neighbor_matching``:

* normalize Make/CLI knobs (policy, metric, columns, detection options)
* verify engine-native YAML paths stay under ``DATA_DIR`` (``/data``)
* enforce detection layout rules (COCO JSON file vs KITTI label directory)
* preflight S/B parquet schemas via :mod:`adapters.data_mining.tmm_parquet`
* confirm post-run artifacts such as ``final_unique_files.parquet``

Failures raise :class:`~adapters.data_mining.tmm_parquet.TmmParquetError`.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from adapters.data_mining.tmm_parquet import (
    TMM_METRICS,
    TmmColumnSelection,
    TmmParquetError,
    _selected_input,
    container_data_path,
    validate_tmm_parquet_pair,
)

UNM_ALLOCATION_POLICIES = frozenset({"global", "class_stratified"})
UNM_DETECTION_FORMATS = frozenset({"coco", "kitti"})
UNM_DETECTION_FILE_ROLES = frozenset({"source_detection_file", "target_detection_file"})
UNM_FINAL_PARQUET = "final_unique_files.parquet"
UNM_SUMMARY_JSON = "summary.json"


def _require_data_path(payload: Mapping[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.startswith("/data/"):
        raise TmmParquetError(f"UNM config {key} must be an absolute /data/... path")
    return value.removeprefix("/data/")


def _optional_data_path(payload: Mapping[str, object], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.startswith("/data/"):
        raise TmmParquetError(f"UNM config {key} must be an absolute /data/... path or null")
    return value.removeprefix("/data/")


def _nonempty_column(name: object, *, field: str) -> str:
    if not isinstance(name, str) or not name.strip():
        raise TmmParquetError(f"{field} must be a non-empty string")
    return name.strip()


def _validate_detection_path(
    candidate: Path,
    *,
    role: str,
    detection_format: str | None,
    label: str,
) -> None:
    """Enforce the TAO DS annotation shape: COCO JSON file or KITTI label directory."""
    if not candidate.exists():
        return
    fmt = str(detection_format or "").strip().lower() or None
    if fmt == "kitti":
        if not candidate.is_dir():
            raise TmmParquetError(f"{role} must be a KITTI label directory: {label}")
        if not any(candidate.glob("*.txt")):
            raise TmmParquetError(f"{role} KITTI directory has no .txt label files: {label}")
        return
    if fmt == "coco" and not candidate.is_file():
        raise TmmParquetError(f"{role} must be a COCO JSON file: {label}")
    if fmt is None and not candidate.is_file():
        raise TmmParquetError(f"{role} directory requires detection_format kitti: {label}")


def resolve_optional_data_selection(
    data_dir: str | Path,
    selection: str | None,
    *,
    role: str,
    must_exist: bool = True,
    detection_format: str | None = None,
) -> str | None:
    """Map an optional DATA_DIR-relative selection to a ``/data/...`` container path."""
    if selection is None:
        return None
    text = str(selection).strip()
    if not text:
        return None
    if text.startswith("/data/"):
        relative = text.removeprefix("/data/")
    else:
        relative = text
    root = Path(data_dir).resolve()
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise TmmParquetError(f"{role} must remain under DATA_DIR") from exc
    if must_exist and not candidate.exists():
        raise TmmParquetError(f"{role} not found under DATA_DIR: {relative}")
    if role in UNM_DETECTION_FILE_ROLES:
        _validate_detection_path(
            candidate,
            role=role,
            detection_format=detection_format,
            label=relative,
        )
    return container_data_path(candidate, root)


def validate_unm_direct_options(
    *,
    allocation_policy: str = "global",
    metric: str = "euclidean",
    desired_unique_count: int = 100,
    candidate_expansion_factor: int = 5,
    source_embedding_column: str = "embedding",
    target_embedding_column: str = "embedding",
    source_filepath_column: str = "filepath",
    target_filepath_column: str = "filepath",
    exclude_path: str | None = None,
    source_detection_file: str | None = None,
    target_detection_file: str | None = None,
    detection_format: str | None = None,
    rare_class_list: str = "",
    save_embeddings: bool = False,
    visualize: bool = False,
) -> dict[str, Any]:
    """Normalize Make/CLI UNM knobs into vendor experiment-spec fields.

    Checks policy/metric enums, positive integer bounds, boolean flags, and
    the extra detection inputs required by ``class_stratified``. Returns the
    payload fragment later written into the generated UNM YAML.
    """
    if (
        isinstance(desired_unique_count, bool)
        or not isinstance(desired_unique_count, int)
        or desired_unique_count < 1
    ):
        raise TmmParquetError("desired_unique_count must be a positive integer")
    if (
        isinstance(candidate_expansion_factor, bool)
        or not isinstance(candidate_expansion_factor, int)
        or candidate_expansion_factor < 1
    ):
        raise TmmParquetError("candidate_expansion_factor must be a positive integer")

    policy = str(allocation_policy or "").strip().lower()
    if policy not in UNM_ALLOCATION_POLICIES:
        raise TmmParquetError(
            f"allocation_policy must be one of: {', '.join(sorted(UNM_ALLOCATION_POLICIES))}"
        )
    normalized_metric = str(metric or "").strip().lower()
    if normalized_metric not in TMM_METRICS:
        raise TmmParquetError(f"distance_metric must be one of: {', '.join(sorted(TMM_METRICS))}")
    if not isinstance(save_embeddings, bool):
        raise TmmParquetError("save_embeddings must be a boolean")
    if not isinstance(visualize, bool):
        raise TmmParquetError("visualize must be a boolean")

    format_value = detection_format
    if format_value is not None:
        format_value = str(format_value).strip().lower() or None
    if format_value is not None and format_value not in UNM_DETECTION_FORMATS:
        raise TmmParquetError(
            f"detection_format must be one of: {', '.join(sorted(UNM_DETECTION_FORMATS))}"
        )

    rare = rare_class_list if isinstance(rare_class_list, str) else ""
    if policy == "class_stratified":
        if not source_detection_file or not str(source_detection_file).startswith("/data/"):
            raise TmmParquetError("class_stratified requires source_detection_file under /data")
        if not target_detection_file or not str(target_detection_file).startswith("/data/"):
            raise TmmParquetError("class_stratified requires target_detection_file under /data")
        if format_value is None:
            raise TmmParquetError("class_stratified requires detection_format")
        if not rare.strip():
            raise TmmParquetError("class_stratified requires rare_class_list")

    return {
        "desired_unique_count": desired_unique_count,
        "allocation_policy": policy,
        "distance_metric": normalized_metric,
        "candidate_expansion_factor": candidate_expansion_factor,
        "source_embedding_column": _nonempty_column(
            source_embedding_column, field="source_embedding_column"
        ),
        "target_embedding_column": _nonempty_column(
            target_embedding_column, field="target_embedding_column"
        ),
        "source_filepath_column": _nonempty_column(
            source_filepath_column, field="source_filepath_column"
        ),
        "target_filepath_column": _nonempty_column(
            target_filepath_column, field="target_filepath_column"
        ),
        "exclude_path": exclude_path,
        "source_detection_file": source_detection_file,
        "target_detection_file": target_detection_file,
        "detection_format": format_value,
        "rare_class_list": rare,
        "save_embeddings": save_embeddings,
        "visualize": visualize,
    }


def validate_unm_output(output_dir: str | Path, data_dir: str | Path) -> dict[str, Any]:
    """Confirm UNM wrote its required artifacts under ``DATA_DIR``.

    Requires ``final_unique_files.parquet`` and records whether
    ``summary.json`` is present. Used as post-run evidence after a live
    ``unique_neighbor_matching`` Docker job.
    """
    root = Path(data_dir).resolve()
    out = Path(output_dir)
    if not out.is_absolute():
        out = (root / out).resolve()
    else:
        out = out.resolve()
    try:
        out.relative_to(root)
    except ValueError as exc:
        raise TmmParquetError("UNM output_dir must remain under DATA_DIR") from exc
    final_parquet = out / UNM_FINAL_PARQUET
    summary = out / UNM_SUMMARY_JSON
    if not final_parquet.is_file():
        raise TmmParquetError(f"UNM output missing {UNM_FINAL_PARQUET} under {out}")
    evidence: dict[str, Any] = {
        "output_dir": str(out),
        "final_unique_files_parquet": str(final_parquet),
        "summary_json": str(summary) if summary.is_file() else None,
    }
    return evidence


def validate_unm_config_inputs(config_file: str | Path, data_dir: str | Path) -> int:
    """Preflight an engine-native UNM YAML against host ``DATA_DIR`` contents.

    Ensures required ``/data/...`` paths exist, detection files match the
    declared format, and the S/B parquet pair shares one embedding dimension.
    Returns that common dimension for caller evidence.
    """
    config_path = Path(config_file)
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise TmmParquetError(f"Unable to read UNM config: {config_path}") from exc
    if not isinstance(payload, Mapping):
        raise TmmParquetError("UNM config must contain a mapping")

    source_rel = _require_data_path(payload, "source_path")
    target_rel = _require_data_path(payload, "target_path")
    output_rel = _require_data_path(payload, "output_dir")

    desired = payload.get("desired_unique_count")
    if isinstance(desired, bool) or not isinstance(desired, int) or desired < 1:
        raise TmmParquetError("UNM config desired_unique_count must be a positive integer")
    expansion = payload.get("candidate_expansion_factor", 5)
    if isinstance(expansion, bool) or not isinstance(expansion, int) or expansion < 1:
        raise TmmParquetError("UNM config candidate_expansion_factor must be a positive integer")
    for flag in ("save_embeddings", "visualize"):
        value = payload.get(flag, False)
        if not isinstance(value, bool):
            raise TmmParquetError(f"UNM config {flag} must be a boolean")

    options = validate_unm_direct_options(
        allocation_policy=str(payload.get("allocation_policy", "global")),
        metric=str(payload.get("distance_metric", "euclidean")),
        desired_unique_count=desired,
        candidate_expansion_factor=expansion,
        source_embedding_column=str(payload.get("source_embedding_column", "embedding")),
        target_embedding_column=str(payload.get("target_embedding_column", "embedding")),
        source_filepath_column=str(payload.get("source_filepath_column", "filepath")),
        target_filepath_column=str(payload.get("target_filepath_column", "filepath")),
        exclude_path=(
            payload["exclude_path"] if isinstance(payload.get("exclude_path"), str) else None
        ),
        source_detection_file=(
            payload["source_detection_file"]
            if isinstance(payload.get("source_detection_file"), str)
            else None
        ),
        target_detection_file=(
            payload["target_detection_file"]
            if isinstance(payload.get("target_detection_file"), str)
            else None
        ),
        detection_format=(
            str(payload["detection_format"])
            if payload.get("detection_format") is not None
            else None
        ),
        rare_class_list=str(payload.get("rare_class_list", "")),
        save_embeddings=bool(payload.get("save_embeddings", False)),
        visualize=bool(payload.get("visualize", False)),
    )

    root = Path(data_dir).resolve()
    if not root.is_dir():
        raise TmmParquetError(f"DATA_DIR not found: {root}")

    output = (root / output_rel).resolve()
    try:
        output_relative = output.relative_to(root)
    except ValueError as exc:
        raise TmmParquetError("UNM config output_dir must remain under /data") from exc
    if not output_relative.parts:
        raise TmmParquetError("UNM config output_dir must name a directory under /data")

    target = _selected_input(root, target_rel, "Target")
    source = _selected_input(root, source_rel, "Source")

    source_det = payload.get("source_detection_file")
    target_det = payload.get("target_detection_file")
    for optional_key, optional_rel in (
        ("exclude_path", _optional_data_path(payload, "exclude_path")),
        (
            "source_detection_file",
            source_det.removeprefix("/data/") if isinstance(source_det, str) else None,
        ),
        (
            "target_detection_file",
            target_det.removeprefix("/data/") if isinstance(target_det, str) else None,
        ),
    ):
        if optional_rel is None:
            continue
        candidate = (root / optional_rel).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise TmmParquetError(f"UNM config {optional_key} must remain under /data") from exc
        if not candidate.exists():
            raise TmmParquetError(f"UNM config {optional_key} not found under DATA_DIR")
        if optional_key in UNM_DETECTION_FILE_ROLES:
            _validate_detection_path(
                candidate,
                role=f"UNM config {optional_key}",
                detection_format=(
                    str(payload["detection_format"])
                    if payload.get("detection_format") is not None
                    else None
                ),
                label=optional_rel,
            )

    return validate_tmm_parquet_pair(
        target,
        source,
        columns=TmmColumnSelection(
            target_embed=options["target_embedding_column"],
            source_embed=options["source_embedding_column"],
            target_filepath=options["target_filepath_column"],
            source_filepath=options["source_filepath_column"],
        ),
    )
