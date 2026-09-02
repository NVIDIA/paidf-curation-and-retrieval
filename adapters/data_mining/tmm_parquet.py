# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare and check parquet inputs for TAO DS ``tmm nearest_neighbors``.

The DS image expects single parquet files with an identity column
(``filepath`` by default) and an ``embedding`` vector column. Operators may
still stage directories of parquet shards or legacy ``file_name`` layouts.
This helper:

* collapses directory selections into one parquet under ``DATA_DIR``
* normalizes identity/embedding column names for the experiment spec
* rejects empty selections, non-finite vectors, and dimension mismatches
* preflights engine-native ``TMM_CONFIG_FILE`` YAML path references

It does not perform k-NN search; :class:`~adapters.docker_jobs.DataMiningDockerRunner`
runs ``tmm nearest_neighbors`` after these checks pass.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real
from pathlib import Path

import pandas as pd
import yaml

TMM_METRICS = frozenset({"cosine", "euclidean", "manhattan"})


class TmmParquetError(ValueError):
    """Invalid layout or schema for TMM nearest_neighbors inputs."""


@dataclass(frozen=True)
class TmmColumnSelection:
    """Embedding/filepath column names declared for a target/source pair."""

    target_embed: str = "embedding"
    source_embed: str = "embedding"
    target_filepath: str = "filepath"
    source_filepath: str = "filepath"


def _collect_parquet_paths(path: Path) -> list[Path]:
    if path.is_file():
        if path.suffix.lower() != ".parquet":
            raise TmmParquetError(f"Expected a .parquet file, got: {path}")
        return [path]
    if not path.is_dir():
        raise TmmParquetError(f"Target/source path not found: {path}")
    files = sorted(path.glob("*.parquet"))
    if not files:
        raise TmmParquetError(f"No .parquet files under: {path}")
    return files


def _ensure_filepath_column(frame: pd.DataFrame, filepath_column: str) -> pd.DataFrame:
    if filepath_column in frame.columns:
        return frame
    if filepath_column != "filepath":
        raise TmmParquetError(f"Parquet must include the {filepath_column} column for TMM mining")
    for legacy in ("file_name", "id"):
        if legacy in frame.columns:
            return frame.rename(columns={legacy: "filepath"})
    raise TmmParquetError("Parquet must include a filepath, file_name, or id column for TMM mining")


def _load_and_normalize(
    paths: list[Path],
    *,
    embed_column: str = "embedding",
    filepath_column: str = "filepath",
) -> pd.DataFrame:
    frames = [pd.read_parquet(p) for p in paths]
    frame = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    if embed_column not in frame.columns:
        raise TmmParquetError(f"Parquet must include the {embed_column} embedding column")
    validate_embedding_dimension(frame, column=embed_column)
    return _ensure_filepath_column(frame, filepath_column)


def validate_embedding_dimension(
    frame: pd.DataFrame,
    *,
    error_type: type[ValueError] = TmmParquetError,
    column: str = "embedding",
) -> int:
    """Require finite, non-empty vectors with one consistent dimension."""
    dimensions: set[int] = set()
    for row_number, embedding in enumerate(frame[column], start=1):
        if isinstance(embedding, (str, bytes)):
            raise error_type(f"Embedding row {row_number} must be a non-empty numeric vector")
        try:
            values = list(embedding)
        except TypeError as exc:
            raise error_type(
                f"Embedding row {row_number} must be a non-empty numeric vector"
            ) from exc
        if not values:
            raise error_type(f"Embedding row {row_number} must be a non-empty numeric vector")
        if any(not isinstance(value, Real) or not math.isfinite(float(value)) for value in values):
            raise error_type(f"Embedding row {row_number} must contain finite numbers")
        dimensions.add(len(values))
    if not dimensions:
        raise error_type("Parquet must contain at least one embedding row")
    if len(dimensions) != 1:
        raise error_type(f"Parquet contains mixed embedding dimensions: {sorted(dimensions)}")
    return dimensions.pop()


def _selected_input(root: Path, selection: str, role: str) -> Path:
    if not selection.strip():
        raise TmmParquetError(f"{role} input selection must be non-empty")
    candidate = root / selection
    if not candidate.exists() and not candidate.suffix:
        parquet_candidate = candidate.with_suffix(".parquet")
        if parquet_candidate.exists():
            candidate = parquet_candidate
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise TmmParquetError(f"{role} input must be under DATA_DIR: {selection}") from exc
    return resolved


def _selected_output_dir(root: Path, selection: str) -> Path:
    if not selection.strip():
        raise TmmParquetError("TMM preparation output selection must be non-empty")
    resolved = (root / selection).resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise TmmParquetError(
            f"TMM preparation output must be under DATA_DIR: {selection}"
        ) from exc
    if not relative.parts:
        raise TmmParquetError("TMM preparation output must not be DATA_DIR itself")
    return resolved


def validate_embed_column_name(name: object, *, field: str) -> str:
    """Validate an optional TMM embedding/filepath column name."""
    if not isinstance(name, str) or not name.strip():
        raise TmmParquetError(f"{field} must be a non-empty string")
    return name.strip()


def _config_column(payload: Mapping[str, object], key: str, default: str) -> str:
    """Read an optional column name from a TAO experiment spec mapping."""
    if key not in payload:
        return default
    return validate_embed_column_name(payload[key], field=f"TMM config {key}")


def validate_distance_threshold(distance_threshold: object) -> float:
    """Validate nearest_neighbors distance_threshold (-1.0 disables filtering)."""
    if isinstance(distance_threshold, bool) or not isinstance(distance_threshold, Real):
        raise TmmParquetError("distance_threshold must be a finite number")
    value = float(distance_threshold)
    if not math.isfinite(value):
        raise TmmParquetError("distance_threshold must be a finite number")
    return value


def validate_tmm_run_options(
    data_dir: str | Path,
    *,
    output_subdir: str,
    prep_subdir: str = "_tmm_prep",
    topn: int = 5,
    metric: str = "cosine",
    distance_threshold: float = -1.0,
) -> tuple[Path, Path, str, float]:
    """Validate TMM run knobs and return bounded output/prep dirs plus metric."""
    root = Path(data_dir).resolve()
    if not root.is_dir():
        raise TmmParquetError(f"DATA_DIR not found: {root}")
    if isinstance(topn, bool) or not isinstance(topn, int) or topn < 1:
        raise TmmParquetError("topn must be a positive integer")
    normalized_metric = str(metric or "").strip().lower()
    if normalized_metric not in TMM_METRICS:
        raise TmmParquetError(f"knn_metric must be one of: {', '.join(sorted(TMM_METRICS))}")
    threshold = validate_distance_threshold(distance_threshold)
    return (
        _selected_output_dir(root, output_subdir),
        _selected_output_dir(root, prep_subdir),
        normalized_metric,
        threshold,
    )


def validate_tmm_parquet_pair(
    target_path: Path,
    source_path: Path,
    *,
    columns: TmmColumnSelection | None = None,
) -> int:
    """Validate selected TMM inputs and return their shared embedding dimension."""
    if target_path.resolve() == source_path.resolve():
        raise TmmParquetError("Target and source selections must be different")
    selection = columns or TmmColumnSelection()
    target = _load_and_normalize(
        _collect_parquet_paths(target_path),
        embed_column=selection.target_embed,
        filepath_column=selection.target_filepath,
    )
    source = _load_and_normalize(
        _collect_parquet_paths(source_path),
        embed_column=selection.source_embed,
        filepath_column=selection.source_filepath,
    )
    target_dimension = validate_embedding_dimension(target, column=selection.target_embed)
    source_dimension = validate_embedding_dimension(source, column=selection.source_embed)
    if target_dimension != source_dimension:
        raise TmmParquetError(
            "Target/source embedding dimensions do not match: "
            f"target={target_dimension}, source={source_dimension}"
        )
    return target_dimension


def validate_tmm_config_inputs(config_file: str | Path, data_dir: str | Path) -> int:
    """Validate supported TMM nearest_neighbors runtime fields and parquet inputs."""
    config_path = Path(config_file)
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise TmmParquetError(f"Unable to read TMM config: {config_path}") from exc
    if not isinstance(payload, Mapping):
        raise TmmParquetError("TMM config must contain a mapping")

    selections: dict[str, str] = {}
    for key in ("target_parquet", "source_parquet", "output_parquet"):
        value = payload.get(key)
        if not isinstance(value, str) or not value.startswith("/data/"):
            raise TmmParquetError(f"TMM config {key} must be an absolute /data/... path")
        selections[key] = value.removeprefix("/data/")

    topn = payload.get("topn")
    if isinstance(topn, bool) or not isinstance(topn, int) or topn < 1:
        raise TmmParquetError("TMM config topn must be a positive integer")

    metric = payload.get("knn_metric")
    if not isinstance(metric, str) or metric not in TMM_METRICS:
        raise TmmParquetError(
            f"TMM config knn_metric must be one of: {', '.join(sorted(TMM_METRICS))}"
        )

    filter_by_label = payload.get("filter_by_label")
    if not isinstance(filter_by_label, str) or filter_by_label not in {"true", "false"}:
        raise TmmParquetError('TMM config filter_by_label must be the string "true" or "false"')

    if "distance_threshold" in payload:
        validate_distance_threshold(payload["distance_threshold"])

    columns = TmmColumnSelection(
        target_embed=_config_column(payload, "target_embed_column_name", "embedding"),
        source_embed=_config_column(payload, "source_embed_column_name", "embedding"),
    )

    root = Path(data_dir).resolve()
    if not root.is_dir():
        raise TmmParquetError(f"DATA_DIR not found: {root}")
    output = (root / selections["output_parquet"]).resolve()
    try:
        output_relative = output.relative_to(root)
    except ValueError as exc:
        raise TmmParquetError("TMM config output_parquet must remain under /data") from exc
    if not output_relative.parts:
        raise TmmParquetError("TMM config output_parquet must name a file under /data")
    target = _selected_input(root, selections["target_parquet"], "Target")
    source = _selected_input(root, selections["source_parquet"], "Source")
    return validate_tmm_parquet_pair(target, source, columns=columns)


def prepare_tmm_parquet_pair(
    data_dir: str | Path,
    *,
    target_subdir: str = "S",
    source_subdir: str = "B",
    prep_subdir: str = "_tmm_prep",
    columns: TmmColumnSelection | None = None,
) -> tuple[Path, Path]:
    """Materialize ``target.parquet`` / ``source.parquet`` under ``prep_subdir``.

    ``target_subdir`` / ``source_subdir`` may be directories of parquet files or
    a single ``.parquet`` file path relative to ``data_dir``.

    Returns:
        Absolute host paths to the prepared target and source parquet files.
    """
    root = Path(data_dir).resolve()
    if not root.is_dir():
        raise TmmParquetError(f"DATA_DIR not found: {root}")

    target_src = _selected_input(root, target_subdir, "Target")
    source_src = _selected_input(root, source_subdir, "Source")
    prep = _selected_output_dir(root, prep_subdir)
    prep.mkdir(parents=True, exist_ok=True)

    target_out = prep / "target.parquet"
    source_out = prep / "source.parquet"

    selection = columns or TmmColumnSelection()
    target_frame = _load_and_normalize(
        _collect_parquet_paths(target_src),
        embed_column=selection.target_embed,
        filepath_column=selection.target_filepath,
    )
    source_frame = _load_and_normalize(
        _collect_parquet_paths(source_src),
        embed_column=selection.source_embed,
        filepath_column=selection.source_filepath,
    )
    target_dimension = validate_embedding_dimension(target_frame, column=selection.target_embed)
    source_dimension = validate_embedding_dimension(source_frame, column=selection.source_embed)
    if target_src == source_src:
        raise TmmParquetError("Target and source selections must be different")
    if target_dimension != source_dimension:
        raise TmmParquetError(
            "Target/source embedding dimensions do not match: "
            f"target={target_dimension}, source={source_dimension}"
        )
    target_frame.to_parquet(target_out, index=False)
    source_frame.to_parquet(source_out, index=False)
    return target_out, source_out


def container_data_path(host_path: Path, data_dir: str | Path) -> str:
    """Map an absolute host path under ``data_dir`` to the ``/data/...`` mount path."""
    root = Path(data_dir).resolve()
    resolved = host_path.resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise TmmParquetError(f"Prepared parquet {resolved} is not under DATA_DIR {root}") from exc
    return f"/data/{relative.as_posix()}"
