# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build Curator remote-API config from env for OpenAI-compatible HTTP clients.

Cosmos Curator does not take endpoint URLs from the pipeline cookbook. Stages
that use an OpenAI-compatible HTTP client (vLLM, NIM, public OpenAI, etc.)
read ``api_key`` + ``base_url`` from ``/cosmos_curator/config/cosmos_curator.yaml``.
This helper writes that file from environment variables (Airflow / SQA / Make).

Curator ``openai.<role>`` ← stage (when cookbook selects ``openai`` backend)::

  caption      captioning_algorithm / event caption
  filter       vlm_filter_endpoint
  classifier   video_classifier_endpoint
  enhance      enhance_captions_lm_variant
  embedding    OpenAI-compatible embedding (alternative to local IV2)

Per-role URLs (override independently)::

  COSMOS_CURATOR_OPENAI_CAPTION_BASE_URL
  COSMOS_CURATOR_OPENAI_FILTER_BASE_URL
  COSMOS_CURATOR_OPENAI_CLASSIFIER_BASE_URL
  COSMOS_CURATOR_OPENAI_ENHANCE_BASE_URL
  COSMOS_CURATOR_OPENAI_EMBEDDING_BASE_URL
  COSMOS_CURATOR_OPENAI_<ROLE>_API_KEY   (default EMPTY)

SQA aliases (Scenario A defaults)::

  SQA_VLM_BASE_URL → caption; also filter/classifier when those roles are unset
  SQA_LLM_BASE_URL → enhance

Optional: HF_TOKEN / COSMOS_CURATOR_HF_API_KEY; --merge-from host config.
Local GPU stages (SeedVR, SAM3, IV2, in-process Qwen) do not use these URLs.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

_MOUNT = "/cosmos_curator/config"


class OpenAIConfigEnvError(ValueError):
    """Invalid OpenAI-compatible endpoint env configuration."""


def _get(env: Mapping[str, str], *keys: str) -> str | None:
    for key in keys:
        value = env.get(key)
        if value and str(value).strip():
            return str(value).strip()
    return None


def _http_url(url: str, role: str) -> str:
    cleaned = url.strip().rstrip("/")
    if not cleaned.startswith(("http://", "https://")):
        raise OpenAIConfigEnvError(f"openai.{role}.base_url must be http(s), got: {url!r}")
    return cleaned


def _endpoint(url: str, role: str, env: Mapping[str, str]) -> dict[str, str]:
    key = _get(env, f"COSMOS_CURATOR_OPENAI_{role.upper()}_API_KEY") or "EMPTY"
    return {"api_key": key, "base_url": _http_url(url, role)}


def build_openai_section(environ: Mapping[str, str] | None = None) -> dict[str, Any] | None:
    """Build ``openai:`` role map from env (OpenAI-compatible HTTP clients)."""
    env = environ if environ is not None else os.environ
    caption = _get(env, "COSMOS_CURATOR_OPENAI_CAPTION_BASE_URL", "SQA_VLM_BASE_URL")
    enhance = _get(env, "COSMOS_CURATOR_OPENAI_ENHANCE_BASE_URL", "SQA_LLM_BASE_URL")
    # Filter/classifier default to caption/VLM URL; set explicitly to split VLMs per stage.
    filter_url = _get(env, "COSMOS_CURATOR_OPENAI_FILTER_BASE_URL") or caption
    classifier_url = _get(env, "COSMOS_CURATOR_OPENAI_CLASSIFIER_BASE_URL") or caption
    embedding = _get(env, "COSMOS_CURATOR_OPENAI_EMBEDDING_BASE_URL")

    urls = {
        "caption": caption,
        "filter": filter_url,
        "classifier": classifier_url,
        "enhance": enhance,
        "embedding": embedding,
    }
    if not any(urls.values()):
        return None

    return {role: _endpoint(url, role, env) for role, url in urls.items() if url}


def _load_merge(path: Path) -> dict[str, Any]:
    if path.is_dir():
        for name in ("cosmos_curator.yaml", "config.yaml"):
            candidate = path / name
            if candidate.is_file():
                path = candidate
                break
        else:
            return {}
    if not path.is_file():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise OpenAIConfigEnvError(f"Config must be a mapping: {path}")
    return data


def prepare_config_dir(
    output_dir: str | Path,
    *,
    merge_from: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> Path | None:
    """Write ``cosmos_curator.yaml`` (+ ``config.yaml``) from env; return dir or None."""
    env = environ if environ is not None else os.environ
    base = _load_merge(Path(merge_from).expanduser()) if merge_from else {}
    openai = build_openai_section(env)
    if openai is None and not base:
        return None

    cfg = dict(base)
    if openai:
        cfg["openai"] = {**(cfg.get("openai") or {}), **openai}
    hf = _get(env, "COSMOS_CURATOR_HF_API_KEY", "HF_TOKEN")
    if hf and "huggingface" not in cfg:
        cfg["huggingface"] = {"api_key": hf}

    out = Path(output_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    payload = yaml.safe_dump(cfg, default_flow_style=False, sort_keys=False)
    (out / "cosmos_curator.yaml").write_text(payload, encoding="utf-8")
    (out / "config.yaml").write_text(payload, encoding="utf-8")
    return out.resolve()


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "prepare":
        argv = argv[1:]

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--merge-from", default="")
    parser.add_argument("--print-docker-mount", action="store_true")
    args = parser.parse_args(argv)

    try:
        result = prepare_config_dir(args.output_dir, merge_from=args.merge_from or None)
    except OpenAIConfigEnvError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if result is None:
        if args.print_docker_mount:
            print("")
        return 0
    print(f"-v {result}:{_MOUNT}:ro" if args.print_docker_mount else result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
