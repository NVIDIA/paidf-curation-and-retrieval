# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SeedVR2 checkpoint ensure helpers for Cosmos Curator preflight.

Mirrors the pseudo-labeling layout and HuggingFace sources:

  <MODELS_DIR>/seedvr2/
    ema_vae.pth              # from ByteDance-Seed/SeedVR2-7B (fallback: 3B)
    seedvr2_ema_3b.pth       # from ByteDance-Seed/SeedVR2-3B
    seedvr2_ema_7b.pth       # from ByteDance-Seed/SeedVR2-7B
    pos_emb.pt               # fixed text embeds (Curator SuperResolutionStage)
    neg_emb.pt

Curator resolves DiT/VAE weights from ``./ckpts`` under
``/opt/cosmos-curator/SeedVR`` (host ``seedvr2/`` is bind-mounted there).

Curator also loads ``pos_emb.pt`` / ``neg_emb.pt`` from the HF weights path
``/config/models/ByteDance-Seed/SeedVR2-{3B|7B}/``. Those files are stored in
host ``seedvr2/`` (writable) and bind-mounted onto the HF paths at run time —
avoids writing into a possibly root-owned empty HF stub directory.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections.abc import Sequence
from pathlib import Path

import requests

HF_REPO_SEEDVR2_3B_DEFAULT = "ByteDance-Seed/SeedVR2-3B"
HF_REPO_SEEDVR2_7B_DEFAULT = "ByteDance-Seed/SeedVR2-7B"
SEEDVR_CONTAINER_CKPTS = "/opt/cosmos-curator/SeedVR/ckpts"
SEEDVR_CONTAINER_HF_3B = "/config/models/ByteDance-Seed/SeedVR2-3B"
SEEDVR_CONTAINER_HF_7B = "/config/models/ByteDance-Seed/SeedVR2-7B"
EMA_VAE = "ema_vae.pth"
EMA_3B = "seedvr2_ema_3b.pth"
EMA_7B = "seedvr2_ema_7b.pth"
POS_EMB = "pos_emb.pt"
NEG_EMB = "neg_emb.pt"
TEXT_EMBEDS = (POS_EMB, NEG_EMB)


class SeedVRCkptError(RuntimeError):
    """Raised when SeedVR2 checkpoints cannot be ensured."""


def seedvr_ckpts_dir(models_dir: str | Path) -> Path:
    """Return ``<models_dir>/seedvr2`` (host layout matching pseudo-labeling)."""
    return Path(models_dir).expanduser().resolve() / "seedvr2"


def seedvr_text_embeds_dir(models_dir: str | Path) -> Path:
    """Directory for ``pos_emb.pt`` / ``neg_emb.pt``.

    Prefer ``seedvr2/`` when it is writable; otherwise use
    ``<models_dir>/seedvr2_embeds/`` (avoids root-owned seedvr2 stubs).
    """
    ckpts = seedvr_ckpts_dir(models_dir)
    if _dir_is_writable(ckpts):
        return ckpts
    return Path(models_dir).expanduser().resolve() / "seedvr2_embeds"


def _dir_is_writable(path: Path) -> bool:
    """True if ``path`` exists and this process can create files in it."""
    try:
        if not path.is_dir():
            path.mkdir(parents=True, exist_ok=True)
        probe = path / ".paidf_write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return True
    except OSError:
        return False


def normalize_seedvr_variant(variant: str) -> str:
    """Normalize variant string to ``3b`` or ``7b``."""
    text = (variant or "").strip().lower()
    if "7b" in text:
        return "7b"
    if "3b" in text or text in {"", "seedvr2", "seedvr"}:
        return "3b"
    raise SeedVRCkptError(
        f"Unsupported SeedVR variant {variant!r}; expected seedvr2_3b or seedvr2_7b"
    )


def seedvr_hf_repo_id(variant: str) -> str:
    """HuggingFace repo id for the variant (text embeds + DiT live here)."""
    mode = normalize_seedvr_variant(variant)
    if mode == "7b":
        return (os.environ.get("HF_REPO_SEEDVR2_7B") or HF_REPO_SEEDVR2_7B_DEFAULT).strip()
    return (os.environ.get("HF_REPO_SEEDVR2_3B") or HF_REPO_SEEDVR2_3B_DEFAULT).strip()


def seedvr_container_hf_dir(variant: str) -> str:
    """In-container Curator HF weights directory for text embeds."""
    return (
        SEEDVR_CONTAINER_HF_7B
        if normalize_seedvr_variant(variant) == "7b"
        else SEEDVR_CONTAINER_HF_3B
    )


def required_seedvr_filenames(variant: str) -> tuple[str, ...]:
    """Filenames required under ``seedvr2/`` for the given variant."""
    mode = normalize_seedvr_variant(variant)
    dit = EMA_7B if mode == "7b" else EMA_3B
    return (EMA_VAE, dit, POS_EMB, NEG_EMB)


def _seedvr_file_path(models_dir: str | Path, filename: str) -> Path:
    """Resolve host path for a SeedVR asset (ckpts vs text-embed dir).

    Text embeds may live in writable ``seedvr2_embeds/`` when ``seedvr2/`` is
    root-owned. Prefer an existing non-empty file in either location.
    """
    if filename in TEXT_EMBEDS:
        embeds = seedvr_text_embeds_dir(models_dir)
        for root in (seedvr_ckpts_dir(models_dir), embeds):
            path = root / filename
            if path.is_file() and path.stat().st_size > 0:
                return path
        return embeds / filename
    return seedvr_ckpts_dir(models_dir) / filename


def missing_seedvr_ckpts(models_dir: str | Path, variant: str = "seedvr2_3b") -> list[str]:
    """Return required checkpoint basenames that are missing or empty."""
    missing: list[str] = []
    for name in required_seedvr_filenames(variant):
        path = _seedvr_file_path(models_dir, name)
        if not path.is_file() or path.stat().st_size <= 0:
            missing.append(name)
    return missing


def config_requires_seedvr(config_path: str | Path | None) -> bool:
    """True when a Curator YAML enables ``super_resolution``."""
    if not config_path:
        return False
    path = Path(config_path)
    if not path.is_file():
        return False
    raw = path.read_text(encoding="utf-8")
    # Match cookbook form: `super_resolution: true` (ignore comments / false).
    return bool(re.search(r"(?m)^super_resolution:\s*true\s*(?:#.*)?$", raw))


def resolve_hf_token(explicit: str | None = None) -> str | None:
    """Resolve HF token from arg, ``HF_TOKEN``, or cosmos_curator hf_token.txt."""
    if explicit and explicit.strip():
        return explicit.strip()
    env = os.environ.get("HF_TOKEN", "").strip()
    if env:
        return env
    token_path = Path.home() / ".config" / "cosmos_curator" / "hf_token.txt"
    if token_path.is_file():
        value = token_path.read_text(encoding="utf-8").strip()
        return value or None
    return None


def _hf_resolve_url(repo_id: str, filename: str) -> str:
    return f"https://huggingface.co/{repo_id}/resolve/main/{filename}"


def download_hf_file(
    *,
    repo_id: str,
    filename: str,
    dst: Path,
    hf_token: str | None = None,
    timeout_s: int = 600,
    chunk_size: int = 8 * 1024 * 1024,
) -> None:
    """Download one HuggingFace file to ``dst`` (atomic via ``.partial``)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_file() and dst.stat().st_size > 0:
        return

    url = _hf_resolve_url(repo_id, filename)
    headers = {}
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"

    partial = dst.with_suffix(dst.suffix + ".partial")
    with requests.get(url, headers=headers, stream=True, timeout=timeout_s) as response:
        response.raise_for_status()
        with partial.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    handle.write(chunk)
    partial.replace(dst)


def ensure_seedvr_ckpts(
    models_dir: str | Path,
    *,
    variant: str = "seedvr2_3b",
    hf_token: str | None = None,
    download: bool = True,
    hf_repo_3b: str | None = None,
    hf_repo_7b: str | None = None,
) -> Path:
    """Ensure required SeedVR2 files exist under ``<models_dir>/seedvr2``.

    Sources match pseudo-labeling ``ensure_seedvr2_ckpts`` plus Curator text
    embeds:
    - ``ema_vae.pth`` from 7B then 3B repos
    - ``seedvr2_ema_3b.pth`` from 3B repo
    - ``seedvr2_ema_7b.pth`` from 7B repo
    - ``pos_emb.pt`` / ``neg_emb.pt`` from the variant repo
    """
    ckpts_root = seedvr_ckpts_dir(models_dir)
    embeds_root = seedvr_text_embeds_dir(models_dir)
    try:
        embeds_root.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise SeedVRCkptError(f"Cannot create SeedVR text-embed dir {embeds_root}: {exc}") from exc
    missing = missing_seedvr_ckpts(models_dir, variant)
    if not missing:
        return ckpts_root
    if not download:
        raise SeedVRCkptError(
            "Missing SeedVR2 checkpoints under "
            f"{ckpts_root} / {embeds_root}: {', '.join(missing)}. "
            "Run: make download-seedvr2 MODELS_DIR=... "
            f"SEEDVR_VARIANT={variant}"
        )

    token = resolve_hf_token(hf_token)
    repo_3b = (
        hf_repo_3b or os.environ.get("HF_REPO_SEEDVR2_3B") or HF_REPO_SEEDVR2_3B_DEFAULT
    ).strip()
    repo_7b = (
        hf_repo_7b or os.environ.get("HF_REPO_SEEDVR2_7B") or HF_REPO_SEEDVR2_7B_DEFAULT
    ).strip()
    if not token:
        print(
            "WARNING: HF_TOKEN unset; gated HuggingFace downloads may fail. "
            "Set HF_TOKEN or ~/.config/cosmos_curator/hf_token.txt",
            file=sys.stderr,
        )

    variant_repo = repo_7b if normalize_seedvr_variant(variant) == "7b" else repo_3b

    for filename in missing:
        dst = _seedvr_file_path(models_dir, filename)
        if filename == EMA_VAE:
            last_error: Exception | None = None
            for repo_id in (repo_7b, repo_3b):
                try:
                    print(f"[seedvr] downloading {repo_id}:{filename} -> {dst}", flush=True)
                    download_hf_file(repo_id=repo_id, filename=filename, dst=dst, hf_token=token)
                    break
                except Exception as exc:  # noqa: BLE001 — try fallback repo
                    last_error = exc
            else:
                raise SeedVRCkptError(
                    f"Failed to download {filename} from {repo_7b} / {repo_3b}"
                ) from last_error
            continue

        if filename in TEXT_EMBEDS:
            repo_id = variant_repo
        elif filename == EMA_3B:
            repo_id = repo_3b
        else:
            repo_id = repo_7b
        print(f"[seedvr] downloading {repo_id}:{filename} -> {dst}", flush=True)
        try:
            download_hf_file(repo_id=repo_id, filename=filename, dst=dst, hf_token=token)
        except Exception as exc:  # noqa: BLE001
            raise SeedVRCkptError(f"Failed to download {filename} from {repo_id}") from exc

    still = missing_seedvr_ckpts(models_dir, variant)
    if still:
        raise SeedVRCkptError(f"SeedVR2 ckpts still missing after download: {', '.join(still)}")
    return ckpts_root


def seedvr_docker_mount_args(
    models_dir: str | Path,
    *,
    variant: str = "seedvr2_3b",
    container_ckpts: str = SEEDVR_CONTAINER_CKPTS,
    read_only: bool = True,
) -> list[str]:
    """Docker ``-v`` args for SeedVR ckpts dir + HF text-embed file mounts."""
    host = seedvr_ckpts_dir(models_dir)
    mode = "ro" if read_only else "rw"
    args = ["-v", f"{host}:{container_ckpts}:{mode}"]
    hf_dir = seedvr_container_hf_dir(variant)
    for name in TEXT_EMBEDS:
        host_file = _seedvr_file_path(models_dir, name)
        if host_file.is_file() and host_file.stat().st_size > 0:
            args.extend(["-v", f"{host_file}:{hf_dir}/{name}:{mode}"])
    return args


def preflight_seedvr_for_config(
    models_dir: str | Path,
    config_path: str | Path | None,
    *,
    variant: str = "seedvr2_3b",
    ensure: str = "auto",
) -> list[str]:
    """Preflight SeedVR when the config needs SR.

    ``ensure``: ``auto`` (download if needed), ``check`` (fail if missing),
    ``skip`` (no download; mount only if present), ``always`` (ensure even if
    SR disabled).
    """
    mode = (ensure or "auto").strip().lower()
    needs = config_requires_seedvr(config_path) or mode == "always"
    if not needs:
        print(
            "OK: SeedVR not required by config (super_resolution disabled/absent)",
            file=sys.stderr,
        )
        return []

    missing = missing_seedvr_ckpts(models_dir, variant)
    if mode in {"skip", "0", "false", "no"}:
        if missing:
            print(
                "WARNING: SeedVR required but checkpoints missing under "
                f"{seedvr_ckpts_dir(models_dir)}: {', '.join(missing)}. "
                "Run make download-seedvr2 or ENSURE_SEEDVR_CKPTS=auto.",
                file=sys.stderr,
            )
            return []
        print(
            f"OK: SeedVR2 checkpoints present under {seedvr_ckpts_dir(models_dir)}", file=sys.stderr
        )
        return seedvr_docker_mount_args(models_dir, variant=variant)

    download = mode in {"auto", "always", "download", "1", "true", "yes"}
    if mode == "check":
        download = False

    root = ensure_seedvr_ckpts(models_dir, variant=variant, download=download)
    print(f"OK: SeedVR2 checkpoints ready under {root}", file=sys.stderr)
    return seedvr_docker_mount_args(models_dir, variant=variant)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    ensure_p = sub.add_parser("ensure", help="Download missing SeedVR2 checkpoints")
    ensure_p.add_argument("--models-dir", required=True)
    ensure_p.add_argument("--variant", default="seedvr2_3b")
    ensure_p.add_argument("--hf-token", default=None)

    check_p = sub.add_parser("check", help="Fail if required SeedVR2 checkpoints are missing")
    check_p.add_argument("--models-dir", required=True)
    check_p.add_argument("--variant", default="seedvr2_3b")

    pref_p = sub.add_parser("preflight", help="Config-aware SeedVR preflight for make")
    pref_p.add_argument("--models-dir", required=True)
    pref_p.add_argument("--config", default=None)
    pref_p.add_argument("--variant", default="seedvr2_3b")
    pref_p.add_argument(
        "--ensure",
        default="auto",
        help="auto|check|skip|always (default: auto)",
    )
    pref_p.add_argument(
        "--print-docker-args",
        action="store_true",
        help="Print docker -v args on stdout (one line) when SR is required",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for Makefile preflight / download targets."""
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "ensure":
            ensure_seedvr_ckpts(args.models_dir, variant=args.variant, hf_token=args.hf_token)
            return 0
        if args.command == "check":
            missing = missing_seedvr_ckpts(args.models_dir, args.variant)
            if missing:
                raise SeedVRCkptError(
                    f"Missing under {seedvr_ckpts_dir(args.models_dir)}: {', '.join(missing)}"
                )
            print(f"OK: {seedvr_ckpts_dir(args.models_dir)}")
            return 0
        if args.command == "preflight":
            mount_args = preflight_seedvr_for_config(
                args.models_dir,
                args.config,
                variant=args.variant,
                ensure=args.ensure,
            )
            if args.print_docker_args and mount_args:
                # Emit a single shell-friendly token string.
                print(" ".join(mount_args))
            return 0
    except SeedVRCkptError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except requests.HTTPError as exc:
        print(f"ERROR: HuggingFace download failed: {exc}", file=sys.stderr)
        return 1
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
