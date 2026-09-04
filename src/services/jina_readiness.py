"""Structured 'is this machine ready to run Jina CLIP v2?' report.

One source of truth for three checks, consumed by both
``scripts/check_jina_setup.py`` (CLI) and ``GET /settings/jina/readiness``
(the Settings -> Cloud Assets tab in the UI):

  1. torch / GPU   -- is torch a CUDA build; what JINA_DEVICE resolves to
  2. the model     -- is the pinned jinaai/jina-clip-v2 snapshot on disk
  3. the index     -- are the four jina_* cloud artifacts synced + verified

Read-only. Never loads the model or the FAISS index; safe to call from a
status endpoint on every poll.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from src.config.settings import Settings, get_settings

_CUDA_FIX = (
    "pip install torch==2.13.0+cu126 torchvision==0.28.0+cu126 "
    "--index-url https://download.pytorch.org/whl/cu126"
)


def _check(cid: str, label: str, status: str, summary: str, **extra: Any) -> dict:
    out = {"id": cid, "label": label, "status": status, "summary": summary}
    out.update({k: v for k, v in extra.items() if v is not None})
    return out


def _human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} TB"


def _dir_size(p: Path) -> int:
    total = 0
    for root, _dirs, files in os.walk(p):
        for f in files:
            try:
                total += (Path(root) / f).stat().st_size
            except OSError:
                pass
    return total


# --------------------------------------------------------------------------
def _gpu_check(settings: Settings) -> dict:
    want = (settings.jina_device or "auto").strip().lower()
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        return _check(
            "gpu", "torch / GPU", "miss",
            f"torch is not importable ({type(exc).__name__})",
            detail="Install requirements.txt.",
        )
    ver = getattr(torch, "__version__", "?")
    cuda_build = getattr(torch.version, "cuda", None)
    try:
        avail = bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        avail = False
    name = None
    if avail:
        try:
            name = torch.cuda.get_device_name(0)
        except Exception:  # noqa: BLE001
            name = "CUDA device"
    resolved = "cpu" if (want == "cpu" or not avail) else "cuda"

    fields = {
        "torch_version": ver,
        "cuda_build": cuda_build,
        "cuda_available": avail,
        "device_name": name,
        "jina_device_setting": want,
        "resolved_device": resolved,
    }
    if resolved == "cuda":
        return _check(
            "gpu", "torch / GPU", "ok",
            f"torch {ver} (CUDA {cuda_build}) - GPU: {name}",
            detail=f"JINA_DEVICE={want} -> cuda", **fields,
        )
    return _check(
        "gpu", "torch / GPU", "warn",
        f"torch {ver} has no usable CUDA - Jina encodes on CPU (~10x slower)",
        detail=(
            f"JINA_DEVICE={want} -> cpu. If this machine has an NVIDIA GPU, "
            "install a CUDA build of torch."
        ),
        fix=_CUDA_FIX, **fields,
    )


# --------------------------------------------------------------------------
def _expected_pin(settings: Settings) -> tuple[str | None, str]:
    env = (getattr(settings, "jina_model_revision", None) or "").strip()
    if env:
        return env, "JINA_MODEL_REVISION"
    try:
        from src.services.assets import resolve_artifact_path

        meta_path = resolve_artifact_path("jina_index_meta", settings=settings)
        if meta_path is None and settings.jina_index_meta_path:
            meta_path = Path(settings.jina_index_meta_path)
        if meta_path and Path(meta_path).is_file():
            meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
            rev = str(meta.get("model_revision") or "").strip()
            if rev:
                return rev, "jina_index_meta.json"
    except Exception:  # noqa: BLE001
        pass
    return None, ""


def _snapshot_dir(settings: Settings) -> Path | None:
    src = (settings.jina_model_path or "jinaai/jina-clip-v2").strip()
    if src and Path(src).is_dir():
        return Path(src)
    hub = (
        Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
        / "hub" / "models--jinaai--jina-clip-v2" / "snapshots"
    )
    if not hub.is_dir():
        return None
    snaps = [d for d in hub.iterdir() if d.is_dir()]
    return max(snaps, key=lambda d: d.stat().st_mtime) if snaps else None


def _snapshot_revision(d: Path) -> str | None:
    from src.services.jina_retriever import (
        _IMMUTABLE_REV_RE,
        _LOCAL_REVISION_SIDECARS,
        _MOVING_REFS,
    )

    for name in _LOCAL_REVISION_SIDECARS:
        p = d / name
        try:
            if p.is_file():
                raw = p.read_text(encoding="utf-8").strip()
                if raw:
                    return raw
        except OSError:
            pass
    if d.name.lower() not in _MOVING_REFS and _IMMUTABLE_REV_RE.match(d.name):
        return d.name
    return None


def _revs_match(a: str, b: str) -> bool:
    a, b = a.strip().lower(), b.strip().lower()
    return bool(a and b and (a == b or a.startswith(b) or b.startswith(a)))


def _model_check(settings: Settings) -> dict:
    pin, pin_src = _expected_pin(settings)
    d = _snapshot_dir(settings)
    if d is None:
        return _check(
            "model", "Jina CLIP v2 model", "miss",
            "jinaai/jina-clip-v2 is not downloaded",
            detail=(
                "Auto-downloads (~1.7 GB) on the first search when "
                "JINA_LOCAL_FILES_ONLY=false, or pre-fetch it."
            ),
            fix=(
                "python -c \"from huggingface_hub import snapshot_download; "
                f"snapshot_download('jinaai/jina-clip-v2', revision='{pin or '<pin>'}')\""
            ),
            pin=pin, pin_source=pin_src or None,
        )
    has_weights = (d / "model.safetensors").is_file() or (d / "pytorch_model.bin").is_file()
    size = _dir_size(d)
    rev = _snapshot_revision(d)
    common = {"path": str(d), "size_bytes": size, "revision": rev, "pin": pin,
              "pin_source": pin_src or None}
    if not has_weights:
        return _check(
            "model", "Jina CLIP v2 model", "miss",
            f"snapshot at {d.name} has no model weights (incomplete download)",
            fix=(
                "python -c \"from huggingface_hub import snapshot_download; "
                f"snapshot_download('jinaai/jina-clip-v2', revision='{pin or '<pin>'}')\""
            ),
            **common,
        )
    if pin and rev:
        if _revs_match(rev, pin):
            return _check("model", "Jina CLIP v2 model", "ok",
                          f"present ({_human(size)}) - revision matches the pin", **common)
        return _check(
            "model", "Jina CLIP v2 model", "miss",
            f"present ({_human(size)}) but revision {rev} != pin {pin}",
            detail="Re-fetch the pinned commit.",
            fix=(
                "python -c \"from huggingface_hub import snapshot_download; "
                f"snapshot_download('jinaai/jina-clip-v2', revision='{pin}')\""
            ),
            **common,
        )
    if not pin:
        return _check("model", "Jina CLIP v2 model", "warn",
                      f"present ({_human(size)}) but no pin to check it against",
                      detail="Sync the index (its jina_index_meta.json carries the pin) "
                             "or set JINA_MODEL_REVISION.",
                      **common)
    return _check("model", "Jina CLIP v2 model", "warn",
                  f"present ({_human(size)}) - revision cannot be proven from the directory",
                  **common)


# --------------------------------------------------------------------------
def _index_check(settings: Settings) -> dict:
    from src.services.assets import (
        BACKEND_ARTIFACT_NAMES,
        build_asset_store,
        cloud_enabled,
        get_artifact_cache,
        get_manifest,
    )

    names = list(BACKEND_ARTIFACT_NAMES["jina_clip_v2"])
    cache = get_artifact_cache(settings)
    current = cache.get_current()
    common: dict[str, Any] = {"cache_path": str(cache.root), "current_version": current}

    manifest = None
    if cloud_enabled(settings):
        try:
            manifest = get_manifest(build_asset_store(settings))
        except Exception:  # noqa: BLE001
            manifest = None

    if manifest is not None:
        manifest_names = {a.name for a in manifest.artifacts}
        missing = [n for n in names if n not in manifest_names]
        if missing:
            return _check("index", "FAISS index + parquet", "miss",
                          f"manifest {manifest.version} is missing {missing}",
                          detail="Broken publish - nothing to sync.", **common)
        arts = [a for a in manifest.artifacts if a.name in set(names)]
        rows = []
        for a in arts:
            s = cache.slot(manifest.version, a.name, expected_sha=a.sha256, expected_size=a.size)
            rows.append({
                "name": a.name,
                "size_bytes": s.size if s.present else a.size,
                "status": "verified" if s.verified else ("unverified" if s.present else "missing"),
            })
        total = sum(r["size_bytes"] for r in rows)
        common.update(manifest_version=manifest.version, artifacts=rows, total_bytes=total)
        verified = cache.is_version_verified(manifest.version, arts)
        if verified and current == manifest.version:
            return _check("index", "FAISS index + parquet", "ok",
                          f"{len(names)}/{len(names)} artifacts verified "
                          f"({_human(total)}) - version {manifest.version} is current", **common)
        if verified:
            return _check("index", "FAISS index + parquet", "warn",
                          f"version {manifest.version} verified but not current "
                          f"(current={current or 'none'})",
                          detail="Restart / re-sync to promote it.", **common)
        return _check("index", "FAISS index + parquet", "miss",
                      f"version {manifest.version} is not fully synced",
                      detail="Start the app (autosync), press 'Sync artifacts' below, "
                             "or run `python -m launcher`.", **common)

    # No manifest: cloud off / unreachable -> fall back to local paths + on-disk.
    local = settings.jina_faiss_index_path
    if local and Path(local).is_file():
        gaps = [
            lbl for lbl, v in (
                ("JINA_GLOBAL_IDS_PATH", settings.jina_global_ids_path),
                ("JINA_INDEX_META_PATH", settings.jina_index_meta_path),
            ) if not (v and Path(v).is_file())
        ]
        if gaps:
            return _check("index", "FAISS index + parquet", "miss",
                          f"local JINA_FAISS_INDEX_PATH set but {gaps} missing", **common)
        return _check("index", "FAISS index + parquet", "ok",
                      "local JINA_*_PATH files all present", **common)
    if not current:
        return _check("index", "FAISS index + parquet", "miss",
                      "no synced index and cloud assets are off/unreachable",
                      detail="Enable CLOUD_ASSETS_ENABLED + AZURE_STORAGE_CONNECTION_STRING, "
                             "or point JINA_FAISS_INDEX_PATH / JINA_GLOBAL_IDS_PATH / "
                             "JINA_INDEX_META_PATH at local files.", **common)
    if cache.is_version_complete(current, names):
        total = sum(cache.slot(current, n).size for n in names if cache.slot(current, n).present)
        return _check("index", "FAISS index + parquet", "warn",
                      f"version {current}: all files present ({_human(total)}) but the "
                      "manifest is unreachable, so checksums were not re-verified", **common)
    return _check("index", "FAISS index + parquet", "miss",
                  f"version {current} is incomplete on disk and the manifest is unreachable",
                  **common)


# --------------------------------------------------------------------------
def jina_readiness(settings: Settings | None = None) -> dict:
    settings = settings or get_settings()
    from src.services.retrieval_backend import active_backend

    checks = [_gpu_check(settings), _model_check(settings), _index_check(settings)]
    # 'warn' does not block queries (CPU still works, unverified files still load);
    # only a 'miss' means the backend cannot serve.
    ok = all(c["status"] != "miss" for c in checks)
    return {
        "ok": ok,
        "active_backend": active_backend(settings),
        "retrieval_backend_setting": settings.retrieval_backend,
        "cloud_assets_enabled": bool(settings.cloud_assets_enabled),
        "checks": checks,
    }
