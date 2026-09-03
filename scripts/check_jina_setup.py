#!/usr/bin/env python
"""Am I ready to run the Jina CLIP v2 retrieval backend?

A no-server, read-only check every teammate can run:

    python scripts/check_jina_setup.py

It reports three things and, for anything missing, the exact fix:

  1. torch / GPU   -- is torch a CUDA build, and what does JINA_DEVICE=auto pick?
  2. the model     -- is the pinned jinaai/jina-clip-v2 snapshot on disk?
  3. the index     -- are the four jina_* cloud artifacts synced + checksum-verified?

Exit code 0 = ready, 1 = something is missing (CI-friendly).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))

OK, BAD, WARN = "  [ OK ]", "  [MISS]", "  [WARN]"
_problems = 0


def _fail(msg: str) -> None:
    global _problems
    _problems += 1
    print(f"{BAD} {msg}")


def _human(n: int | float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} TB"


def _dir_size(p: Path) -> int:
    total = 0
    for root, _dirs, files in os.walk(p):
        for f in files:
            fp = Path(root) / f
            try:
                total += fp.stat().st_size  # follows the HF blob symlinks
            except OSError:
                pass
    return total


# --------------------------------------------------------------------------
def check_torch(jina_device: str) -> str:
    print("\n[1] torch / GPU")
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        _fail(f"torch is not importable ({type(exc).__name__}: {exc}). Install requirements.txt.")
        return "cpu"

    ver = getattr(torch, "__version__", "?")
    cuda_build = getattr(torch.version, "cuda", None)
    try:
        avail = bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        avail = False

    want = (jina_device or "auto").strip().lower()
    resolved = "cpu"
    if want == "cpu":
        resolved = "cpu"
    elif avail:
        resolved = "cuda"
    # want in {auto, cuda} but not available -> cpu

    if avail:
        try:
            name = torch.cuda.get_device_name(0)
        except Exception:  # noqa: BLE001
            name = "cuda device"
        print(f"{OK} torch {ver} (CUDA {cuda_build}) -- GPU: {name}")
        print(f"       JINA_DEVICE={want} -> {resolved}")
    else:
        print(f"{WARN} torch {ver} has no usable CUDA (cuda build={cuda_build}).")
        print(f"       JINA_DEVICE={want} -> cpu  (Jina encode ~10x slower).")
        print("       If this machine has an NVIDIA GPU, install a CUDA build, e.g.:")
        print("         pip install torch==2.13.0+cu126 torchvision==0.28.0+cu126 \\")
        print("           --index-url https://download.pytorch.org/whl/cu126")
    return resolved


# --------------------------------------------------------------------------
_IMMUTABLE_HEX = None
_MOVING = None
_SIDECARS = None


def _rev_helpers():
    global _IMMUTABLE_HEX, _MOVING, _SIDECARS
    if _IMMUTABLE_HEX is None:
        from src.services.jina_retriever import (
            _IMMUTABLE_REV_RE,
            _LOCAL_REVISION_SIDECARS,
            _MOVING_REFS,
        )

        _IMMUTABLE_HEX, _MOVING, _SIDECARS = _IMMUTABLE_REV_RE, _MOVING_REFS, _LOCAL_REVISION_SIDECARS
    return _IMMUTABLE_HEX, _MOVING, _SIDECARS


def _expected_pin(settings) -> tuple[str | None, str]:
    """(pin, where) -- JINA_MODEL_REVISION wins, else jina_index_meta.json."""
    env = (getattr(settings, "jina_model_revision", None) or "").strip()
    if env:
        return env, "JINA_MODEL_REVISION"
    try:
        import json

        from src.services.assets import resolve_artifact_path

        meta_path = resolve_artifact_path("jina_index_meta", settings=settings) or (
            Path(settings.jina_index_meta_path) if settings.jina_index_meta_path else None
        )
        if meta_path and Path(meta_path).is_file():
            meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
            rev = str(meta.get("model_revision") or "").strip()
            if rev:
                return rev, "jina_index_meta.json"
    except Exception:  # noqa: BLE001
        pass
    return None, ""


def _snapshot_dir(settings) -> Path | None:
    """The on-disk jinaai/jina-clip-v2 snapshot, if any: an explicit local
    JINA_MODEL_PATH dir, else the newest HF cache snapshot."""
    src = (settings.jina_model_path or "jinaai/jina-clip-v2").strip()
    if src and Path(src).is_dir():
        return Path(src)
    hub = Path(
        os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")
    ) / "hub" / "models--jinaai--jina-clip-v2" / "snapshots"
    if not hub.is_dir():
        return None
    snaps = [d for d in hub.iterdir() if d.is_dir()]
    if not snaps:
        return None
    return max(snaps, key=lambda d: d.stat().st_mtime)


def _snapshot_revision(d: Path) -> str | None:
    hex_re, moving, sidecars = _rev_helpers()
    for name in sidecars:
        p = d / name
        try:
            if p.is_file():
                raw = p.read_text(encoding="utf-8").strip()
                if raw:
                    return raw
        except OSError:
            pass
    if d.name.lower() not in moving and hex_re.match(d.name):
        return d.name
    return None


def check_model(settings) -> None:
    print("\n[2] Jina CLIP v2 model")
    pin, where = _expected_pin(settings)
    if pin:
        print(f"       pinned revision: {pin}  (from {where})")
    else:
        print(f"{WARN} no model pin found yet (JINA_MODEL_REVISION unset and no synced "
              "jina_index_meta.json). Sync the index first, or set JINA_MODEL_REVISION.")

    d = _snapshot_dir(settings)
    if d is None:
        _fail("jinaai/jina-clip-v2 is not on disk. It auto-downloads (~1.7 GB) on the "
              "first search when JINA_LOCAL_FILES_ONLY=false, or pre-fetch it:")
        rev = pin or "<pin>"
        print(f"         python -c \"from huggingface_hub import snapshot_download; "
              f"snapshot_download('jinaai/jina-clip-v2', revision='{rev}')\"")
        return

    has_weights = (d / "model.safetensors").is_file() or (d / "pytorch_model.bin").is_file()
    size = _dir_size(d)
    rev = _snapshot_revision(d)
    line = f"present: {d}  ({_human(size)})"
    if not has_weights:
        _fail(f"{line} -- but no model.safetensors / pytorch_model.bin (incomplete download).")
        return
    if rev is None:
        print(f"{WARN} {line}")
        print("       revision cannot be proven from the directory (not an HF snapshot "
              "layout and no jina_model_revision sidecar).")
        return
    if pin:
        a, b = rev.lower(), pin.lower()
        if a == b or a.startswith(b) or b.startswith(a):
            print(f"{OK} {line}\n       revision {rev}  (matches the pin)")
        else:
            _fail(f"{line}\n         revision {rev}  !=  pin {pin}. Re-fetch the pinned commit.")
    else:
        print(f"{OK} {line}\n       revision {rev}")


# --------------------------------------------------------------------------
def check_index(settings) -> None:
    print("\n[3] Jina FAISS index + parquet")
    from src.services.assets import (
        BACKEND_ARTIFACT_NAMES,
        build_asset_store,
        cloud_enabled,
        get_artifact_cache,
        get_manifest,
    )

    names = list(BACKEND_ARTIFACT_NAMES["jina_clip_v2"])
    cache = get_artifact_cache(settings)
    print(f"       cache: {cache.root}")
    current = cache.get_current()

    manifest = None
    if cloud_enabled(settings):
        try:
            manifest = get_manifest(build_asset_store(settings))
        except Exception:  # noqa: BLE001
            manifest = None

    if manifest is not None:
        arts = [a for a in manifest.artifacts if a.name in set(names)]
        missing_from_manifest = [n for n in names if n not in {a.name for a in manifest.artifacts}]
        if missing_from_manifest:
            _fail(f"manifest {manifest.version} does not declare {missing_from_manifest} "
                  "-- broken publish; nothing to sync.")
            return
        verified = cache.is_version_verified(manifest.version, arts)
        rows = []
        for a in arts:
            s = cache.slot(manifest.version, a.name, expected_sha=a.sha256, expected_size=a.size)
            mark = "verified" if s.verified else ("present, UNVERIFIED" if s.present else "MISSING")
            rows.append((a.name, s.size if s.present else a.size, mark))
        total = sum(r[1] for r in rows)
        for n, sz, mark in rows:
            print(f"         {n:<20} {_human(sz):>10}  {mark}")
        if verified and current == manifest.version:
            print(f"{OK} version {manifest.version}: all {len(names)} artifacts verified "
                  f"({_human(total)}); this is the current version.")
        elif verified:
            print(f"{WARN} version {manifest.version} is verified but not marked current "
                  f"(current={current!r}). A restart / re-sync will promote it.")
        else:
            _fail(f"version {manifest.version} is not fully synced. Start the app (autosync) "
                  "or POST /settings/cloud/sync, or `python -m launcher`.")
        return

    # No manifest (cloud off or unreachable): fall back to what's on disk.
    local = settings.jina_faiss_index_path
    if local and Path(local).is_file():
        print(f"{OK} local JINA_FAISS_INDEX_PATH exists: {local}")
        for label, val in (
            ("JINA_GLOBAL_IDS_PATH", settings.jina_global_ids_path),
            ("JINA_INDEX_META_PATH", settings.jina_index_meta_path),
        ):
            if not (val and Path(val).is_file()):
                _fail(f"{label} is not set / missing -- needed alongside the local index.")
        return
    if not current:
        _fail("no synced index and cloud assets are off/unreachable. Either enable "
              "CLOUD_ASSETS_ENABLED + AZURE_STORAGE_CONNECTION_STRING, or point "
              "JINA_FAISS_INDEX_PATH / JINA_GLOBAL_IDS_PATH / JINA_INDEX_META_PATH at local files.")
        return
    if cache.is_version_complete(current, names):
        total = sum(cache.slot(current, n).size for n in names if cache.slot(current, n).present)
        print(f"{WARN} version {current}: all {len(names)} files present ({_human(total)}) "
              "but the manifest is unreachable, so checksums were not re-verified.")
    else:
        _fail(f"version {current} is incomplete on disk and the manifest is unreachable.")


# --------------------------------------------------------------------------
def main() -> int:
    from dotenv import load_dotenv

    load_dotenv(BACKEND_ROOT / ".env")
    from src.config.settings import get_settings

    settings = get_settings()
    print("Jina CLIP v2 setup check")
    print("========================")
    print(f"RETRIEVAL_BACKEND={settings.retrieval_backend!r}  "
          f"CLOUD_ASSETS_ENABLED={bool(settings.cloud_assets_enabled)}  "
          f"JINA_DEVICE={settings.jina_device!r}")

    check_torch(settings.jina_device)
    check_model(settings)
    check_index(settings)

    print("\n" + "=" * 24)
    if _problems == 0:
        print("RESULT: READY -- Jina CLIP v2 can serve searches on this machine.")
        return 0
    print(f"RESULT: NOT READY -- {_problems} item(s) above marked [MISS]. Fixes are inline.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
