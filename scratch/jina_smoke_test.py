"""Local smoke test for the Jina CLIP v2 retrieval backend.

Two modes:

  # 1. Build a *tiny* local index from a few videos pulled from Azure
  #    (~tens of MB, not the 2.65 GB full index) so you can validate the
  #    model + FAISS + result plumbing on your machine quickly.
  python scratch/jina_smoke_test.py build-mini --videos 3 --out ./jina_mini

  # then point .env at it:
  #   RETRIEVAL_BACKEND=jina_clip_v2
  #   JINA_FAISS_INDEX_PATH=<abs>/jina_mini/jina_faiss.index
  #   JINA_GLOBAL_IDS_PATH=<abs>/jina_mini/jina_global_ids.parquet

  # 2. Run a query through JinaRetriever using whatever JINA_*_PATH .env points
  #    at (the mini index above, or the full synced one).
  python scratch/jina_smoke_test.py query "một người đàn ông đang lái xe máy"
  python scratch/jina_smoke_test.py query --image /path/to/frame.jpg

Needs AZURE_STORAGE_CONNECTION_STRING in .env for build-mini, and a local
`jinaai/jina-clip-v2` snapshot (or JINA_LOCAL_FILES_ONLY=false for the first run)
for query.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(BACKEND_ROOT / ".env")


def build_mini(n_videos: int, out_dir: Path) -> None:
    import numpy as np
    from azure.storage.blob import BlobServiceClient

    from src.config.settings import get_settings

    conn = (get_settings().azure_storage_connection_string or "").strip()
    if not conn:
        raise SystemExit("AZURE_STORAGE_CONNECTION_STRING not set in .env")

    RUN = "fine_keyframes_jina_clip_v2_1024d_v2"
    emb = BlobServiceClient.from_connection_string(conn).get_container_client("embeddings")

    # discover a few videos under one namespace
    jina_prefix = f"{RUN}/jina/L21_a/"
    npy_blobs = [b.name for b in emb.list_blobs(name_starts_with=jina_prefix) if b.name.endswith(".npy")]
    npy_blobs = sorted(npy_blobs)[:n_videos]
    if not npy_blobs:
        raise SystemExit(f"no per-video .npy found under {jina_prefix}")

    emb_root = out_dir / "_src" / "embeddings" / "L21_a"
    rec_root = out_dir / "_src" / "records" / "L21_a"
    emb_root.mkdir(parents=True, exist_ok=True)
    rec_root.mkdir(parents=True, exist_ok=True)

    for npy_key in npy_blobs:
        vid = Path(npy_key).stem
        print(f"  downloading {vid} …", flush=True)
        (emb_root / f"{vid}.npy").write_bytes(emb.get_blob_client(npy_key).download_blob().readall())
        rec_key = f"{RUN}/records/L21_a/{vid}.json"
        (rec_root / f"{vid}.json").write_bytes(emb.get_blob_client(rec_key).download_blob().readall())

    # build with the real build script
    sys.argv = [
        "build_jina_index.py",
        "--embeddings-root", str(out_dir / "_src" / "embeddings"),
        "--records-root", str(out_dir / "_src" / "records"),
        "--out-dir", str(out_dir),
        "--model-revision", "smoke-test-unpinned",
        "--embedding-run", RUN + "_mini",
    ]
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "bji", str(BACKEND_ROOT / "scripts" / "cloud" / "build_jina_index.py")
    )
    bji = importlib.util.module_from_spec(spec)
    sys.modules["bji"] = bji
    spec.loader.exec_module(bji)
    bji.main(sys.argv[1:])

    print("\nSet in .env:")
    print("  RETRIEVAL_BACKEND=jina_clip_v2")
    print(f"  JINA_FAISS_INDEX_PATH={out_dir / 'jina_faiss.index'}")
    print(f"  JINA_GLOBAL_IDS_PATH={out_dir / 'jina_global_ids.parquet'}")


def run_query(text: str | None, image: str | None, top_k: int) -> None:
    from src.services.retrieval_backend import get_active_retriever

    r = get_active_retriever()
    print(f"backend_id = {getattr(r, 'backend_id', '?')}\n")
    if image:
        results = r.search_by_image(image, top_k=top_k)
    else:
        results = r.search_visual(text, top_k=top_k)
    for row in results:
        print(
            f"#{row['rank']:>2}  score={row['score']:.4f}  {row['video_id']:<12} "
            f"asset_key={row.get('asset_key') or row.get('frame_path')}  ts={row.get('timestamp')}"
        )
    if not results:
        print("(no results)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build-mini")
    b.add_argument("--videos", type=int, default=3)
    b.add_argument("--out", type=Path, default=Path("./jina_mini"))

    q = sub.add_parser("query")
    q.add_argument("text", nargs="?", default=None)
    q.add_argument("--image", default=None)
    q.add_argument("--top-k", type=int, default=10)

    args = p.parse_args()
    if args.cmd == "build-mini":
        build_mini(args.videos, args.out)
    else:
        if not args.text and not args.image:
            raise SystemExit("give a text query or --image PATH")
        run_query(args.text, args.image, args.top_k)


if __name__ == "__main__":
    main()
