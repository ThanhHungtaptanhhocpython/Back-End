import os
import sys
import re
import io
import csv
import zipfile
import argparse
import logging
from pathlib import Path
from tqdm import tqdm

# Ensure backend root is in sys.path
BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("BatchSubmit")

def normalize_video_name(val):
    val = str(val or "unknown-video").strip()
    return re.sub(r"\.(mp4|mov|avi|mkv|webm)$", "", val, flags=re.I)

def format_video_id(item):
    raw_video = item.get("video_name") or item.get("video_id") or item.get("video_key") or "unknown"
    video = normalize_video_name(raw_video)
    folder = str(item.get("folder_key") or item.get("split") or item.get("namespace") or "").strip()
    if re.match(r"^V\d+", video, re.I):
        if re.match(r"^L\d+", folder, re.I):
            return normalize_video_name(f"{folder}_{video}")
        m = re.search(r"l(\d+)", folder, re.I)
        if m:
            return f"L{m.group(1)}_{video}"
    return video

def format_frame_id(item):
    raw = (
        item.get("global_frame_id")
        or item.get("frame_idx")
        or item.get("frame_id")
        or item.get("frame_key")
        or item.get("id")
        or 0
    )
    try:
        clean_num = re.sub(r"[^\d-]", "", str(raw))
        return int(clean_num) if clean_num else 0
    except Exception:
        return 0

def parse_query_files(input_dir: Path):
    """Scan all .txt query files from a directory and sort them by query number."""
    query_files = []
    if not input_dir.exists():
        logger.error(f"Directory {input_dir} not found!")
        return []

    for f in os.listdir(input_dir):
        if f.endswith(".txt"):
            full_p = input_dir / f
            with open(full_p, "r", encoding="utf-8", errors="ignore") as file:
                content = file.read().strip()
                if content:
                    # Deduce query name and type
                    q_stem = full_p.stem # e.g. query-p1-1-kis
                    m_type = re.search(r"-(kis|qa|trake)$", q_stem, re.IGNORECASE)
                    q_type = m_type.group(1).lower() if m_type else "kis"
                    
                    # Sort key based on number in filename
                    m_num = re.search(r"-(\d+)(?:-[a-z]+)?$", q_stem)
                    q_order = int(m_num.group(1)) if m_num else 999
                    
                    query_files.append({
                        "filename": f"{q_stem}.csv",
                        "stem": q_stem,
                        "order": q_order,
                        "type": q_type,
                        "query_text": content
                    })
                    
    return sorted(query_files, key=lambda x: (x["order"], x["stem"]))

def run_batch_generation(input_dir: str, output_zip: str, topk: int = 100):
    in_path = Path(input_dir)
    queries = parse_query_files(in_path)
    
    if not queries:
        logger.error(f"No valid query .txt files found in {input_dir}!")
        return
        
    logger.info(f"Loaded {len(queries)} queries from '{input_dir}'.")
    logger.info(f"Initializing BEiT-3 / Visual Search Engine...")
    
    from src.services.user_service import getImageDataSingleTextSearch
    
    csv_results = {}
    
    for q_item in tqdm(queries, desc="Generating Submissions"):
        fname = q_item["filename"]
        q_text = q_item["query_text"]
        q_type = q_item["type"]
        
        # Execute search for topk
        try:
            results = getImageDataSingleTextSearch(q_text, topk)
        except Exception as e:
            logger.error(f"Search failed for '{q_item['stem']}': {e}")
            results = []
            
        # Build CSV lines (Max topk items)
        lines = []
        for r in results[:topk]:
            v_name = format_video_id(r)
            f_idx = format_frame_id(r)
            if q_type == "qa":
                ans = r.get("answer", "")
                lines.append(f'{v_name},{f_idx},"{ans}"')
            else:
                lines.append(f"{v_name},{f_idx}")
                
        csv_results[fname] = "\n".join(lines)
        
    # Write to root of Zip archive
    out_p = Path(output_zip)
    logger.info(f"Writing {len(csv_results)} query CSVs directly to root of '{out_p.name}'...")
    
    with zipfile.ZipFile(out_p, "w", zipfile.ZIP_DEFLATED) as z:
        for fname, csv_content in csv_results.items():
            z.writestr(fname, csv_content)
            
    logger.info(f"SUCCESS! Created '{out_p.resolve()}' containing {len(csv_results)} queries (Top-{topk} each).")
    logger.info("This ZIP archive has all files placed at the root, ready for BTC Evaluation!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch Query Submitter for AIC Competition")
    parser.add_argument("--dir", default="THUNGHIEM-bo-de-thi", help="Folder containing query .txt files")
    parser.add_argument("--output", default="batch_submission.zip", help="Output submission zip filename")
    parser.add_argument("--topk", type=int, default=100, help="Number of candidate keyframes per query (Default: 100)")
    
    args = parser.parse_args()
    run_batch_generation(args.dir, args.output, args.topk)
