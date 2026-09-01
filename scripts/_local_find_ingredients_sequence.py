from __future__ import annotations

from collections import defaultdict
import json

from src.services.beit3_retriever import get_beit3_retriever


QUERIES = [
    "diagonal camera view moving upward ending on a close-up of the first raw seafood ingredient for cooking",
    "top-down overhead close-up view of a second raw seafood ingredient for cooking",
    "colorful cooking ingredients arranged together, vegetables spices and food",
    "static wide shot showing all cooking ingredients arranged together on a table",
]
RECALL = 900
PER_VIDEO = 10


def video_id(row):
    return str(row.get("video_id") or row.get("video") or "")


def timestamp(row):
    for key in ("timestamp", "timestamp_s", "time", "pts_time"):
        if row.get(key) not in (None, ""):
            return float(row[key])
    return float(row.get("frame_id") or row.get("frame_idx") or 0)


def score(row):
    return float(row.get("score") or row.get("similarity") or row.get("distance") or 0)


retriever = get_beit3_retriever()
all_results = []
for q in QUERIES:
    rows = retriever.search_visual(q, top_k=RECALL)
    all_results.append(rows)

grouped = defaultdict(lambda: [[] for _ in QUERIES])
for event_idx, rows in enumerate(all_results):
    for rank, row in enumerate(rows, 1):
        item = dict(row)
        item["rank"] = rank
        grouped[video_id(item)][event_idx].append(item)

candidates = []
for vid, event_groups in grouped.items():
    if not vid or any(not group for group in event_groups):
        continue
    pools = [sorted(group, key=score, reverse=True)[:PER_VIDEO] for group in event_groups]
    beam = [(0.0, [])]
    for event_idx, pool in enumerate(pools):
        expanded = []
        for total, seq in beam:
            last_t = timestamp(seq[-1]) if seq else -1.0
            for row in pool:
                t = timestamp(row)
                if t + 0.25 < last_t:
                    continue
                expanded.append((total + score(row), seq + [row]))
        expanded.sort(key=lambda x: x[0], reverse=True)
        beam = expanded[:100]
        if not beam:
            break
    if not beam:
        continue
    for total, seq in beam[:3]:
        span = timestamp(seq[-1]) - timestamp(seq[0])
        if span < 0 or span > 90:
            continue
        adjusted = total - 0.0025 * span
        candidates.append((adjusted, total, span, vid, seq))

candidates.sort(key=lambda x: x[0], reverse=True)
output = []
seen = set()
for adjusted, total, span, vid, seq in candidates:
    if vid in seen:
        continue
    seen.add(vid)
    output.append({
        "video_id": vid,
        "adjusted": adjusted,
        "total": total,
        "span": span,
        "events": [{
            "rank": row["rank"],
            "score": score(row),
            "timestamp": timestamp(row),
            "frame_path": row.get("frame_path"),
            "frame_id": row.get("frame_id"),
            "vector_id": row.get("vector_id"),
        } for row in seq],
    })
    if len(output) >= 30:
        break

print(json.dumps(output, ensure_ascii=False, indent=2))
