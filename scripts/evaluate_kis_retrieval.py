"""Visual-KIS retrieval accuracy harness for ``/users/singletextsearch``.

This is the missing A/B rig for the Jina CLIP v2 query path: ``evaluate_qa_golden``
only ever hits ``/users/qnasearch``, so the plain text->image ranker had no
accuracy measurement at all. Here we fire each ground-truth query straight at
``/users/singletextsearch`` and score the returned items with the existing
``src.evaluation.qa_golden`` rank helpers.

Cases come from two places, both already in the repo:

* ``benchmarks/qa_golden/*.jsonl`` -- 41 cases with per-frame ``evidence``
  (``video_id`` + ``timestamp`` + ``tolerance_seconds``). ``_item_matches_evidence``
  matches those against Jina's fine-keyframe rows by video+timestamp, so the
  older ``.webp``-named ground truth still scores.
* ``benchmarks/qa_stress/*.jsonl`` -- entries carrying ``expected_video_id``
  (video-level ground truth only).

Typical A/B use::

    # current default (JINA_QUERY_TASK="" -> no query-instruction prefix)
    python -X utf8 scripts/evaluate_kis_retrieval.py --top-k 20

    # then set JINA_QUERY_TASK="retrieval.query" in #/settings, restart, and:
    python -X utf8 scripts/evaluate_kis_retrieval.py --top-k 20

Compare ``retrieval_recall_at_k`` / ``video_recall_at_k`` between the two runs.
The 2026-09-04 run made "" the default (equal-or-better on every metric).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.settings import get_settings  # noqa: E402
from src.evaluation.qa_golden import (  # noqa: E402
    _first_evidence_rank,
    _first_video_rank,
    load_golden_cases,
)

JsonDict = dict[str, Any]

GOLDEN_DIR = PROJECT_ROOT / "benchmarks" / "qa_golden"
STRESS_DIR = PROJECT_ROOT / "benchmarks" / "qa_stress"
RUNS_DIR = PROJECT_ROOT / "benchmarks" / "kis_runs"
DEFAULT_RECALL_K = (1, 5, 10, 20)


# ---------------------------------------------------------------------------
# Case loading
# ---------------------------------------------------------------------------
def _golden_cases() -> list[JsonDict]:
    cases: list[JsonDict] = []
    for case in load_golden_cases(GOLDEN_DIR):
        evidence = [f for f in case.get("evidence", []) if isinstance(f, dict) and f.get("video_id")]
        if not case.get("question") or not evidence:
            continue
        cases.append(
            {
                "id": str(case.get("id")),
                "source": case.get("_dataset_file", "qa_golden"),
                "question": str(case.get("question")),
                "evidence": evidence,
                "has_frame_gt": any(f.get("timestamp") is not None or f.get("frame_id") for f in evidence),
            }
        )
    return cases


def _stress_cases() -> list[JsonDict]:
    cases: list[JsonDict] = []
    if not STRESS_DIR.is_dir():
        return cases
    for path in sorted(STRESS_DIR.glob("*.jsonl")):
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            video_id = row.get("expected_video_id")
            question = row.get("question")
            if not video_id or not question:
                continue  # p2_queries.jsonl rows have no ground truth -> skipped
            cases.append(
                {
                    "id": str(row.get("id") or f"{path.stem}:{line_no}"),
                    "source": path.name,
                    "question": str(question),
                    "evidence": [{"video_id": str(video_id)}],
                    "has_frame_gt": False,
                }
            )
    return cases


# ---------------------------------------------------------------------------
# API call
# ---------------------------------------------------------------------------
def _post_search(url: str, query: str, top_k: int, timeout: float) -> tuple[list[JsonDict], float]:
    body = json.dumps({"query": query, "topk": top_k}).encode("utf-8")
    request = urllib.request.Request(
        url, data=body, method="POST",
        headers={"Content-Type": "application/json; charset=utf-8"},
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:400]
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Cannot reach {url}: {exc.reason}") from exc
    elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
    data = payload.get("data") if isinstance(payload.get("data"), dict) else {}
    items = data.get("items") if isinstance(data.get("items"), list) else []
    return [it for it in items if isinstance(it, dict)], elapsed_ms


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def _aggregate(rows: Sequence[JsonDict], recall_k: Sequence[int]) -> JsonDict:
    total = len(rows)
    frame_rows = [r for r in rows if r["has_frame_gt"]]
    out: JsonDict = {
        "cases": total,
        "cases_with_frame_ground_truth": len(frame_rows),
        "request_failures": sum(1 for r in rows if r.get("error")),
    }
    for k in recall_k:
        if frame_rows:
            out[f"retrieval_recall_at_{k}"] = round(
                sum((r.get("evidence_rank") or 1e9) <= k for r in frame_rows) / len(frame_rows), 4
            )
        out[f"video_recall_at_{k}"] = round(
            sum((r.get("video_rank") or 1e9) <= k for r in rows) / total, 4
        ) if total else None
    ranks = [r["evidence_rank"] for r in frame_rows if r.get("evidence_rank")]
    out["mean_reciprocal_rank_frame"] = (
        round(sum(1.0 / r for r in ranks) / len(frame_rows), 4) if frame_rows else None
    )
    vranks = [r["video_rank"] for r in rows if r.get("video_rank")]
    out["mean_reciprocal_rank_video"] = (
        round(sum(1.0 / r for r in vranks) / total, 4) if total else None
    )
    return out


def _run(args: argparse.Namespace) -> int:
    settings = get_settings()
    cases = _golden_cases()
    if not args.golden_only:
        cases += _stress_cases()
    if args.limit:
        cases = cases[: args.limit]
    if not cases:
        print("No cases found.", file=sys.stderr)
        return 2

    url = args.base_url.rstrip("/") + "/users/singletextsearch"
    per_case: list[JsonDict] = []
    for index, case in enumerate(cases, 1):
        row: JsonDict = {
            "id": case["id"], "source": case["source"],
            "has_frame_gt": case["has_frame_gt"],
        }
        try:
            items, latency = _post_search(url, case["question"], args.top_k, args.timeout)
            row["latency_ms"] = latency
            row["evidence_rank"] = _first_evidence_rank(items, case["evidence"], args.tolerance)
            row["video_rank"] = _first_video_rank(items, case["evidence"])
            print(
                f"[{index}/{len(cases)}] {case['id']}: "
                f"frame_rank={row['evidence_rank']} video_rank={row['video_rank']}"
            )
        except Exception as exc:  # noqa: BLE001
            row["error"] = str(exc)
            print(f"[{index}/{len(cases)}] {case['id']}: ERROR {exc}")
            if args.fail_fast:
                per_case.append(row)
                break
        per_case.append(row)

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "api_url": url,
        "top_k": args.top_k,
        "tolerance_seconds": args.tolerance,
        "jina_query_task": settings.jina_query_task,
        "retrieval_backend": settings.retrieval_backend,
        "aggregate": _aggregate(per_case, DEFAULT_RECALL_K),
        "per_case": per_case,
    }

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    task_slug = (settings.jina_query_task or "none").replace(".", "_") or "none"
    out_path = args.output and Path(args.output) or RUNS_DIR / f"kis_report-{task_slug}-{stamp}.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(report["aggregate"], ensure_ascii=False, indent=2))
    print(f"Report: {out_path.resolve()}")
    print(f"JINA_QUERY_TASK = {settings.jina_query_task!r}")
    return 2 if any(r.get("error") for r in per_case) else 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default="http://127.0.0.1:3000")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--tolerance", type=float, default=2.0,
                        help="Same-video timestamp tolerance (s) for a frame-level hit.")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--limit", type=int, help="Only run the first N cases.")
    parser.add_argument("--golden-only", action="store_true",
                        help="Skip benchmarks/qa_stress (video-level ground truth only).")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--output", help="Report path (default benchmarks/kis_runs/...).")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    if args.top_k <= 0:
        print("--top-k must be positive", file=sys.stderr)
        return 2
    return _run(args)


if __name__ == "__main__":
    raise SystemExit(main())
