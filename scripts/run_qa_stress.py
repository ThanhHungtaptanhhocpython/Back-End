"""Run question-only Q&A stress cases and preserve diagnostics as JSONL."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from evaluate_qa_golden import _post_question, _write_jsonl

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = PROJECT_ROOT / "benchmarks" / "qa_stress" / "p2_queries.jsonl"
DEFAULT_RUNS_DIR = PROJECT_ROOT / "benchmarks" / "qa_runs"


def _load_cases(path: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict) or not value.get("id") or not value.get("question"):
            raise ValueError(f"Invalid stress case at {path}:{line_number}")
        cases.append(value)
    return cases


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--base-url", default="http://127.0.0.1:3000")
    parser.add_argument("--endpoint", default="/users/qnasearch")
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--id", action="append", default=[])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    selected_ids = set(args.id)
    cases = [case for case in _load_cases(args.dataset) if not selected_ids or case["id"] in selected_ids]
    if not cases:
        raise ValueError("No stress cases selected")
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output = args.output or DEFAULT_RUNS_DIR / f"p2-stress-{stamp}.jsonl"
    url = args.base_url.rstrip("/") + "/" + args.endpoint.strip("/")
    predictions: list[dict[str, Any]] = []
    for index, case in enumerate(cases, 1):
        print(f"[{index}/{len(cases)}] {case['id']}: {case['question']}", flush=True)
        try:
            result = _post_question(url, str(case["question"]), args.top_k, args.timeout)
            result.update({"id": case["id"], "question": case["question"]})
            print(
                f"  -> {result['status']} ({result.get('confidence')}): "
                f"{result.get('answer') or '<empty>'}",
                flush=True,
            )
        except Exception as exc:
            result = {"id": case["id"], "question": case["question"], "error": str(exc)}
            print(f"  !! {exc}", flush=True)
        predictions.append(result)
        _write_jsonl(output, predictions)
    print(f"Predictions: {output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
