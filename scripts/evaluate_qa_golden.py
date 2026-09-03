"""Validate and evaluate the grounded video Q&A golden datasets.

Examples:
    python scripts/evaluate_qa_golden.py validate --check-files
    python scripts/evaluate_qa_golden.py run --limit 3
    python scripts/evaluate_qa_golden.py score predictions.jsonl
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
    GoldenDatasetError,
    evaluate_predictions,
    load_golden_cases,
    validate_golden_cases,
)

JsonDict = dict[str, Any]
DEFAULT_DATASET = PROJECT_ROOT / "benchmarks" / "qa_golden"
DEFAULT_RUNS_DIR = PROJECT_ROOT / "benchmarks" / "qa_runs"


def _json_dump(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def _load_json_records(path: Path) -> list[JsonDict]:
    text = path.read_text(encoding="utf-8-sig").strip()
    if not text:
        return []
    if path.suffix.lower() == ".json":
        value = json.loads(text)
        if isinstance(value, dict) and isinstance(value.get("predictions"), list):
            value = value["predictions"]
        if not isinstance(value, list):
            raise ValueError("Prediction JSON must be a list or contain a predictions list.")
        return [row for row in value if isinstance(row, dict)]
    records: list[JsonDict] = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: prediction must be a JSON object")
        records.append(value)
    return records


def _write_jsonl(path: Path, records: Sequence[JsonDict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(json.dumps(row, ensure_ascii=False) for row in records)
    path.write_text(content + ("\n" if content else ""), encoding="utf-8")


def _write_report(path: Path, report: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_json_dump(report) + "\n", encoding="utf-8")


def _post_question(url: str, question: str, top_k: int, timeout: float) -> JsonDict:
    body = json.dumps({"query": question, "topk": top_k}).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json; charset=utf-8"},
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
        data = payload.get("data") if isinstance(payload.get("data"), dict) else {}
        meta = data.get("meta") if isinstance(data.get("meta"), dict) else {}
        items = data.get("items") if isinstance(data.get("items"), list) else []
        return {
            "status": str(meta.get("status") or "missing"),
            "answer": str(meta.get("answer") or payload.get("message") or ""),
            "confidence": meta.get("confidence"),
            "latency_ms": elapsed_ms,
            "items": items,
            "meta": meta,
        }
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Cannot reach Q&A API at {url}: {exc.reason}") from exc


def _select_cases(
    cases: Sequence[JsonDict],
    selected_ids: Sequence[str],
    limit: int | None,
) -> list[JsonDict]:
    wanted = set(selected_ids)
    selected = [case for case in cases if not wanted or str(case.get("id")) in wanted]
    missing = wanted - {str(case.get("id")) for case in selected}
    if missing:
        raise ValueError(f"Unknown golden case ids: {', '.join(sorted(missing))}")
    return selected[:limit] if limit is not None else selected


def _default_output_paths() -> tuple[Path, Path]:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return (
        DEFAULT_RUNS_DIR / f"qa_predictions-{stamp}.jsonl",
        DEFAULT_RUNS_DIR / f"qa_report-{stamp}.json",
    )


def _command_validate(args: argparse.Namespace) -> int:
    cases = load_golden_cases(args.dataset)
    settings = get_settings()
    summary = validate_golden_cases(
        cases,
        keyframes_root=args.keyframes_root or settings.keyframes_root,
        check_files=args.check_files,
    )
    print(_json_dump(summary))
    return 0


def _command_score(args: argparse.Namespace) -> int:
    cases = load_golden_cases(args.dataset)
    validate_golden_cases(cases)
    predictions = _load_json_records(Path(args.predictions))
    report = evaluate_predictions(cases, predictions)
    if args.report:
        _write_report(Path(args.report), report)
        print(f"Report: {Path(args.report).resolve()}")
    print(_json_dump(report["aggregate"]))
    print(_json_dump({"failure_counts": report["failure_counts"]}))
    return 0


def _command_run(args: argparse.Namespace) -> int:
    cases = load_golden_cases(args.dataset)
    settings = get_settings()
    validate_golden_cases(
        cases,
        keyframes_root=args.keyframes_root or settings.keyframes_root,
        check_files=args.check_files,
    )
    selected = _select_cases(cases, args.id, args.limit)
    if not selected:
        raise ValueError("No golden cases selected.")

    default_predictions, default_report = _default_output_paths()
    predictions_path = Path(args.output) if args.output else default_predictions
    report_path = Path(args.report) if args.report else default_report
    url = args.base_url.rstrip("/") + "/" + args.endpoint.strip("/")
    predictions: list[JsonDict] = []
    failures = 0
    for index, case in enumerate(selected, 1):
        case_id = str(case.get("id"))
        print(f"[{index}/{len(selected)}] {case_id}: {case.get('question')}")
        try:
            result = _post_question(url, str(case.get("question")), args.top_k, args.timeout)
            result.update({"id": case_id, "question": case.get("question")})
            print(
                f"  -> {result['status']} ({result.get('confidence')}): "
                f"{result.get('answer') or '<empty>'}"
            )
        except Exception as exc:
            failures += 1
            result = {
                "id": case_id,
                "question": case.get("question"),
                "status": "error",
                "answer": "",
                "confidence": None,
                "latency_ms": None,
                "items": [],
                "error": str(exc),
            }
            print(f"  -> ERROR: {exc}")
            if args.fail_fast:
                predictions.append(result)
                break
        predictions.append(result)
        _write_jsonl(predictions_path, predictions)

    report = evaluate_predictions(selected, predictions)
    report["run"] = {
        "api_url": url,
        "selected_cases": len(selected),
        "completed_predictions": len(predictions),
        "request_failures": failures,
        "top_k": args.top_k,
    }
    _write_jsonl(predictions_path, predictions)
    _write_report(report_path, report)
    print(f"Predictions: {predictions_path.resolve()}")
    print(f"Report: {report_path.resolve()}")
    print(_json_dump(report["aggregate"]))
    return 2 if failures else 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.set_defaults(dataset=str(DEFAULT_DATASET))
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="Validate dataset schema and evidence files.")
    validate.add_argument("--dataset", default=str(DEFAULT_DATASET))
    validate.add_argument("--keyframes-root")
    validate.add_argument("--check-files", action="store_true")
    validate.set_defaults(handler=_command_validate)

    score = subparsers.add_parser("score", help="Score an existing prediction JSON/JSONL file.")
    score.add_argument("predictions")
    score.add_argument("--dataset", default=str(DEFAULT_DATASET))
    score.add_argument("--report")
    score.set_defaults(handler=_command_score)

    run = subparsers.add_parser("run", help="Run golden questions against the local Q&A API.")
    run.add_argument("--dataset", default=str(DEFAULT_DATASET))
    run.add_argument("--base-url", default="http://127.0.0.1:3000")
    run.add_argument("--endpoint", default="/users/qnasearch")
    run.add_argument("--top-k", type=int, default=100)
    run.add_argument("--timeout", type=float, default=300.0)
    run.add_argument("--limit", type=int)
    run.add_argument("--id", action="append", default=[])
    run.add_argument("--output")
    run.add_argument("--report")
    run.add_argument("--keyframes-root")
    run.add_argument("--check-files", action="store_true")
    run.add_argument("--fail-fast", action="store_true")
    run.set_defaults(handler=_command_run)
    return parser


def main() -> int:
    """Run the selected golden-dataset command."""
    parser = _build_parser()
    args = parser.parse_args()
    if getattr(args, "top_k", 1) <= 0:
        parser.error("--top-k must be positive")
    if getattr(args, "limit", 1) is not None and getattr(args, "limit", 1) <= 0:
        parser.error("--limit must be positive")
    try:
        return int(args.handler(args))
    except (GoldenDatasetError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
