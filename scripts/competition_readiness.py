from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.services.competition_readiness import run_readiness_audit  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run AIC HCM 2026 competition readiness checks.")
    parser.add_argument("--deep", action="store_true", help="Probe Elasticsearch and load the active retriever.")
    parser.add_argument("--query", default="", help="Optional query to execute when --deep is enabled.")
    parser.add_argument("--json", action="store_true", help="Print the full JSON report.")
    args = parser.parse_args()

    report = run_readiness_audit(deep=args.deep, query=args.query.strip() or None)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        summary = report["summary"]
        print(f"ready={report['ready']} pass={summary['pass']} warn={summary['warn']} fail={summary['fail']}")
        for item in report["checks"]:
            marker = {"pass": "OK", "warn": "WARN", "fail": "FAIL"}[item["status"]]
            print(f"[{marker}] {item['category']}/{item['name']}: {item['detail']}")
    return 0 if report["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
