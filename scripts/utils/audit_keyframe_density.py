#!/usr/bin/env python
"""Audit keyframe density from map-keyframes CSV files.

The competition tasks often ask for exact first/last moments. This script
flags videos whose extracted keyframes are too sparse for that kind of query.
"""

from __future__ import annotations

import argparse
import csv
import statistics
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Gap:
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float

    @property
    def seconds(self) -> float:
        return self.end_time - self.start_time


@dataclass(frozen=True)
class VideoAudit:
    video_id: str
    keyframes: int
    duration_s: float
    avg_gap_s: float
    median_gap_s: float
    p95_gap_s: float
    max_gap: Gap | None
    large_gaps: list[Gap]


def backend_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_float(value: str, fallback: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def parse_int(value: str, fallback: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return fallback


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    rank = (len(ordered) - 1) * pct
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    weight = rank - lo
    return ordered[lo] * (1 - weight) + ordered[hi] * weight


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return sorted(rows, key=lambda row: parse_float(row.get("pts_time", "0")))


def audit_video(path: Path, gap_threshold_s: float) -> VideoAudit:
    rows = read_rows(path)
    video_id = path.stem
    if not rows:
        return VideoAudit(video_id, 0, 0.0, 0.0, 0.0, 0.0, None, [])

    gaps: list[Gap] = []
    for prev, curr in zip(rows, rows[1:]):
        gaps.append(
            Gap(
                start_frame=parse_int(prev.get("frame_idx", "0")),
                end_frame=parse_int(curr.get("frame_idx", "0")),
                start_time=parse_float(prev.get("pts_time", "0")),
                end_time=parse_float(curr.get("pts_time", "0")),
            )
        )

    gap_values = [gap.seconds for gap in gaps if gap.seconds >= 0]
    large_gaps = [gap for gap in gaps if gap.seconds >= gap_threshold_s]
    max_gap = max(gaps, key=lambda gap: gap.seconds, default=None)
    duration_s = parse_float(rows[-1].get("pts_time", "0")) - parse_float(rows[0].get("pts_time", "0"))

    return VideoAudit(
        video_id=video_id,
        keyframes=len(rows),
        duration_s=max(0.0, duration_s),
        avg_gap_s=statistics.fmean(gap_values) if gap_values else 0.0,
        median_gap_s=statistics.median(gap_values) if gap_values else 0.0,
        p95_gap_s=percentile(gap_values, 0.95),
        max_gap=max_gap,
        large_gaps=large_gaps,
    )


def risk_label(audit: VideoAudit, gap_threshold_s: float) -> str:
    if audit.keyframes < 2:
        return "NO_DATA"
    if audit.p95_gap_s >= gap_threshold_s * 2 or len(audit.large_gaps) >= 10:
        return "HIGH"
    if audit.p95_gap_s >= gap_threshold_s or len(audit.large_gaps) >= 3:
        return "MEDIUM"
    return "OK"


def format_time(seconds: float) -> str:
    seconds = max(0.0, seconds)
    minutes = int(seconds // 60)
    rem = seconds - minutes * 60
    return f"{minutes:02d}:{rem:05.2f}"


def iter_map_files(map_dir: Path, videos: list[str]) -> list[Path]:
    if videos:
        return [map_dir / f"{video_id}.csv" for video_id in videos]
    return sorted(map_dir.glob("*.csv"))


def build_parser() -> argparse.ArgumentParser:
    root = backend_root()
    parser = argparse.ArgumentParser(description="Audit keyframe timeline density.")
    parser.add_argument("--map-dir", type=Path, default=root / "src" / "dict" / "map-keyframes")
    parser.add_argument("--videos", nargs="*", default=[], help="Optional video IDs, e.g. L24_V024 L25_V041.")
    parser.add_argument("--gap-threshold", type=float, default=5.0, help="Seconds considered too sparse for Q&A/TRAKE.")
    parser.add_argument("--top-gaps", type=int, default=5, help="How many largest gaps to print per video.")
    parser.add_argument("--sort-by", choices=("risk", "p95", "max", "video"), default="risk")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    files = iter_map_files(args.map_dir, args.videos)
    missing = [path.stem for path in files if not path.exists()]
    audits = [audit_video(path, args.gap_threshold) for path in files if path.exists()]

    risk_order = {"HIGH": 0, "MEDIUM": 1, "OK": 2, "NO_DATA": 3}
    if args.sort_by == "risk":
        audits.sort(key=lambda audit: (risk_order[risk_label(audit, args.gap_threshold)], -audit.p95_gap_s, audit.video_id))
    elif args.sort_by == "p95":
        audits.sort(key=lambda audit: (-audit.p95_gap_s, audit.video_id))
    elif args.sort_by == "max":
        audits.sort(key=lambda audit: (-(audit.max_gap.seconds if audit.max_gap else 0.0), audit.video_id))
    else:
        audits.sort(key=lambda audit: audit.video_id)

    print("video_id,keyframes,duration,avg_gap,median_gap,p95_gap,max_gap,large_gaps,risk")
    for audit in audits:
        max_gap_s = audit.max_gap.seconds if audit.max_gap else 0.0
        label = risk_label(audit, args.gap_threshold)
        print(
            f"{audit.video_id},{audit.keyframes},{format_time(audit.duration_s)},"
            f"{audit.avg_gap_s:.2f},{audit.median_gap_s:.2f},{audit.p95_gap_s:.2f},"
            f"{max_gap_s:.2f},{len(audit.large_gaps)},{label}"
        )
        for gap in sorted(audit.large_gaps, key=lambda item: item.seconds, reverse=True)[: args.top_gaps]:
            print(
                f"  GAP {format_time(gap.start_time)}->{format_time(gap.end_time)} "
                f"({gap.seconds:.2f}s), frames {gap.start_frame}->{gap.end_frame}"
            )

    if missing:
        print("missing:", ", ".join(missing))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
