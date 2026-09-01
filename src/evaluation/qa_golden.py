"""Validation and metrics for the versioned video Q&A golden datasets."""

from __future__ import annotations

import json
import math
import re
import statistics
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

JsonDict = dict[str, Any]

ALLOWED_ANSWER_TYPES = {
    "action",
    "asr",
    "color",
    "count",
    "location",
    "object",
    "ocr",
    "other",
    "person",
    "spatial",
    "temporal",
    "yes_no",
}
ALLOWED_LANGUAGES = {"en", "vi"}
ALLOWED_STATUSES = {"answered", "uncertain"}
DEFAULT_RETRIEVAL_K = (1, 5, 10, 100)

NUMBER_WORDS = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "khong": "0",
    "mot": "1",
    "hai": "2",
    "ba": "3",
    "bon": "4",
    "nam": "5",
    "sau": "6",
    "bay": "7",
    "tam": "8",
    "chin": "9",
    "muoi": "10",
}

ENGLISH_OUTPUT_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "black",
    "blue",
    "bowl",
    "cable",
    "cannot",
    "dance",
    "evidence",
    "flatbread",
    "glass",
    "green",
    "insufficient",
    "is",
    "motorbike",
    "motorcycle",
    "object",
    "one",
    "orange",
    "person",
    "phone",
    "purple",
    "red",
    "sausage",
    "sausages",
    "smartphone",
    "spatula",
    "the",
    "this",
    "three",
    "truck",
    "two",
    "unknown",
    "vehicle",
    "vertical",
    "horizontal",
    "left",
    "right",
    "woman",
    "man",
    "white",
    "yellow",
}


class GoldenDatasetError(ValueError):
    """Raised when a golden dataset violates the documented schema."""


def _dataset_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise GoldenDatasetError(f"Golden dataset path does not exist: {path}")
    files = sorted(path.glob("*.jsonl"))
    if not files:
        raise GoldenDatasetError(f"No .jsonl golden datasets found under: {path}")
    return files


def load_golden_cases(path: str | Path) -> list[JsonDict]:
    """Load one JSONL golden file or every JSONL file in a directory.

    Args:
        path: Dataset file or directory containing versioned JSONL files.

    Returns:
        Cases in deterministic filename and line order.

    Raises:
        GoldenDatasetError: If JSON is invalid or a line is not an object.
    """
    cases: list[JsonDict] = []
    for file_path in _dataset_files(Path(path)):
        for line_number, line in enumerate(file_path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                case = json.loads(line)
            except json.JSONDecodeError as exc:
                raise GoldenDatasetError(f"{file_path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(case, dict):
                raise GoldenDatasetError(f"{file_path}:{line_number}: each line must be a JSON object")
            case = dict(case)
            case["_dataset_file"] = file_path.name
            case["_dataset_line"] = line_number
            cases.append(case)
    return cases


def _text(value: object) -> str:
    return " ".join(str(value or "").strip().split())


def _case_location(case: JsonDict) -> str:
    return f"{case.get('_dataset_file', 'dataset')}:{case.get('_dataset_line', '?')}"


def _obvious_english_output(value: object, answer_type: object) -> bool:
    if answer_type in {"count", "ocr"}:
        return False
    text = _text(value)
    if re.search(r"[à-ỹÀ-ỸđĐ]", text):
        return False
    tokens = set(re.findall(r"[A-Za-z]+", text.casefold()))
    return bool(tokens & ENGLISH_OUTPUT_WORDS)


def validate_golden_cases(
    cases: Sequence[JsonDict],
    keyframes_root: str | Path | None = None,
    check_files: bool = False,
) -> JsonDict:
    """Validate golden case schema and optionally verify evidence files.

    Args:
        cases: Loaded golden cases.
        keyframes_root: Root used to resolve relative evidence frame paths.
        check_files: Whether every evidence image must exist on disk.

    Returns:
        Dataset size and distribution summary.

    Raises:
        GoldenDatasetError: If one or more validation errors are found.
    """
    errors: list[str] = []
    seen_ids: set[str] = set()
    root = Path(keyframes_root).resolve() if keyframes_root else None
    if check_files and root is None:
        errors.append("keyframes_root is required when check_files=True")

    for case in cases:
        location = _case_location(case)
        case_id = _text(case.get("id"))
        if not case_id:
            errors.append(f"{location}: id is required")
        elif case_id in seen_ids:
            errors.append(f"{location}: duplicate id {case_id!r}")
        seen_ids.add(case_id)

        if case.get("schema_version") != "1.0":
            errors.append(f"{location}: schema_version must be '1.0'")
        if not _text(case.get("question")):
            errors.append(f"{location}: question is required")
        if case.get("language") not in ALLOWED_LANGUAGES:
            errors.append(f"{location}: language must be one of {sorted(ALLOWED_LANGUAGES)}")
        if case.get("answer_type") not in ALLOWED_ANSWER_TYPES:
            errors.append(f"{location}: invalid answer_type {case.get('answer_type')!r}")
        if case.get("expected_status") not in ALLOWED_STATUSES:
            errors.append(f"{location}: invalid expected_status {case.get('expected_status')!r}")

        answer = _text(case.get("answer"))
        if not answer:
            errors.append(f"{location}: answer is required")
        elif len(answer) > 100:
            errors.append(f"{location}: answer exceeds 100 characters")
        aliases = case.get("aliases", [])
        if not isinstance(aliases, list) or any(not _text(alias) for alias in aliases):
            errors.append(f"{location}: aliases must be a list of non-empty strings")
        elif any(len(_text(alias)) > 100 for alias in aliases):
            errors.append(f"{location}: aliases must not exceed 100 characters")
        if _obvious_english_output(answer, case.get("answer_type")):
            errors.append(f"{location}: natural-language answer must be Vietnamese")
        if isinstance(aliases, list) and any(
            _obvious_english_output(alias, case.get("answer_type"))
            for alias in aliases
        ):
            errors.append(f"{location}: natural-language aliases must be Vietnamese")

        evidence = case.get("evidence")
        if not isinstance(evidence, list):
            errors.append(f"{location}: evidence must be a list")
            evidence = []
        if case.get("expected_status") == "answered" and not evidence:
            errors.append(f"{location}: answered cases require at least one evidence frame")
        for index, frame in enumerate(evidence):
            if not isinstance(frame, dict):
                errors.append(f"{location}: evidence[{index}] must be an object")
                continue
            for field in ("split", "video_id", "frame_id", "frame_path", "timestamp"):
                if frame.get(field) in (None, ""):
                    errors.append(f"{location}: evidence[{index}].{field} is required")
            if frame.get("tolerance_seconds") is not None:
                try:
                    tolerance = float(frame["tolerance_seconds"])
                    if not math.isfinite(tolerance) or tolerance < 0:
                        raise ValueError
                except (TypeError, ValueError):
                    errors.append(
                        f"{location}: evidence[{index}].tolerance_seconds must be non-negative"
                    )
            if root is not None and frame.get("frame_path"):
                candidate = (root / str(frame["frame_path"])).resolve()
                try:
                    candidate.relative_to(root)
                except ValueError:
                    errors.append(f"{location}: evidence[{index}] escapes keyframes_root")
                    continue
                if check_files and not candidate.is_file():
                    errors.append(f"{location}: evidence file not found: {candidate}")

        review = case.get("review")
        if not isinstance(review, dict) or review.get("status") != "manually_verified":
            errors.append(f"{location}: review.status must be 'manually_verified'")

    if errors:
        preview = "\n".join(f"- {error}" for error in errors[:50])
        suffix = f"\n... and {len(errors) - 50} more" if len(errors) > 50 else ""
        raise GoldenDatasetError(f"Golden dataset validation failed:\n{preview}{suffix}")

    return {
        "cases": len(cases),
        "datasets": dict(sorted(Counter(str(case.get("_dataset_file")) for case in cases).items())),
        "answer_types": dict(sorted(Counter(str(case.get("answer_type")) for case in cases).items())),
        "languages": dict(sorted(Counter(str(case.get("language")) for case in cases).items())),
        "evidence_frames": sum(len(case.get("evidence") or []) for case in cases),
        "files_checked": bool(check_files),
    }


def _fold_diacritics(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value)
    return "".join(char for char in decomposed if not unicodedata.combining(char))


def normalize_answer(value: object, relaxed: bool = False) -> str:
    """Normalize an answer for strict or accent-insensitive comparison."""
    text = unicodedata.normalize("NFKC", _text(value)).casefold()
    if relaxed:
        text = _fold_diacritics(text)
    text = re.sub(r"[^\w]+", " ", text, flags=re.UNICODE)
    tokens = text.split()
    if relaxed:
        tokens = [NUMBER_WORDS.get(token, token) for token in tokens]
    return " ".join(tokens)


def _token_f1(prediction: str, target: str) -> float:
    predicted_tokens = prediction.split()
    target_tokens = target.split()
    if not predicted_tokens or not target_tokens:
        return float(predicted_tokens == target_tokens)
    overlap = sum((Counter(predicted_tokens) & Counter(target_tokens)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted_tokens)
    recall = overlap / len(target_tokens)
    return 2 * precision * recall / (precision + recall)


def score_answer(prediction: object, case: JsonDict) -> JsonDict:
    """Score one generated answer against all accepted golden aliases."""
    accepted = [_text(case.get("answer")), *[_text(value) for value in case.get("aliases", [])]]
    strict_prediction = normalize_answer(prediction)
    relaxed_prediction = normalize_answer(prediction, relaxed=True)
    strict_targets = [normalize_answer(value) for value in accepted]
    relaxed_targets = [normalize_answer(value, relaxed=True) for value in accepted]
    strict_exact = bool(strict_prediction and strict_prediction in strict_targets)
    relaxed_exact = bool(relaxed_prediction and relaxed_prediction in relaxed_targets)
    token_f1 = max((_token_f1(relaxed_prediction, target) for target in relaxed_targets), default=0.0)
    number_match = False
    if case.get("answer_type") == "count":
        predicted_numbers = re.findall(r"\d+(?:\.\d+)?", relaxed_prediction)
        target_numbers = {number for target in relaxed_targets for number in re.findall(r"\d+(?:\.\d+)?", target)}
        number_match = bool(predicted_numbers and predicted_numbers[0] in target_numbers)
    return {
        "strict_exact": strict_exact,
        "relaxed_exact": relaxed_exact,
        "number_match": number_match,
        "token_f1": round(token_f1, 6),
        "correct": bool(relaxed_exact or number_match or token_f1 >= 0.8),
    }


def _vietnamese_output_compliant(answer: str, case: JsonDict) -> bool:
    if not answer:
        return False
    if case.get("answer_type") in {"count", "ocr"}:
        return True
    if re.search(r"[à-ỹÀ-ỸđĐ]", answer):
        return True
    return not _obvious_english_output(answer, case.get("answer_type"))


def _first_value(item: JsonDict, *keys: str) -> object:
    backend = item.get("backend") if isinstance(item.get("backend"), dict) else {}
    for key in keys:
        value = item.get(key)
        if value not in (None, ""):
            return value
        value = backend.get(key)
        if value not in (None, ""):
            return value
    return None


def _frame_number(value: object) -> str:
    text = Path(_text(value)).stem
    digits = re.findall(r"\d+", text)
    return (digits[-1].lstrip("0") or "0") if digits else text.casefold()


def _normal_path(value: object) -> str:
    return _text(value).replace("\\", "/").lstrip("./").casefold()


def _item_matches_evidence(item: JsonDict, evidence: JsonDict, tolerance_seconds: float) -> bool:
    item_path = _normal_path(_first_value(item, "frame_path", "image_path"))
    evidence_path = _normal_path(evidence.get("frame_path"))
    if item_path and evidence_path and (item_path == evidence_path or item_path.endswith("/" + evidence_path)):
        return True
    item_video = _text(_first_value(item, "video_id", "video_key", "videoKey", "video_name")).casefold()
    evidence_video = _text(evidence.get("video_id")).casefold()
    if not item_video or item_video != evidence_video:
        return False
    item_frame = _frame_number(
        _first_value(item, "submission_frame_id", "frame_id", "frame_key", "frameKey", "frame_name")
    )
    evidence_frame = _frame_number(evidence.get("frame_id"))
    if item_frame and item_frame == evidence_frame:
        return True
    try:
        item_time = float(_first_value(item, "timestamp", "timestamp_s"))
        evidence_time = float(evidence.get("timestamp"))
    except (TypeError, ValueError):
        return False
    try:
        case_tolerance = float(evidence.get("tolerance_seconds") or 0.0)
    except (TypeError, ValueError):
        case_tolerance = 0.0
    effective_tolerance = max(tolerance_seconds, case_tolerance)
    return math.isfinite(item_time) and abs(item_time - evidence_time) <= effective_tolerance


def _first_evidence_rank(
    items: Sequence[JsonDict],
    evidence: Sequence[JsonDict],
    tolerance_seconds: float,
) -> int | None:
    for rank, item in enumerate(items, 1):
        if any(_item_matches_evidence(item, frame, tolerance_seconds) for frame in evidence):
            return rank
    return None


def _first_video_rank(items: Sequence[JsonDict], evidence: Sequence[JsonDict]) -> int | None:
    videos = {_text(frame.get("video_id")).casefold() for frame in evidence}
    for rank, item in enumerate(items, 1):
        video = _text(_first_value(item, "video_id", "video_key", "videoKey", "video_name")).casefold()
        if video and video in videos:
            return rank
    return None


def _percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))
    return round(ordered[index], 3)


def _aggregate(per_case: Sequence[JsonDict], retrieval_k: Sequence[int]) -> JsonDict:
    total = len(per_case)
    if total == 0:
        return {"cases": 0}

    def rate(key: str) -> float:
        return round(sum(bool(case.get(key)) for case in per_case) / total, 6)

    token_scores = [float(case.get("token_f1") or 0.0) for case in per_case]
    confidence_pairs = [
        (float(case["confidence"]), float(case.get("confidence_target") or 0.0))
        for case in per_case
        if isinstance(case.get("confidence"), (int, float))
    ]
    latencies = [float(case["latency_ms"]) for case in per_case if isinstance(case.get("latency_ms"), (int, float))]
    output: JsonDict = {
        "cases": total,
        "prediction_coverage": rate("has_prediction"),
        "status_accuracy": rate("status_correct"),
        "answer_strict_exact": rate("strict_exact"),
        "answer_relaxed_exact": rate("relaxed_exact"),
        "answer_accuracy": rate("answer_correct"),
        "answer_language_compliance": rate("answer_language_compliant"),
        "mean_token_f1": round(statistics.fmean(token_scores), 6),
        "format_compliance": rate("format_compliant"),
        "supporting_evidence_accuracy": rate("supporting_evidence_hit"),
        "video_recall_at_1": round(sum((case.get("video_rank") or math.inf) <= 1 for case in per_case) / total, 6),
    }
    for k in retrieval_k:
        output[f"retrieval_recall_at_{k}"] = round(
            sum((case.get("evidence_rank") or math.inf) <= k for case in per_case) / total,
            6,
        )
        output[f"video_recall_at_{k}"] = round(
            sum((case.get("video_rank") or math.inf) <= k for case in per_case) / total,
            6,
        )
    if confidence_pairs:
        output["confidence_brier"] = round(
            statistics.fmean((confidence - correct) ** 2 for confidence, correct in confidence_pairs),
            6,
        )
    else:
        output["confidence_brier"] = None
    output["latency_ms_p50"] = _percentile(latencies, 0.50)
    output["latency_ms_p95"] = _percentile(latencies, 0.95)
    return output


def evaluate_predictions(
    cases: Sequence[JsonDict],
    predictions: Iterable[JsonDict],
    retrieval_k: Sequence[int] = DEFAULT_RETRIEVAL_K,
    timestamp_tolerance_seconds: float = 2.0,
) -> JsonDict:
    """Evaluate answer quality, evidence retrieval, formatting, and calibration.

    Args:
        cases: Validated golden cases.
        predictions: Records keyed by the golden ``id``.
        retrieval_k: Recall cutoffs to report.
        timestamp_tolerance_seconds: Same-video timestamp tolerance for evidence matching.

    Returns:
        Aggregate metrics, sliced metrics, and per-case diagnostics.
    """
    prediction_by_id = {_text(record.get("id")): record for record in predictions if _text(record.get("id"))}
    per_case: list[JsonDict] = []
    for case in cases:
        case_id = _text(case.get("id"))
        prediction = prediction_by_id.get(case_id, {})
        status = _text(prediction.get("status")).lower() or "missing"
        answer = _text(prediction.get("answer"))
        answer_metrics = score_answer(answer, case)
        raw_items = prediction.get("items")
        items = [item for item in raw_items if isinstance(item, dict)] if isinstance(raw_items, list) else []
        evidence = [frame for frame in case.get("evidence", []) if isinstance(frame, dict)]
        evidence_rank = _first_evidence_rank(items, evidence, timestamp_tolerance_seconds)
        video_rank = _first_video_rank(items, evidence)
        supporting_items = [item for item in items if bool(item.get("qa_supporting"))]
        supporting_rank = _first_evidence_rank(supporting_items, evidence, timestamp_tolerance_seconds)
        format_compliant = bool(answer and len(answer) <= 100 and "\n" not in answer and "\r" not in answer)
        answer_language_compliant = _vietnamese_output_compliant(answer, case)
        expected_status = _text(case.get("expected_status")).lower()
        failures: list[str] = []
        if not prediction:
            failures.append("missing_prediction")
        if status != expected_status:
            failures.append("wrong_status")
        if expected_status == "answered" and not answer_metrics["correct"]:
            failures.append("wrong_answer")
        if expected_status == "answered" and evidence_rank is None:
            failures.append("retrieval_miss")
        if not format_compliant:
            failures.append("format_violation")
        if not answer_language_compliant:
            failures.append("answer_language_violation")
        if status == "answered" and supporting_rank is None:
            failures.append("supporting_evidence_miss")
        detail: JsonDict = {
            "id": case_id,
            "dataset": case.get("_dataset_file"),
            "language": case.get("language"),
            "answer_type": case.get("answer_type"),
            "difficulty": case.get("difficulty"),
            "expected_answer": case.get("answer"),
            "predicted_answer": answer,
            "expected_status": expected_status,
            "predicted_status": status,
            "has_prediction": bool(prediction),
            "status_correct": status == expected_status,
            "answer_correct": (
                bool(answer_metrics["correct"])
                if expected_status == "answered"
                else status == "uncertain"
            ),
            "confidence_target": float(expected_status == "answered" and bool(answer_metrics["correct"])),
            "format_compliant": format_compliant,
            "answer_language_compliant": answer_language_compliant,
            "evidence_rank": evidence_rank,
            "video_rank": video_rank,
            "supporting_evidence_hit": (
                supporting_rank is not None if expected_status == "answered" else status == "uncertain"
            ),
            "confidence": prediction.get("confidence"),
            "latency_ms": prediction.get("latency_ms"),
            "failures": failures,
            **answer_metrics,
        }
        per_case.append(detail)

    by_type = {
        name: _aggregate([case for case in per_case if case.get("answer_type") == name], retrieval_k)
        for name in sorted({str(case.get("answer_type")) for case in per_case})
    }
    by_language = {
        name: _aggregate([case for case in per_case if case.get("language") == name], retrieval_k)
        for name in sorted({str(case.get("language")) for case in per_case})
    }
    by_dataset = {
        name: _aggregate([case for case in per_case if case.get("dataset") == name], retrieval_k)
        for name in sorted({str(case.get("dataset")) for case in per_case})
    }
    return {
        "schema_version": "1.0",
        "aggregate": _aggregate(per_case, retrieval_k),
        "by_answer_type": by_type,
        "by_language": by_language,
        "by_dataset": by_dataset,
        "failure_counts": dict(sorted(Counter(failure for case in per_case for failure in case["failures"]).items())),
        "per_case": per_case,
    }
