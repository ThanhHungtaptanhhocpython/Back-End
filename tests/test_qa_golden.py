"""Regression tests for the grounded Q&A golden evaluation harness."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.evaluation.qa_golden import (
    GoldenDatasetError,
    evaluate_predictions,
    load_golden_cases,
    score_answer,
    validate_golden_cases,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _case(**overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "schema_version": "1.0",
        "id": "case-1",
        "question": "Phương tiện gì?",
        "language": "vi",
        "answer_type": "object",
        "expected_status": "answered",
        "answer": "xe máy",
        "aliases": ["mô tô"],
        "evidence": [
            {
                "split": "L21_a",
                "video_id": "L21_V001",
                "frame_id": "020406",
                "frame_path": "L21_a/L21_V001/020406.webp",
                "timestamp": 680.2,
            }
        ],
        "difficulty": "easy",
        "tags": ["visual"],
        "review": {
            "status": "manually_verified",
            "reviewer": "test",
            "date": "2026-08-30",
        },
        "_dataset_file": "test.jsonl",
        "_dataset_line": 1,
    }
    value.update(overrides)
    return value


class GoldenDatasetTests(unittest.TestCase):
    """Verify schema checks, answer normalization, and retrieval diagnostics."""

    def test_bundled_golden_dataset_is_valid_and_balanced(self) -> None:
        cases = load_golden_cases(PROJECT_ROOT / "benchmarks" / "qa_golden")
        summary = validate_golden_cases(cases)

        # 40 language-balanced base cases (20 vi / 20 en) plus one Vietnamese
        # competition regression case in competition_regressions_v1.jsonl.
        # Keep these numbers in lock-step with benchmarks/qa_golden/README.md.
        self.assertEqual(summary["cases"], 41)
        self.assertEqual(
            summary["datasets"],
            {
                "abstention_v1.jsonl": 6,
                "competition_regressions_v1.jsonl": 1,
                "count_ocr_v1.jsonl": 10,
                "temporal_multievent_v1.jsonl": 2,
                "visual_attributes_v1.jsonl": 10,
                "visual_core_v1.jsonl": 12,
            },
        )
        self.assertEqual(summary["languages"], {"en": 20, "vi": 21})
        self.assertEqual(summary["evidence_frames"], 48)

        # The only departure from a 20/20 language split is the single
        # Vietnamese competition regression case.
        regression_cases = [
            case for case in cases
            if case["_dataset_file"] == "competition_regressions_v1.jsonl"
        ]
        self.assertEqual(len(regression_cases), 1)
        self.assertEqual(regression_cases[0]["language"], "vi")
        self.assertEqual(regression_cases[0]["expected_status"], "answered")

    def test_relaxed_answer_accepts_missing_vietnamese_diacritics(self) -> None:
        metrics = score_answer("xe may", _case())

        self.assertFalse(metrics["strict_exact"])
        self.assertTrue(metrics["relaxed_exact"])
        self.assertTrue(metrics["correct"])

    def test_count_words_match_numeric_answer(self) -> None:
        case = _case(answer_type="count", answer="2", aliases=["hai"])
        metrics = score_answer("hai", case)

        self.assertTrue(metrics["relaxed_exact"])
        self.assertTrue(metrics["number_match"])

    def test_evaluator_separates_answer_and_retrieval_metrics(self) -> None:
        case = _case()
        prediction = {
            "id": "case-1",
            "status": "answered",
            "answer": "mô tô",
            "confidence": 0.8,
            "latency_ms": 125.0,
            "items": [
                {
                    "frame_path": "L21_a/L21_V001/020406.webp",
                    "video_id": "L21_V001",
                    "frame_id": "020406",
                    "qa_supporting": True,
                }
            ],
        }

        report = evaluate_predictions([case], [prediction])

        aggregate = report["aggregate"]
        self.assertEqual(aggregate["answer_accuracy"], 1.0)
        self.assertEqual(aggregate["retrieval_recall_at_1"], 1.0)
        self.assertEqual(aggregate["supporting_evidence_accuracy"], 1.0)
        self.assertEqual(report["failure_counts"], {})

    def test_uncertain_case_rewards_abstention_without_supporting_frame(self) -> None:
        case = _case(
            expected_status="uncertain",
            answer="Không đủ bằng chứng để xác định.",
            aliases=["Không xác định được"],
        )
        prediction = {
            "id": "case-1",
            "status": "uncertain",
            "answer": "Không đủ bằng chứng để xác định.",
            "confidence": 0.1,
            "items": [{"frame_path": "L21_a/L21_V001/020406.webp"}],
        }

        report = evaluate_predictions([case], [prediction])

        self.assertEqual(report["aggregate"]["status_accuracy"], 1.0)
        self.assertEqual(report["aggregate"]["answer_accuracy"], 1.0)
        self.assertEqual(report["aggregate"]["confidence_brier"], 0.01)
        self.assertNotIn("supporting_evidence_miss", report["failure_counts"])

    def test_evaluator_rejects_english_final_answer_language(self) -> None:
        case = _case(answer_type="color", answer="màu đỏ", aliases=["đỏ"])
        prediction = {
            "id": "case-1",
            "status": "answered",
            "answer": "red",
            "confidence": 0.9,
            "items": [],
        }

        report = evaluate_predictions([case], [prediction])

        self.assertEqual(report["aggregate"]["answer_language_compliance"], 0.0)
        self.assertEqual(report["failure_counts"]["answer_language_violation"], 1)

    def test_validator_rejects_frame_path_outside_keyframe_root(self) -> None:
        case = _case()
        case["evidence"] = [
            {
                "split": "L21_a",
                "video_id": "L21_V001",
                "frame_id": "020406",
                "frame_path": "../secret.webp",
                "timestamp": 680.2,
            }
        ]
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(GoldenDatasetError, "escapes keyframes_root"):
                validate_golden_cases([case], directory, check_files=False)


if __name__ == "__main__":
    unittest.main()
