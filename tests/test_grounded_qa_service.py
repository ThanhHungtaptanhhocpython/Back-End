from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.services.grounded_qa_service import _candidate_from_evidence, grounded_video_qa


def _settings(**overrides):
    values = {
        "qa_retrieval_pool": 6,
        "qa_text_evidence_top_k": 4,
        "qa_evidence_window_seconds": 15.0,
        "qa_max_frames": 3,
        "qa_per_video_limit": 2,
        "qa_vlm_enabled": True,
        "qa_min_confidence": 0.55,
        "qa_max_tokens": 300,
        "agent_vlm_enabled": True,
        "agent_vlm_model": "test/model",
        "agent_vlm_timeout_seconds": 1.0,
        "agent_vlm_image_max_side": 512,
        "agent_vlm_max_retries": 0,
        "agent_vlm_retry_backoff_seconds": 0.0,
        "openrouter_api_key": "test-key",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _frame(vector_id=10, video_id="L21_V001", timestamp=12.0):
    return {
        "vector_id": vector_id,
        "faiss_id": vector_id,
        "video_id": video_id,
        "frame_name": f"{vector_id:06d}.webp",
        "frame_path": f"L21_a/{video_id}/{vector_id:06d}.webp",
        "timestamp": timestamp,
        "score": 0.8,
    }


def _run(payload, *, confidence_threshold=0.55):
    retriever = MagicMock()
    retriever.search_visual.return_value = [_frame()]
    with (
        patch("src.services.grounded_qa_service.get_settings", return_value=_settings(qa_min_confidence=confidence_threshold)),
        patch("src.services.grounded_qa_service._get_retriever", return_value=retriever),
        patch("src.services.grounded_qa_service.Translation") as translator,
        patch("src.services.grounded_qa_service._collect_text_evidence", return_value={"ocr": [], "asr": []}),
        patch("src.services.grounded_qa_service.resolve_keyframe_path", return_value=Path("frame.webp")),
        patch("src.services.grounded_qa_service._image_to_data_url", return_value="data:image/webp;base64,AA=="),
        patch("src.services.grounded_qa_service._request_answer", return_value=payload),
    ):
        translator.return_value.return_value = "a person in the scene"
        return grounded_video_qa("Trong cảnh có gì?", 5)


def test_grounded_qa_returns_answer_and_supporting_frame():
    frames, summary = _run({
        "status": "answered",
        "answer": "Có một người trong cảnh.",
        "confidence": 0.86,
        "reason": "Frame f1 cho thấy một người.",
        "supporting_frame_ids": ["f1"],
        "used_ocr_evidence": False,
        "used_asr_evidence": False,
    })

    assert summary["status"] == "answered"
    assert summary["evaluated_frames"] == 1
    assert frames[0]["qa_supporting"] is True
    assert frames[0]["answer"] == "Có một người trong cảnh."


def test_grounded_qa_rejects_low_confidence_answer():
    frames, summary = _run({
        "status": "answered",
        "answer": "Có thể là một người.",
        "confidence": 0.4,
        "reason": "Hình ảnh không rõ.",
        "supporting_frame_ids": ["f1"],
        "used_ocr_evidence": False,
        "used_asr_evidence": False,
    })

    assert summary["status"] == "uncertain"
    assert summary["confidence"] == 0.0
    assert not frames[0]["qa_supporting"]


def test_grounded_qa_rejects_unknown_supporting_frame_id():
    _, summary = _run({
        "status": "answered",
        "answer": "Một người.",
        "confidence": 0.9,
        "reason": "Visible evidence.",
        "supporting_frame_ids": ["f99"],
        "used_ocr_evidence": False,
        "used_asr_evidence": False,
    })

    assert summary["status"] == "uncertain"
    assert any("unexpected supporting frame ids" in error for error in summary["errors"])


def test_timestamp_evidence_outside_window_is_discarded():
    retriever = MagicMock()
    retriever.get_nearest_frame.return_value = {**_frame(timestamp=50.0), "timestamp_delta": 30.0}
    row = {"video_id": "L21_V001", "timestamp": 20.0, "ocr_text": "sample", "_score": 2.0}

    candidate = _candidate_from_evidence(retriever, row, "ocr", max_timestamp_delta=15.0)

    assert candidate is None
