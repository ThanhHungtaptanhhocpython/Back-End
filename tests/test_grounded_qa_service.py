import re
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from PIL import Image

from src.services.grounded_qa_service import (
    _candidate_from_evidence,
    _detail_image_data_urls,
    _diversify_evidence_groups,
    _question_plan,
    _relevant_text_evidence,
    _select_detail_frame_ids,
    _split_visual_events,
    grounded_video_qa,
)


def _settings(**overrides):
    values = {
        "qa_retrieval_pool": 6,
        "qa_visual_query_limit": 3,
        "qa_text_evidence_top_k": 4,
        "qa_evidence_window_seconds": 15.0,
        "qa_event_window_seconds": 8.0,
        "qa_max_evidence_groups": 3,
        "qa_context_frames_per_group": 3,
        "qa_max_frames": 3,
        "qa_per_video_limit": 2,
        "qa_vlm_enabled": True,
        "qa_detail_pass_enabled": True,
        "qa_detail_model": "test/detail-model",
        "qa_detail_max_frames": 2,
        "qa_detail_grid_size": 3,
        "qa_detail_image_max_side": 900,
        "qa_verify_enabled": True,
        "qa_min_confidence": 0.55,
        "qa_return_best_guess": True,
        "qa_uncertain_confidence_cap": 0.49,
        "qa_answer_max_chars": 100,
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


def _run(payload, *, confidence_threshold=0.55, verification=None):
    payload = {**payload, "answer_language": "vi"}
    verification = verification or {
        "verified": True,
        "canonical_answer": payload.get("answer", ""),
        "confidence": payload.get("confidence", 0.0),
        "reason": payload.get("reason", "Đã kiểm chứng bằng chứng."),
        "supporting_frame_ids": payload.get("supporting_frame_ids", []),
        "answer_language": "vi",
    }
    retriever = MagicMock()
    retriever.search_visual.return_value = [_frame()]
    with (
        patch(
            "src.services.grounded_qa_service.get_settings",
            return_value=_settings(qa_min_confidence=confidence_threshold),
        ),
        patch("src.services.grounded_qa_service._get_retriever", return_value=retriever),
        patch("src.services.grounded_qa_service.Translation") as translator,
        patch("src.services.grounded_qa_service._collect_text_evidence", return_value={"ocr": [], "asr": []}),
        patch("src.services.grounded_qa_service.resolve_keyframe_path", return_value=Path("frame.webp")),
        patch("src.services.grounded_qa_service._image_to_data_url", return_value="data:image/webp;base64,AA=="),
        patch("src.services.grounded_qa_service._request_answer", return_value=payload),
        patch(
            "src.services.grounded_qa_service._request_verification",
            return_value=verification,
        ),
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
    assert summary["answer"] == "Có thể là một người."
    assert summary["confidence"] == 0.4
    assert summary["answer_mode"] == "best_guess"
    assert frames[0]["qa_supporting"]


def test_verifier_rejection_keeps_most_likely_vietnamese_candidate():
    frames, summary = _run(
        {
            "status": "answered",
            "answer": "Ô tô",
            "confidence": 0.85,
            "reason": "Bằng chứng ban đầu.",
            "supporting_frame_ids": ["f1"],
            "used_ocr_evidence": False,
            "used_asr_evidence": False,
        },
        verification={
            "verified": False,
            "canonical_answer": "Xe máy",
            "confidence": 0.35,
            "reason": "Khung hình nghiêng về xe máy nhưng chưa đủ rõ.",
            "supporting_frame_ids": ["f1"],
            "answer_language": "vi",
        },
    )

    assert summary["status"] == "uncertain"
    assert summary["answer"] == "Xe máy"
    assert summary["confidence"] == 0.35
    assert summary["answer_mode"] == "best_guess"
    assert frames[0]["qa_supporting"]


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


def test_question_plan_classifies_count_and_generates_multiple_visual_queries():
    with patch("src.services.grounded_qa_service.Translation") as translator:
        translator.return_value.return_value = "How many red cars are visible in the scene?"
        plan = _question_plan("Có bao nhiêu xe màu đỏ trong cảnh?")

    assert plan["answer_type"] == "count"
    assert "số nguyên" in plan["expected_answer_format"]
    assert plan["visual_queries"][0].lower().startswith("red cars")
    assert len(plan["visual_queries"]) >= 2


def test_question_plan_enables_ocr_for_map_questions():
    with patch("src.services.grounded_qa_service.Translation") as translator:
        translator.return_value.return_value = "earthquake distribution map with a color legend"
        plan = _question_plan("Bản đồ phân bố động đất có chú giải màu")

    assert plan["ocr_query"]


def test_question_plan_classifies_object_spatial_and_logo_ocr():
    with patch("src.services.grounded_qa_service.Translation") as translator:
        translator.return_value.return_value = "visual query"
        object_plan = _question_plan("What kind of container holds the crystals?")
        spatial_plan = _question_plan("Điện thoại được cầm theo hướng dọc hay ngang?")
        ocr_plan = _question_plan("Logo kênh truyền hình nào xuất hiện trong ảnh?")

    assert object_plan["answer_type"] == "object"
    assert spatial_plan["answer_type"] == "spatial"
    assert ocr_plan["answer_type"] == "ocr"
    assert ocr_plan["ocr_query"]


def test_question_plan_removes_interrogative_words_from_visual_focus():
    translated = (
        "In the night scene, which two-wheeled vehicle has its rear fairing "
        "removed and its red shock absorbers exposed?"
    )
    with patch("src.services.grounded_qa_service.Translation") as translator:
        translator.return_value.return_value = translated
        plan = _question_plan("Phương tiện nào có giảm xóc đỏ?")

    focus = plan["visual_queries"][0].lower()
    assert "which" not in focus
    assert " has " not in f" {focus} "
    assert "red shock absorbers" in focus
    assert any("motorcycle" in query.lower() for query in plan["visual_queries"])


def test_temporal_question_is_split_into_visual_events_and_scale_variants():
    translated = (
        "The image shows a fish being placed on a scale, followed by a scene of "
        "another fish of the same type being held by a person by the tail. "
        "What is the final number displayed on the scale?"
    )
    with patch("src.services.grounded_qa_service.Translation") as translator:
        translator.return_value.return_value = translated
        plan = _question_plan(
            "Một con cá được đặt lên cân, sau đó một con khác bị cầm đuôi. "
            "Con số cuối cùng là bao nhiêu?",
            visual_query_limit=8,
        )

    assert plan["answer_type"] == "count"
    assert plan["needs_temporal_context"] is True
    assert len(plan["event_queries"]) == 2
    assert any("plastic container" in query.lower() and "digital scale" in query.lower() for query in plan["visual_queries"])
    assert any("holding" in query.lower() and "tail" in query.lower() for query in plan["visual_queries"])
    assert any("small shark" in query.lower() for query in plan["visual_queries"])
    assert {"e1", "e2"}.issubset(set(plan["visual_query_event_ids"]))
    shark_scale_index = next(index for index, query in enumerate(plan["visual_queries"]) if "small shark" in query.lower() and "scale" in query.lower())
    assert plan["visual_query_priorities"][shark_scale_index] >= 3


def test_split_visual_events_drops_final_interrogative():
    events = _split_visual_events(
        "The video shows a fish on a scale, followed by another fish held by its tail. "
        "What number is displayed?"
    )

    assert events == ["a fish on a scale", "another fish held by its tail"]


def test_detail_images_include_full_frame_and_zoom_grid(tmp_path):
    path = tmp_path / "frame.png"
    Image.new("RGB", (120, 90), color=(25, 80, 140)).save(path)

    parts = _detail_image_data_urls(path, max_side=500, grid_size=3)

    assert len(parts) == 10
    assert parts[0][0] == "toàn cảnh"
    assert all(url.startswith("data:image/jpeg;base64,") for _, url in parts)


def test_count_question_uses_detail_pass_for_tiny_display():
    first_pass = {
        "status": "uncertain",
        "answer": "Không thể xác định",
        "confidence": 0.1,
        "reason": "Màn hình cân quá nhỏ trong toàn cảnh.",
        "supporting_frame_ids": ["f1"],
        "used_ocr_evidence": False,
        "used_asr_evidence": False,
        "answer_language": "vi",
    }
    detail_pass = {
        "status": "uncertain",
        "answer": "5",
        "confidence": 0.48,
        "reason": "Ô phóng to cho thấy chữ số 5 trên màn hình cân.",
        "supporting_frame_ids": ["f1"],
        "used_ocr_evidence": True,
        "used_asr_evidence": False,
        "answer_language": "vi",
    }
    retriever = MagicMock()
    retriever.search_visual.return_value = [_frame()]
    request = MagicMock(side_effect=[first_pass, detail_pass])
    with (
        patch("src.services.grounded_qa_service.get_settings", return_value=_settings()),
        patch("src.services.grounded_qa_service._get_retriever", return_value=retriever),
        patch("src.services.grounded_qa_service.Translation") as translator,
        patch("src.services.grounded_qa_service._collect_text_evidence", return_value={"ocr": [], "asr": []}),
        patch("src.services.grounded_qa_service.resolve_keyframe_path", return_value=Path("frame.webp")),
        patch("src.services.grounded_qa_service._image_to_data_url", return_value="data:image/webp;base64,AA=="),
        patch(
            "src.services.grounded_qa_service._detail_image_data_urls",
            return_value=[("toàn cảnh", "data:image/jpeg;base64,AA==")],
        ),
        patch("src.services.grounded_qa_service._request_answer", request),
        patch("src.services.grounded_qa_service._request_verification") as verify,
    ):
        translator.return_value.return_value = "a small shark on a digital scale, final display number"
        frames, summary = grounded_video_qa("Con số cuối cùng trên cân là bao nhiêu?", 5)

    assert request.call_count == 2
    verify.assert_not_called()
    assert summary["detail_pass"] == "refined"
    assert summary["answer"] == "5"
    assert summary["answer_mode"] == "best_guess"
    assert frames[0]["qa_supporting"] is True


def test_detail_pass_prefers_group_with_strongest_complete_event_sequence():
    selected = {
        "f1": {"qa_group_id": "wrong", "qa_video_event_worst_rank": 118},
        "f3": {"qa_group_id": "target", "qa_video_event_worst_rank": 15},
        "f13": {"qa_group_id": "wrong", "qa_video_event_worst_rank": 118},
        "f15": {"qa_group_id": "target", "qa_video_event_worst_rank": 15},
    }

    detail_ids = _select_detail_frame_ids(
        ["f1", "f3", "f13", "f15"],
        selected,
        limit=2,
        temporal_question=True,
    )

    assert detail_ids == ["f3", "f15"]


def test_diversified_groups_keep_specific_query_alternatives():
    groups = [
        {"id": "generic", "score": 5.0, "best_query_ranks": {0: 1}},
        {"id": "specific-1", "score": 1.0, "best_query_ranks": {1: 1}},
        {"id": "specific-2", "score": 0.9, "best_query_ranks": {1: 2}},
        {"id": "specific-3", "score": 0.8, "best_query_ranks": {1: 3}},
    ]

    selected = _diversify_evidence_groups(
        groups,
        ["fish", "small shark in a plastic container on a digital scale"],
        limit=4,
    )

    assert [group["id"] for group in selected[:3]] == ["specific-1", "specific-2", "specific-3"]


def test_diversified_groups_honor_expansion_priority_before_query_length():
    groups = [
        {"id": "long-base", "score": 4.0, "best_query_ranks": {0: 1}},
        {"id": "hypothesis", "score": 1.0, "best_query_ranks": {1: 1}},
    ]

    selected = _diversify_evidence_groups(
        groups,
        ["a very long generic base retrieval query", "short hypothesis"],
        limit=2,
        query_priorities=[0, 3],
    )

    assert selected[0]["id"] == "hypothesis"


def test_text_evidence_is_limited_to_attached_video_moment():
    evidence = {
        "ocr": [
            {"video_id": "L21_V001", "timestamp": 12.5, "ocr_text": "near"},
            {"video_id": "L21_V001", "timestamp": 90.0, "ocr_text": "far"},
            {"video_id": "L21_V002", "timestamp": 12.0, "ocr_text": "wrong video"},
            {"video_id": "L21_V001", "timestamp": 12.5, "ocr_text": "near"},
        ],
        "asr": [],
    }

    filtered = _relevant_text_evidence(
        evidence,
        [_frame(timestamp=12.0)],
        per_modality_limit=4,
        max_timestamp_delta=15.0,
    )

    assert [row["ocr_text"] for row in filtered["ocr"]] == ["near"]


def test_competition_answer_is_canonical_and_limited_to_100_characters():
    long_answer = "Câu trả lời: " + ("một câu trả lời rất dài " * 10)
    frames, summary = _run({
        "status": "answered",
        "answer": long_answer,
        "confidence": 0.9,
        "reason": "Visible evidence.",
        "supporting_frame_ids": ["f1"],
        "used_ocr_evidence": False,
        "used_asr_evidence": False,
    })

    assert summary["status"] == "answered"
    assert len(summary["answer"]) <= 100
    assert not summary["answer"].lower().startswith("câu trả lời:")
    assert frames[0]["answer"] == summary["answer"]


def test_english_natural_language_answer_is_rejected():
    _, summary = _run({
        "status": "answered",
        "answer": "A red motorcycle",
        "confidence": 0.9,
        "reason": "Visible evidence.",
        "supporting_frame_ids": ["f1"],
        "used_ocr_evidence": False,
        "used_asr_evidence": False,
    })

    assert summary["status"] == "uncertain"
    assert summary["answer_language"] == "vi"
    assert re.search(r"[à-ỹđ]", summary["answer"], re.IGNORECASE)
