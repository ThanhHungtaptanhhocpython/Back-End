"""Grounded retrieve-evidence-answer pipeline for video Q&A."""

from __future__ import annotations

import json
import logging
import math
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.config.settings import get_settings
from src.services.openrouter_vlm_verifier import _extract_json_object, _image_to_data_url, resolve_keyframe_path
from src.utils.nlp_processing import Translation

logger = logging.getLogger(__name__)

ANSWER_SCHEMA = {
    "name": "grounded_video_answer",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "status": {"type": "string", "enum": ["answered", "uncertain"]},
            "answer": {"type": "string"},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "reason": {"type": "string"},
            "supporting_frame_ids": {"type": "array", "items": {"type": "string"}},
            "used_ocr_evidence": {"type": "boolean"},
            "used_asr_evidence": {"type": "boolean"},
        },
        "required": [
            "status",
            "answer",
            "confidence",
            "reason",
            "supporting_frame_ids",
            "used_ocr_evidence",
            "used_asr_evidence",
        ],
    },
}

SYSTEM_PROMPT = """You answer questions about retrieved video evidence.
Use only the attached keyframes and OCR/ASR snippets. Never use outside knowledge to fill a missing fact.
Answer in the same language as the user's question.
If the evidence does not directly support a reliable answer, return status=uncertain and say that evidence is insufficient.
For counting, colors, identities, actions, spatial relations, and visible objects, rely on visible frames.
For spoken content, rely on ASR snippets. For written text, rely on OCR snippets.
Only cite frame ids that are attached. Return strict JSON only, no markdown.
"""

OCR_INTENT = re.compile(r"\b(?:ocr|text|written|read|sign|caption|subtitle|chu|chữ|ghi gì|viet gi|viết gì|biển|bien|bảng|bang)\b", re.IGNORECASE)
ASR_INTENT = re.compile(r"\b(?:asr|audio|speech|say|said|saying|hear|nghe|nói|noi|phát biểu|phat bieu)\b", re.IGNORECASE)


def _clean(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
        return parsed if math.isfinite(parsed) else default
    except (TypeError, ValueError):
        return default


def _identity(frame: Dict[str, Any]) -> str:
    for key in ("vector_id", "faiss_id", "frame_path", "global_frame_id", "frame_name"):
        if frame.get(key) not in (None, ""):
            return str(frame[key])
    return ""


def _question_plan(question: str) -> Dict[str, str]:
    visual_query = Translation()(question) or question
    return {
        "question": question,
        "visual_query": visual_query,
        "ocr_query": question if OCR_INTENT.search(question) else "",
        "asr_query": question if ASR_INTENT.search(question) else "",
    }


def _get_retriever() -> Any:
    from src.services.beit3_retriever import get_beit3_retriever

    return get_beit3_retriever()


def _collect_text_evidence(plan: Dict[str, str], top_k: int) -> Dict[str, List[Dict[str, Any]]]:
    evidence: Dict[str, List[Dict[str, Any]]] = {"ocr": [], "asr": []}
    if not plan["ocr_query"] and not plan["asr_query"]:
        return evidence
    try:
        from src.services.user_service import get_elastic_processor

        processor = get_elastic_processor()
        if plan["ocr_query"]:
            evidence["ocr"] = processor.search_ocr(plan["ocr_query"], topk=top_k)
        if plan["asr_query"]:
            evidence["asr"] = processor.search_asr(plan["asr_query"], topk=top_k)
    except Exception as exc:
        logger.warning("Grounded Q&A text evidence unavailable: %s", exc)
    return evidence


def _evidence_timestamp(row: Dict[str, Any], modality: str) -> float:
    if modality == "asr":
        if row.get("nearest_timestamp") is not None:
            return _float(row.get("nearest_timestamp"))
        start = _float(row.get("start_time"))
        end = _float(row.get("end_time"), start)
        return (start + end) / 2.0
    return _float(row.get("timestamp"))


def _evidence_text(row: Dict[str, Any], modality: str) -> str:
    return _clean(row.get("text") if modality == "asr" else row.get("ocr_text"))[:500]


def _candidate_from_evidence(
    retriever: Any,
    row: Dict[str, Any],
    modality: str,
    max_timestamp_delta: float,
) -> Dict[str, Any] | None:
    frame = None
    if modality == "asr" and row.get("nearest_faiss_id") is not None:
        frame = retriever.get_frame_by_vector_id(row.get("nearest_faiss_id"))
    if frame is None and row.get("video_id"):
        frame = retriever.get_nearest_frame(str(row.get("video_id")), _evidence_timestamp(row, modality))
    if frame is None:
        return None
    target_timestamp = _evidence_timestamp(row, modality)
    timestamp_delta = frame.get("timestamp_delta")
    if timestamp_delta is None and frame.get("timestamp") is not None:
        timestamp_delta = abs(_float(frame.get("timestamp")) - target_timestamp)
    if _float(timestamp_delta, float("inf")) > max_timestamp_delta:
        return None
    frame = dict(frame)
    frame.setdefault("score", 0.0)
    frame["qa_evidence_priority"] = max(_float(frame.get("qa_evidence_priority")), _float(row.get("_score"), 1.0))
    frame.setdefault("qa_text_evidence", []).append({
        "modality": modality,
        "text": _evidence_text(row, modality),
        "timestamp": _evidence_timestamp(row, modality),
        "video_id": row.get("video_id"),
    })
    return frame


def _merge_candidates(visual: List[Dict[str, Any]], evidence_frames: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []
    for rank, frame in enumerate(visual, 1):
        item = dict(frame)
        item["qa_visual_rank"] = rank
        item["qa_visual_score"] = _float(item.get("score"))
        key = _identity(item)
        if key and key not in merged:
            merged[key] = item
            order.append(key)
    for frame in evidence_frames:
        key = _identity(frame)
        if not key:
            continue
        if key not in merged:
            merged[key] = dict(frame)
            order.append(key)
            continue
        current = merged[key]
        current["qa_evidence_priority"] = max(
            _float(current.get("qa_evidence_priority")),
            _float(frame.get("qa_evidence_priority")),
        )
        current.setdefault("qa_text_evidence", []).extend(frame.get("qa_text_evidence") or [])
    return [merged[key] for key in order]


def _select_frames(candidates: List[Dict[str, Any]], limit: int, per_video_limit: int) -> List[Dict[str, Any]]:
    ranked = sorted(
        candidates,
        key=lambda frame: (
            bool(frame.get("qa_text_evidence")),
            _float(frame.get("qa_evidence_priority")),
            _float(frame.get("qa_visual_score", frame.get("score"))),
            -int(frame.get("qa_visual_rank") or 100000),
        ),
        reverse=True,
    )
    selected: List[Dict[str, Any]] = []
    counts: Dict[str, int] = {}
    for frame in ranked:
        video_id = str(frame.get("video_id") or "unknown")
        if counts.get(video_id, 0) >= per_video_limit:
            continue
        if resolve_keyframe_path(frame) is None:
            continue
        selected.append(frame)
        counts[video_id] = counts.get(video_id, 0) + 1
        if len(selected) >= limit:
            break
    return selected


def _request_answer(messages: List[Dict[str, Any]], model: str, timeout: float, max_tokens: int) -> Dict[str, Any]:
    settings = get_settings()
    body = {
        "model": model,
        "temperature": 0,
        "max_tokens": max_tokens,
        "messages": messages,
        "response_format": {"type": "json_schema", "json_schema": ANSWER_SCHEMA},
    }

    def send(payload: Dict[str, Any]) -> Dict[str, Any]:
        request = urllib.request.Request(
            settings.openrouter_base_url.rstrip("/") + "/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
            headers={
                "Authorization": f"Bearer {settings.openrouter_api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": settings.openrouter_site_url,
                "X-Title": settings.openrouter_app_name,
            },
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
        content = (((result.get("choices") or [{}])[0].get("message") or {}).get("content") or "")
        return _extract_json_object(content)

    try:
        return send(body)
    except urllib.error.HTTPError as exc:
        if exc.code not in {400, 404, 422}:
            raise
        fallback = dict(body)
        fallback.pop("response_format", None)
        return send(fallback)


def _normalise_answer(payload: Dict[str, Any], frame_ids: set[str]) -> Tuple[Dict[str, Any], List[str]]:
    errors: List[str] = []
    status = _clean(payload.get("status")).lower()
    answer = _clean(payload.get("answer"))[:1500]
    reason = _clean(payload.get("reason"))[:500]
    try:
        confidence = float(payload["confidence"])
    except (KeyError, TypeError, ValueError):
        confidence = -1.0
    supporting = payload.get("supporting_frame_ids")
    if status not in {"answered", "uncertain"}:
        errors.append("invalid status")
    if not answer or not reason:
        errors.append("answer/reason is required")
    if not 0.0 <= confidence <= 1.0:
        errors.append("confidence out of range")
    if not isinstance(supporting, list) or any(not isinstance(value, str) for value in supporting):
        errors.append("supporting_frame_ids must be strings")
        supporting = []
    unexpected = set(supporting) - frame_ids
    if unexpected:
        errors.append(f"unexpected supporting frame ids: {sorted(unexpected)}")
    for key in ("used_ocr_evidence", "used_asr_evidence"):
        if not isinstance(payload.get(key), bool):
            errors.append(f"{key} must be boolean")
    return {
        "status": status,
        "answer": answer,
        "confidence": max(0.0, min(1.0, confidence)),
        "reason": reason,
        "supporting_frame_ids": [value for value in supporting if value in frame_ids],
        "used_ocr_evidence": bool(payload.get("used_ocr_evidence")),
        "used_asr_evidence": bool(payload.get("used_asr_evidence")),
    }, errors


def _uncertain_answer(question: str, reason: str) -> Dict[str, Any]:
    has_vietnamese = bool(re.search(r"[à-ỹÀ-ỸđĐ]", question))
    answer = (
        "Không đủ bằng chứng đã truy xuất để trả lời chắc chắn."
        if has_vietnamese
        else "The retrieved evidence is insufficient for a reliable answer."
    )
    return {
        "status": "uncertain",
        "answer": answer,
        "confidence": 0.0,
        "reason": reason,
        "supporting_frame_ids": [],
        "used_ocr_evidence": False,
        "used_asr_evidence": False,
    }


def grounded_video_qa(question: str, top_k: int) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    settings = get_settings()
    plan = _question_plan(question)
    retriever = _get_retriever()
    pool_size = max(top_k, int(settings.qa_retrieval_pool))
    visual_frames = retriever.search_visual(plan["visual_query"], top_k=pool_size)
    text_evidence = _collect_text_evidence(plan, max(1, int(settings.qa_text_evidence_top_k)))
    evidence_frames: List[Dict[str, Any]] = []
    for modality in ("ocr", "asr"):
        for row in text_evidence[modality]:
            frame = _candidate_from_evidence(
                retriever,
                row,
                modality,
                max(0.0, float(settings.qa_evidence_window_seconds)),
            )
            if frame is not None:
                evidence_frames.append(frame)

    candidates = _merge_candidates(visual_frames, evidence_frames)
    selected = _select_frames(
        candidates,
        max(1, min(int(settings.qa_max_frames), 16)),
        max(1, int(settings.qa_per_video_limit)),
    )
    selected_ids = {f"f{index}": frame for index, frame in enumerate(selected, 1)}
    verdict = _uncertain_answer(question, "No grounded VLM answer was produced.")
    errors: List[str] = []
    retries = 0

    if settings.qa_vlm_enabled and settings.agent_vlm_enabled and settings.openrouter_api_key and selected:
        evidence_lines = []
        for modality in ("ocr", "asr"):
            for row in text_evidence[modality][: int(settings.qa_text_evidence_top_k)]:
                text = _evidence_text(row, modality)
                if text:
                    evidence_lines.append(
                        f"{modality.upper()} video={row.get('video_id')} time={_evidence_timestamp(row, modality):.3f}: {text}"
                    )
        prompt_lines = [
            f"Question: {question}",
            f"Visual retrieval query: {plan['visual_query']}",
            "Text evidence:",
            *(evidence_lines or ["No OCR/ASR evidence was retrieved."]),
            "Attached keyframes:",
        ]
        content: List[Dict[str, Any]] = [{"type": "text", "text": "\n".join(prompt_lines)}]
        max_side = max(256, min(int(settings.agent_vlm_image_max_side), 1200))
        for frame_id, frame in selected_ids.items():
            path = resolve_keyframe_path(frame)
            content.append({
                "type": "text",
                "text": f"{frame_id}: video={frame.get('video_id')} timestamp={frame.get('timestamp')} frame={frame.get('frame_name')}",
            })
            content.append({"type": "image_url", "image_url": {"url": _image_to_data_url(Path(path), max_side)}})

        max_retries = max(0, min(int(settings.agent_vlm_max_retries), 3))
        payload: Dict[str, Any] = {}
        for attempt in range(max_retries + 1):
            try:
                payload = _request_answer(
                    [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": content}],
                    settings.agent_vlm_model,
                    float(settings.agent_vlm_timeout_seconds),
                    int(settings.qa_max_tokens),
                )
                break
            except Exception as exc:
                errors.append(_clean(exc)[:180])
                logger.warning("Grounded Q&A VLM request failed: %s", exc)
                if attempt >= max_retries:
                    break
                retries += 1
                time.sleep(max(0.0, float(settings.agent_vlm_retry_backoff_seconds)) * (2**attempt))
        parsed, contract_errors = _normalise_answer(payload, set(selected_ids))
        errors.extend(contract_errors)
        if not contract_errors:
            verdict = parsed

    if verdict["status"] == "answered" and verdict["confidence"] < float(settings.qa_min_confidence):
        verdict = _uncertain_answer(question, "VLM confidence was below the configured threshold.")

    supporting = set(verdict["supporting_frame_ids"])
    selected_identity_to_id = {_identity(frame): frame_id for frame_id, frame in selected_ids.items()}
    selected_identity = {_identity(frame) for frame in selected}
    ordered_candidates = selected + [frame for frame in candidates if _identity(frame) not in selected_identity]
    output: List[Dict[str, Any]] = []
    for rank, frame in enumerate(ordered_candidates[: max(1, top_k)], 1):
        item = dict(frame)
        frame_id = selected_identity_to_id.get(_identity(item))
        item["rank"] = rank
        item["answer"] = verdict["answer"]
        item["qa_status"] = verdict["status"]
        item["qa_confidence"] = verdict["confidence"]
        item["qa_reason"] = verdict["reason"]
        item["qa_supporting"] = bool(frame_id and frame_id in supporting)
        item["qa_evidence_id"] = frame_id
        item["qa_used_ocr_evidence"] = verdict["used_ocr_evidence"]
        item["qa_used_asr_evidence"] = verdict["used_asr_evidence"]
        output.append(item)

    summary = {
        "status": verdict["status"],
        "answer": verdict["answer"],
        "confidence": verdict["confidence"],
        "reason": verdict["reason"],
        "supporting_frame_ids": verdict["supporting_frame_ids"],
        "retrieved_frames": len(visual_frames),
        "evaluated_frames": len(selected),
        "ocr_evidence": len(text_evidence["ocr"]),
        "asr_evidence": len(text_evidence["asr"]),
        "retries": retries,
        "errors": errors[:6],
    }
    for item in output:
        item["qa_summary"] = summary
    return output, summary
