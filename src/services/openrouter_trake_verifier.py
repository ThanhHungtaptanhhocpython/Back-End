"""OpenRouter sequence verifier for ordered TRAKE results."""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

from src.config.settings import get_settings
from src.services.openrouter_vlm_verifier import _extract_json_object, _image_to_data_url

logger = logging.getLogger(__name__)

VALID_DECISIONS = {"match", "partial", "wrong", "uncertain"}

SYSTEM_PROMPT = """You are a conservative visual verifier for ordered video-event retrieval.
Each sequence contains one keyframe per requested event, in chronological order, from one video.
Judge each event against only its corresponding image. A sequence is a match only when every requested event is visibly supported by the correct image.
Do not infer missing actions, people, objects, vehicle context, or transitions from nearby frames, product labels, captions, or the text prompt.
If any event is only generic packaging, cooking, machinery, food prep, cartons without the requested objects, or an unclear partial view, mark that event missing and do not use decision=match.
For multi-scene requests, repeated visually similar frames cannot satisfy distinct events unless each frame clearly shows the distinct requested action/object/location.
Be skeptical: if the visible evidence is ambiguous, use partial, wrong, or uncertain instead of match.
The reason must describe concrete visual evidence from the images. Keep each reason under 12 words and never write paragraph reasons. Do not simply restate the target events. Return strict JSON only.

Return every supplied sequence id exactly once using:
{
  "items": [
    {
      "id": "s1",
      "score": 0.0,
      "decision": "match|partial|wrong|uncertain",
      "reason": "terse visual reason under 12 words",
      "matched_events": [1],
      "missing_events": [2]
    }
  ]
}
"""

JSON_SCHEMA = {
    "name": "trake_sequence_verdicts",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "id": {"type": "string"},
                        "score": {"type": "number", "minimum": 0, "maximum": 1},
                        "decision": {"type": "string", "enum": sorted(VALID_DECISIONS)},
                        "reason": {"type": "string"},
                        "matched_events": {"type": "array", "items": {"type": "integer"}},
                        "missing_events": {"type": "array", "items": {"type": "integer"}},
                    },
                    "required": ["id", "score", "decision", "reason", "matched_events", "missing_events"],
                },
            }
        },
        "required": ["items"],
    },
}


def _clean(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _summarise_exception(exc: BaseException) -> str:
    if isinstance(exc, urllib.error.HTTPError):
        body = ""
        try:
            body = exc.read().decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001 - best-effort diagnostics only
            body = ""
        details = _clean(body) or _clean(exc)
        return f"HTTP {exc.code}: {details}"[:1000]
    return _clean(exc)[:1000]


def _payload_preview(payload: Dict[str, Any]) -> str:
    if not payload:
        return ""
    try:
        return _clean(json.dumps(payload, ensure_ascii=False))[:500]
    except (TypeError, ValueError):
        return _clean(payload)[:500]


def _extract_trake_json(content: Any) -> Dict[str, Any]:
    if isinstance(content, list):
        content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
    raw = str(content or "")
    payload = _extract_json_object(raw)
    if not payload:
        cleaned = _clean(raw)
        head = cleaned[:500]
        tail = cleaned[-300:] if len(cleaned) > 500 else ""
        detail = f"raw_len={len(raw)} head={head}"
        if tail:
            detail += f" tail={tail}"
        raise ValueError(f"no JSON object in VLM response; {detail}")
    return payload


def _trake_max_tokens(base_max_tokens: int, sequence_count: int, event_count: int) -> int:
    base = max(1, int(base_max_tokens or 0))
    per_sequence = 140 + (max(1, int(event_count or 0)) * 20)
    estimated = 256 + (max(1, int(sequence_count or 0)) * per_sequence)
    return min(max(base, estimated), 3072)


def _request_via_gateway(messages: List[Dict[str, Any]], max_tokens: int) -> Dict[str, Any]:
    from src.services.ai import gateway as ai_gateway
    from src.services.ai.base import AllProvidersFailed

    try:
        payload, _attempts, _provider = ai_gateway.vision_completion(
            messages,
            max_tokens=max_tokens,
            response_format={"type": "json_schema", "json_schema": JSON_SCHEMA},
        )
    except AllProvidersFailed as exc:
        raise urllib.error.URLError(f"vision chain exhausted: {exc}") from exc
    content = (((payload.get("choices") or [{}])[0].get("message") or {}).get("content") or "")
    return _extract_trake_json(content)


def _request(messages: List[Dict[str, Any]], model: str, timeout: float, max_tokens: int) -> Dict[str, Any]:
    settings = get_settings()
    from src.services.openrouter_vlm_verifier import vision_gateway_available

    if vision_gateway_available(settings):
        return _request_via_gateway(messages, max_tokens)
    body = {
        "model": model,
        "temperature": 0,
        "max_tokens": max_tokens,
        "messages": messages,
        "response_format": {"type": "json_schema", "json_schema": JSON_SCHEMA},
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
        return _extract_trake_json(content)

    try:
        return send(body)
    except urllib.error.HTTPError as exc:
        if exc.code not in {400, 404, 422}:
            raise
        fallback = dict(body)
        fallback.pop("response_format", None)
        return send(fallback)


def _select_verification_sequences(sequences: List[Dict], limit: int) -> List[Dict]:
    """Select a diverse, exploratory VLM pool from a ranked sequence list.

    TRAKE can rank several wrong videos above a lower-scoring correct story.
    Verify top unique videos first, then sample deeper unique videos across the
    remaining pool before spending slots on variants from already-seen videos.
    """
    if limit <= 0 or not sequences:
        return []

    selected: List[Dict] = []
    selected_ids: set[int] = set()
    seen_videos: set[str] = set()
    for sequence in sequences:
        if not sequence.get("trace_verification_candidate"):
            continue
        video_id = _clean(sequence.get("video_id"))
        if video_id and video_id in seen_videos:
            continue
        selected.append(sequence)
        selected_ids.add(id(sequence))
        if video_id:
            seen_videos.add(video_id)
        if len(selected) >= limit:
            return selected[:limit]

    best_by_video: List[Dict] = list(selected)
    for sequence in sequences:
        if id(sequence) in selected_ids:
            continue
        video_id = _clean(sequence.get("video_id"))
        if video_id and video_id in seen_videos:
            continue
        if video_id:
            seen_videos.add(video_id)
        best_by_video.append(sequence)

    if len(best_by_video) <= limit:
        selected = list(best_by_video)
    else:
        head_count = max(1, min(len(best_by_video), limit // 2))
        selected = list(best_by_video[:head_count])
        selected_ids = {id(sequence) for sequence in selected}
        remaining = best_by_video[head_count:]
        slots = limit - len(selected)
        if slots > 0 and remaining:
            if slots >= len(remaining):
                sampled = remaining
            else:
                sampled = []
                for slot in range(slots):
                    position = round((slot + 1) * (len(remaining) - 1) / (slots + 1))
                    sampled.append(remaining[position])
            for sequence in sampled:
                if id(sequence) not in selected_ids:
                    selected.append(sequence)
                    selected_ids.add(id(sequence))
                if len(selected) >= limit:
                    break

    selected_ids = {id(sequence) for sequence in selected}
    for sequence in sequences:
        if len(selected) >= limit:
            break
        if id(sequence) in selected_ids:
            continue
        selected.append(sequence)
        selected_ids.add(id(sequence))
    return selected[:limit]


def _normalise(payload: Dict[str, Any], expected_ids: set[str], event_count: int) -> Tuple[Dict[str, Dict], List[str]]:
    rows = payload.get("items")
    if not isinstance(rows, list):
        return {}, ["response.items must be an array"]
    verdicts: Dict[str, Dict] = {}
    errors: List[str] = []
    valid_events = set(range(1, event_count + 1))
    for row in rows:
        if not isinstance(row, dict):
            errors.append("non-object verdict")
            continue
        item_id = _clean(row.get("id"))
        if item_id not in expected_ids or item_id in verdicts:
            errors.append(f"unexpected or duplicate sequence id: {item_id}")
            continue
        try:
            score = float(row["score"])
        except (KeyError, TypeError, ValueError):
            errors.append(f"invalid score for {item_id}")
            continue
        decision = _clean(row.get("decision")).lower()
        reason = _clean(row.get("reason"))[:500]
        matched = row.get("matched_events")
        missing = row.get("missing_events")
        if not 0.0 <= score <= 1.0 or decision not in VALID_DECISIONS or not reason:
            errors.append(f"invalid verdict contract for {item_id}")
            continue
        if not isinstance(matched, list) or not isinstance(missing, list):
            errors.append(f"invalid event arrays for {item_id}")
            continue
        try:
            matched_set = {int(value) for value in matched}
            missing_set = {int(value) for value in missing}
        except (TypeError, ValueError):
            errors.append(f"non-integer event id for {item_id}")
            continue
        if not matched_set.issubset(valid_events) or not missing_set.issubset(valid_events):
            errors.append(f"event id out of range for {item_id}")
            continue
        verdicts[item_id] = {
            "score": score,
            "decision": decision,
            "reason": reason,
            "matched_events": sorted(matched_set),
            "missing_events": sorted(missing_set),
        }
    for item_id in sorted(expected_ids - set(verdicts)):
        errors.append(f"missing verdict for {item_id}")
    return verdicts, errors


def verify_trake_sequences(
    sequences: List[Dict],
    events: List[str],
    resolve_image_path: Callable[[Dict], str],
    shared_context: str = "",
) -> Tuple[List[Dict], Dict[str, Any]]:
    settings = get_settings()
    from src.services.openrouter_vlm_verifier import vision_gateway_available

    _vlm_ready = bool(settings.openrouter_api_key) or vision_gateway_available(settings)
    if (
        not settings.trake_vlm_enabled
        or not _vlm_ready
        or not sequences
        or not events
    ):
        return sequences, {"enabled": False, "status": "disabled", "evaluated": 0}

    limit = max(1, min(int(settings.trake_vlm_max_sequences), len(sequences), 12))
    candidates: List[Tuple[str, Dict, List[Path]]] = []
    missing_images = 0
    for sequence in _select_verification_sequences(sequences, limit):
        paths: List[Path] = []
        for frame in sequence.get("frame_details") or []:
            path = Path(resolve_image_path(frame))
            if not path.is_file():
                paths = []
                missing_images += 1
                break
            paths.append(path)
        if len(paths) == len(events):
            candidates.append((f"s{len(candidates) + 1}", sequence, paths))

    if not candidates:
        logger.warning(
            "TRAKE VLM fallback: no resolvable candidate images requested=%d missing_images=%d",
            limit,
            missing_images,
        )
        return sequences, {
            "enabled": True,
            "status": "fallback",
            "evaluated": 0,
            "missing_images": missing_images,
        }

    lines = ["Ordered target events:"]
    clean_context = _clean(shared_context)
    if clean_context:
        lines.append(f"Shared sequence context: {clean_context}")
    for index, event in enumerate(events, 1):
        lines.append(f"Event {index}: {_clean(event)}")
    lines.append("Candidate sequences follow. Images for each sequence are attached in event order.")
    content: List[Dict[str, Any]] = [{"type": "text", "text": "\n".join(lines)}]
    max_side = max(256, min(int(settings.agent_vlm_image_max_side), 1200))
    for sequence_id, sequence, paths in candidates:
        timestamps = sequence.get("timestamps") or []
        content.append({"type": "text", "text": f"Sequence {sequence_id}, video={sequence.get('video_id')}, timestamps={timestamps}"})
        for event_index, path in enumerate(paths, 1):
            content.append({"type": "text", "text": f"{sequence_id} event {event_index}"})
            content.append({"type": "image_url", "image_url": {"url": _image_to_data_url(path, max_side)}})

    payload: Dict[str, Any] = {}
    errors: List[str] = []
    retries = 0
    max_retries = max(0, min(int(settings.agent_vlm_max_retries), 3))
    for attempt in range(max_retries + 1):
        try:
            payload = _request(
                [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": content}],
                settings.agent_vlm_model,
                float(settings.agent_vlm_timeout_seconds),
                _trake_max_tokens(settings.agent_vlm_max_tokens, len(candidates), len(events)),
            )
            break
        except (urllib.error.URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError) as exc:
            error = _summarise_exception(exc)
            errors.append(error)
            logger.warning(
                "TRAKE VLM request failed: attempt=%d/%d error=%s",
                attempt + 1,
                max_retries + 1,
                error,
            )
            if attempt >= max_retries:
                break
            retries += 1
            time.sleep(max(0.0, float(settings.agent_vlm_retry_backoff_seconds)) * (2**attempt))

    expected = {sequence_id for sequence_id, _sequence, _paths in candidates}
    verdicts, contract_errors = _normalise(payload, expected, len(events))
    if not verdicts:
        preview = _payload_preview(payload)
        logger.warning(
            "TRAKE VLM fallback: no usable verdicts requested=%d errors=%s contract_errors=%s payload=%s",
            len(candidates),
            errors[:3],
            contract_errors[:6],
            preview,
        )
        return sequences, {
            "enabled": True,
            "status": "fallback",
            "evaluated": 0,
            "requested": len(candidates),
            "retries": retries,
            "errors": errors[:3],
            "contract_errors": contract_errors[:6],
            "payload_preview": preview,
        }

    retrieval_scores = [float(sequence.get("total_score") or 0.0) for _sid, sequence, _paths in candidates]
    low, high = min(retrieval_scores), max(retrieval_scores)
    span = high - low
    for sequence_id, sequence, _paths in candidates:
        verdict = verdicts.get(sequence_id)
        if not verdict:
            continue
        retrieval = float(sequence.get("total_score") or 0.0)
        retrieval_norm = (retrieval - low) / span if span > 1e-9 else 1.0
        sequence["vlm_score"] = verdict["score"]
        sequence["vlm_decision"] = verdict["decision"]
        sequence["vlm_reason"] = verdict["reason"]
        sequence["vlm_matched_events"] = verdict["matched_events"]
        sequence["vlm_missing_events"] = verdict["missing_events"]
        sequence["verification_score"] = (0.68 * verdict["score"]) + (0.32 * retrieval_norm)

    sequences.sort(
        key=lambda sequence: (
            sequence.get("verification_score") is not None,
            float(sequence.get("verification_score", sequence.get("total_score", 0.0))),
        ),
        reverse=True,
    )
    return sequences, {
        "enabled": True,
        "status": "verified" if len(verdicts) == len(candidates) else "partial",
        "evaluated": len(verdicts),
        "requested": len(candidates),
        "missing_images": missing_images,
        "retries": retries,
        "errors": errors[:3],
        "contract_errors": contract_errors[:6],
    }
