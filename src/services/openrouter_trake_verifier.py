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

SYSTEM_PROMPT = """You are a strict verifier for ordered video-event retrieval.
Each sequence contains one keyframe per requested event, in chronological order, from one video.
Judge whether every image visibly matches its corresponding event and whether the whole ordered sequence matches the request.
Do not invent unseen actions or infer missing transitions. Return strict JSON only.

Return every supplied sequence id exactly once using:
{
  "items": [
    {
      "id": "s1",
      "score": 0.0,
      "decision": "match|partial|wrong|uncertain",
      "reason": "short sequence-level reason",
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


def _request(messages: List[Dict[str, Any]], model: str, timeout: float, max_tokens: int) -> Dict[str, Any]:
    settings = get_settings()
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
        return _extract_json_object(content)

    try:
        return send(body)
    except urllib.error.HTTPError as exc:
        if exc.code not in {400, 404, 422}:
            raise
        fallback = dict(body)
        fallback.pop("response_format", None)
        return send(fallback)


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
) -> Tuple[List[Dict], Dict[str, Any]]:
    settings = get_settings()
    if (
        not settings.trake_vlm_enabled
        or not settings.agent_vlm_enabled
        or not settings.openrouter_api_key
        or not sequences
        or not events
    ):
        return sequences, {"enabled": False, "status": "disabled", "evaluated": 0}

    limit = max(1, min(int(settings.trake_vlm_max_sequences), len(sequences), 10))
    candidates: List[Tuple[str, Dict, List[Path]]] = []
    missing_images = 0
    for sequence in sequences[:limit]:
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
        return sequences, {
            "enabled": True,
            "status": "fallback",
            "evaluated": 0,
            "missing_images": missing_images,
        }

    lines = ["Ordered target events:"]
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
                int(settings.agent_vlm_max_tokens),
            )
            break
        except (urllib.error.URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(_clean(exc)[:180])
            if attempt >= max_retries:
                break
            retries += 1
            time.sleep(max(0.0, float(settings.agent_vlm_retry_backoff_seconds)) * (2**attempt))

    expected = {sequence_id for sequence_id, _sequence, _paths in candidates}
    verdicts, contract_errors = _normalise(payload, expected, len(events))
    if not verdicts:
        return sequences, {
            "enabled": True,
            "status": "fallback",
            "evaluated": 0,
            "requested": len(candidates),
            "retries": retries,
            "errors": errors[:3],
            "contract_errors": contract_errors[:6],
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
        key=lambda sequence: float(sequence.get("verification_score", sequence.get("total_score", 0.0))),
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
