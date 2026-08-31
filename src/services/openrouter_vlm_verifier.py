"""OpenRouter vision verifier for Agent Search candidates."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
import re
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from PIL import Image

from src.config.settings import get_settings

logger = logging.getLogger(__name__)

VERDICT_CONTRACT_VERSION = "agent-vlm-verdict-v2"
VALID_DECISIONS = {"match", "partial", "wrong", "uncertain"}
_CACHE_LOCK = threading.Lock()

VERDICT_JSON_SCHEMA = {
    "name": "agent_vlm_verdicts",
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
                        "matched": {"type": "array", "items": {"type": "string"}},
                        "missing": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["id", "score", "decision", "reason", "matched", "missing"],
                },
            }
        },
        "required": ["items"],
    },
}

SYSTEM_PROMPT = """You are a strict visual verifier for keyframe retrieval.
You receive a natural-language target description, a search plan, and several candidate keyframe images.
Score each image by how well it visually matches the target.
Return strict JSON only, no markdown.

Rules:
- Score 1.0 only when the visible image strongly matches the important details.
- Penalize missing core entities, wrong scene, wrong action, wrong order, or wrong camera angle.
- Do not invent details. If uncertain, use a lower score and explain what is missing.
- Temporal descriptions may require nearby frames; score the current still image based on visible evidence only.
- Treat every checklist item as a constraint, not as permission to infer an unseen detail.
- Return every supplied candidate id exactly once. Never add an id that was not supplied.
- Use decision=match only when all visually verifiable core constraints are visible.
- matched and missing must contain short target constraints, not generic prose.

JSON schema:
{
  "items": [
    {"id": "c1", "score": 0.0, "decision": "match|partial|wrong|uncertain", "reason": "short reason", "matched": ["..."], "missing": ["..."]}
  ]
}
"""


def _clean(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _frame_identity(item: Dict[str, Any], fallback: str) -> str:
    for key in ("global_frame_id", "frame_path", "frame_name", "video_id", "faiss_id", "vector_id", "id"):
        value = item.get(key)
        if value not in (None, "", [], {}):
            return str(value)
    return fallback


def _first(item: Dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = item.get(key)
        if value not in (None, "", [], {}):
            return value
    return None


def _candidate_frame_file(item: Dict[str, Any]) -> str:
    video_id = str(_first(item, "video_id", "video_key", "videoKey") or "").strip()
    raw = str(_first(item, "frame_path", "image_path", "keyframe_path") or "").replace("\\", "/").strip("/")
    if raw:
        return raw

    frame_name = str(_first(item, "frame_name", "frameName") or "").strip()
    frame_id = str(_first(item, "frame_id", "frame_key", "frameKey", "n") or "").strip()
    split = str(_first(item, "split", "namespace", "folder_key", "folderKey") or "").strip()
    if not split and video_id:
        split = video_id.split("_")[0]
    if frame_name:
        file_name = frame_name
        prefix = f"{video_id}_"
        if video_id and file_name.startswith(prefix):
            file_name = file_name[len(prefix):]
    else:
        file_name = frame_id
    if file_name and not re.search(r"\.(?:webp|jpe?g|png)$", file_name, flags=re.IGNORECASE):
        file_name = f"{file_name}.webp"
    if split and video_id and file_name:
        return f"{split}/{video_id}/{file_name}"
    return file_name


def resolve_keyframe_path(item: Dict[str, Any]) -> Path | None:
    raw = _candidate_frame_file(item)
    if not raw:
        return None
    raw_path = Path(raw)
    candidates: List[Path] = []
    if raw_path.is_absolute():
        candidates.append(raw_path)
    root = get_settings().get_keyframes_root()
    candidates.append(root / raw)

    video_id = str(_first(item, "video_id", "video_key", "videoKey") or "").strip()
    frame_name = str(_first(item, "frame_name", "frameName") or "").strip()
    split = str(_first(item, "split", "namespace", "folder_key", "folderKey") or "").strip()
    if not split and video_id:
        split = video_id.split("_")[0]
    if split and video_id and frame_name:
        stripped = frame_name[len(f"{video_id}_"):] if frame_name.startswith(f"{video_id}_") else frame_name
        for name in (stripped, frame_name):
            if name:
                if not re.search(r"\.(?:webp|jpe?g|png)$", name, flags=re.IGNORECASE):
                    candidates.append(root / split / video_id / f"{name}.webp")
                    candidates.append(root / split / video_id / f"{name}.jpg")
                    candidates.append(root / split / video_id / f"{name}.png")
                candidates.append(root / split / video_id / name)

    for candidate in candidates:
        try:
            if candidate.exists() and candidate.is_file():
                return candidate
        except OSError:
            continue
    return None


def _image_to_data_url(path: Path, max_side: int) -> str:
    with Image.open(path) as image:
        image = image.convert("RGB")
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        output = io.BytesIO()
        image.save(output, format="JPEG", quality=82, optimize=True)
    encoded = base64.b64encode(output.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _extract_json_object(value: str) -> Dict[str, Any]:
    text = str(value or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return {}
    try:
        parsed = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _normalise_vlm_items(
    payload: Dict[str, Any],
    expected_ids: set[str] | None = None,
) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
    values = payload.get("items")
    if not isinstance(values, list):
        return {}, ["response.items must be an array"]
    output: Dict[str, Dict[str, Any]] = {}
    errors: List[str] = []
    for item in values:
        if not isinstance(item, dict):
            errors.append("response contains a non-object item")
            continue
        item_id = _clean(item.get("id"))
        if not item_id:
            errors.append("response item is missing id")
            continue
        if expected_ids is not None and item_id not in expected_ids:
            errors.append(f"unexpected candidate id: {item_id}")
            continue
        if item_id in output:
            errors.append(f"duplicate candidate id: {item_id}")
            continue
        try:
            score = float(item["score"])
        except (KeyError, TypeError, ValueError):
            errors.append(f"invalid score for {item_id}")
            continue
        if not 0.0 <= score <= 1.0:
            errors.append(f"score out of range for {item_id}")
            continue
        decision = _clean(item.get("decision")).lower()
        if decision not in VALID_DECISIONS:
            errors.append(f"invalid decision for {item_id}: {decision or '<blank>'}")
            continue
        if not isinstance(item.get("matched"), list) or not isinstance(item.get("missing"), list):
            errors.append(f"matched/missing must be arrays for {item_id}")
            continue
        if any(not isinstance(value, str) for value in item["matched"] + item["missing"]):
            errors.append(f"matched/missing must contain strings for {item_id}")
            continue
        reason = _clean(item.get("reason"))[:500]
        if not reason:
            errors.append(f"missing reason for {item_id}")
            continue
        matched = item["matched"]
        missing = item["missing"]
        output[item_id] = {
            "score": score,
            "decision": decision,
            "reason": reason,
            "matched": [_clean(value) for value in matched if _clean(value)][:8],
            "missing": [_clean(value) for value in missing if _clean(value)][:8],
        }
    if expected_ids is not None:
        for missing_id in sorted(expected_ids - set(output)):
            errors.append(f"missing verdict for candidate id: {missing_id}")
    return output, errors


def vision_gateway_available(settings: Any = None) -> bool:
    """True when the AI gateway is on and has at least one usable Vision provider."""
    settings = settings or get_settings()
    if not getattr(settings, "ai_gateway_enabled", False):
        return False
    try:
        from src.services.ai import gateway as ai_gateway

        return ai_gateway.vision_available(settings)
    except Exception:
        return False


def _request_vlm_via_gateway(messages: List[Dict[str, Any]], max_tokens: int) -> Dict[str, Any]:
    from src.services.ai import gateway as ai_gateway
    from src.services.ai.base import AllProvidersFailed

    try:
        payload, _attempts, _provider = ai_gateway.vision_completion(
            messages,
            max_tokens=max_tokens,
            response_format={"type": "json_schema", "json_schema": VERDICT_JSON_SCHEMA},
        )
    except AllProvidersFailed as exc:
        # Surface as a network-style error so the caller records a batch failure
        # and returns a no-VLM result with a clear status.
        raise urllib.error.URLError(f"vision chain exhausted: {exc}") from exc
    content = (((payload.get("choices") or [{}])[0].get("message") or {}).get("content") or "")
    if isinstance(content, list):
        content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return _extract_json_object(content)


def _request_openrouter_vlm(messages: List[Dict[str, Any]], model: str, max_tokens: int, timeout: float) -> Dict[str, Any]:
    settings = get_settings()
    if vision_gateway_available(settings):
        return _request_vlm_via_gateway(messages, max_tokens)
    body = {
        "model": model,
        "temperature": 0,
        "max_tokens": max_tokens,
        "messages": messages,
        "response_format": {"type": "json_schema", "json_schema": VERDICT_JSON_SCHEMA},
    }

    def send(request_body: Dict[str, Any]) -> Dict[str, Any]:
        request = urllib.request.Request(
            settings.openrouter_base_url.rstrip("/") + "/chat/completions",
            data=json.dumps(request_body).encode("utf-8"),
            method="POST",
            headers={
                "Authorization": f"Bearer {settings.openrouter_api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": settings.openrouter_site_url,
                "X-Title": settings.openrouter_app_name,
            },
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        content = (((payload.get("choices") or [{}])[0].get("message") or {}).get("content") or "")
        return _extract_json_object(content)

    try:
        return send(body)
    except urllib.error.HTTPError as exc:
        # Some OpenRouter routes do not expose structured output even when the
        # underlying model can still follow the strict JSON prompt.
        if exc.code not in {400, 404, 422}:
            raise
        fallback_body = dict(body)
        fallback_body.pop("response_format", None)
        return send(fallback_body)


def _chunks(values: List[Any], size: int) -> Iterable[List[Any]]:
    for index in range(0, len(values), max(1, size)):
        yield values[index : index + max(1, size)]


def _prompt_fingerprint(plan: Dict[str, Any], model: str) -> str:
    checks = []
    for check in (plan.get("must_have_checks") or [])[:10]:
        if isinstance(check, dict):
            checks.append(_clean(check.get("label") or check.get("query_en")))
    value = {
        "contract": VERDICT_CONTRACT_VERSION,
        "model": model,
        "original_query": _clean(plan.get("original_query")),
        "visual_query": _clean(plan.get("visual_query")),
        "checks": checks,
    }
    return hashlib.sha256(json.dumps(value, ensure_ascii=True, sort_keys=True).encode("utf-8")).hexdigest()


def _cache_key(prompt_fingerprint: str, identity: str, path: Path) -> str:
    try:
        stat = path.stat()
        image_version = f"{stat.st_size}:{stat.st_mtime_ns}"
    except OSError:
        image_version = "unknown"
    raw = f"{prompt_fingerprint}|{identity}|{path.resolve()}|{image_version}"
    return hashlib.sha256(raw.encode("utf-8", errors="replace")).hexdigest()


def _read_verdict_cache(path: Path, ttl_seconds: int) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.warning("Ignoring unreadable Agent VLM cache: %s", path)
        return {}
    entries = payload.get("entries") if isinstance(payload, dict) else None
    if not isinstance(entries, dict):
        return {}
    if ttl_seconds <= 0:
        return entries
    cutoff = time.time() - ttl_seconds
    fresh: Dict[str, Dict[str, Any]] = {}
    for key, value in entries.items():
        if not isinstance(value, dict):
            continue
        try:
            created_at = float(value.get("created_at") or 0.0)
        except (TypeError, ValueError):
            continue
        if created_at >= cutoff:
            fresh[key] = value
    return fresh


def _write_verdict_cache(path: Path, entries: Dict[str, Dict[str, Any]], max_entries: int) -> None:
    if max_entries > 0 and len(entries) > max_entries:
        newest = sorted(entries.items(), key=lambda pair: float(pair[1].get("created_at") or 0.0), reverse=True)
        entries = dict(newest[:max_entries])
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps({"version": 1, "entries": entries}, ensure_ascii=True, separators=(",", ":")),
        encoding="utf-8",
    )
    temporary.replace(path)


def _request_batch_with_retries(
    messages: List[Dict[str, Any]],
    model: str,
    max_tokens: int,
    timeout: float,
    max_retries: int,
    backoff_seconds: float,
) -> Tuple[Dict[str, Any], int]:
    retries_used = 0
    for attempt in range(max_retries + 1):
        try:
            return _request_openrouter_vlm(messages, model, max_tokens, timeout), retries_used
        except (urllib.error.URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError):
            if attempt >= max_retries:
                raise
            retries_used += 1
            if backoff_seconds > 0:
                time.sleep(backoff_seconds * (2**attempt))
    return {}, retries_used


def _score_for_selection(item: Dict[str, Any]) -> float:
    for key in ("verification_score", "agent_score", "final_score", "normalized_score", "score", "_score"):
        try:
            return max(0.0, float(item.get(key)))
        except (TypeError, ValueError):
            continue
    return 0.0


def _select_candidate_frames(
    frames: List[Dict[str, Any]],
    candidate_limit: int,
    pool_limit: int,
    per_video_limit: int,
) -> List[Dict[str, Any]]:
    if candidate_limit <= 0:
        return []
    pool = sorted(
        frames[: max(candidate_limit, min(pool_limit, len(frames)))],
        key=_score_for_selection,
        reverse=True,
    )
    selected: List[Dict[str, Any]] = []
    selected_ids = set()
    video_counts: Dict[str, int] = {}

    def add(item: Dict[str, Any]) -> bool:
        identity = _frame_identity(item, "")
        if identity in selected_ids:
            return False
        selected_ids.add(identity)
        selected.append(item)
        video = _clean(_first(item, "video_id", "video_key", "videoKey")) or "__unknown__"
        video_counts[video] = video_counts.get(video, 0) + 1
        return len(selected) >= candidate_limit

    for item in pool:
        video = _clean(_first(item, "video_id", "video_key", "videoKey")) or "__unknown__"
        if video_counts.get(video, 0) >= per_video_limit:
            continue
        if add(item):
            return selected

    for item in pool:
        if add(item):
            break
    return selected


def verify_frames_with_openrouter_vlm(frames: List[Dict[str, Any]], plan: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    settings = get_settings()
    _gateway_ready = vision_gateway_available(settings)
    if not settings.agent_vlm_enabled or not frames or (
        not settings.openrouter_api_key and not _gateway_ready
    ):
        return frames, {"enabled": False, "method": "none", "evaluated": 0}

    candidate_limit = max(1, min(int(settings.agent_vlm_max_candidates or 12), len(frames), 50))
    pool_limit = max(candidate_limit, min(int(settings.agent_vlm_candidate_pool or 40), len(frames), 100))
    per_video_limit = max(1, min(int(settings.agent_vlm_per_video_limit or 3), candidate_limit))
    batch_size = max(1, min(int(settings.agent_vlm_batch_size or 4), 8))
    max_side = max(256, min(int(settings.agent_vlm_image_max_side or 768), 1600))
    candidates: List[Tuple[str, Dict[str, Any], Path, str]] = []
    missing_images = 0

    selected_frames = _select_candidate_frames(frames, candidate_limit, pool_limit, per_video_limit)
    for item in selected_frames:
        path = resolve_keyframe_path(item)
        if path is None:
            missing_images += 1
            continue
        candidate_id = f"c{len(candidates) + 1}"
        candidates.append((candidate_id, item, path, _frame_identity(item, candidate_id)))

    if not candidates:
        return frames, {
            "enabled": True,
            "method": "openrouter_vlm",
            "status": "fallback",
            "fallback_used": True,
            "evaluated": 0,
            "missing_images": missing_images,
            "error": "No local keyframe images resolved.",
        }

    results: Dict[str, Dict[str, Any]] = {}
    result_sources: Dict[str, str] = {}
    errors: List[str] = []
    contract_errors: List[str] = []
    retries_used = 0
    api_calls = 0
    model = settings.agent_vlm_model
    checklist = plan.get("must_have_checks") or []
    checklist_text = "; ".join(_clean(check.get("label") or check.get("query_en")) for check in checklist[:10] if isinstance(check, dict))

    cache_enabled = bool(settings.agent_vlm_cache_enabled)
    cache_path = settings.get_agent_vlm_cache_path()
    cache_entries: Dict[str, Dict[str, Any]] = {}
    cache_keys: Dict[str, str] = {}
    cache_hits = 0
    if cache_enabled:
        with _CACHE_LOCK:
            cache_entries = _read_verdict_cache(cache_path, max(0, int(settings.agent_vlm_cache_ttl_seconds or 0)))
        fingerprint = _prompt_fingerprint(plan, model)
        for candidate_id, _item, path, identity in candidates:
            key = _cache_key(fingerprint, identity, path)
            cache_keys[candidate_id] = key
            cached = cache_entries.get(key)
            verdict = cached.get("verdict") if isinstance(cached, dict) else None
            if not isinstance(verdict, dict):
                continue
            normalised, cache_contract_errors = _normalise_vlm_items(
                {"items": [{"id": candidate_id, **verdict}]},
                {candidate_id},
            )
            if cache_contract_errors or candidate_id not in normalised:
                continue
            results[candidate_id] = normalised[candidate_id]
            result_sources[candidate_id] = "cache"
            cache_hits += 1

    uncached_candidates = [candidate for candidate in candidates if candidate[0] not in results]
    cache_updates: Dict[str, Dict[str, Any]] = {}

    for batch in _chunks(uncached_candidates, batch_size):
        text_lines = [
            f"Target description: {_clean(plan.get('original_query'))}",
            f"Main English query: {_clean(plan.get('visual_query'))}",
            f"Required visual constraints: {checklist_text or 'Use only the target description and main query.'}",
            "Candidate images are attached in order with ids below. Return one complete verdict for every id exactly once.",
            "Judge only visible evidence in each still image. Put every absent core constraint in missing.",
        ]
        for candidate_id, item, _path, identity in batch:
            text_lines.append(
                f"{candidate_id}: video={_clean(_first(item, 'video_id', 'video_key', 'videoKey'))}, "
                f"frame={_clean(_first(item, 'frame_name', 'frame_id', 'frame_key', 'n'))}, identity={identity}"
            )

        content: List[Dict[str, Any]] = [{"type": "text", "text": "\n".join(text_lines)}]
        try:
            for candidate_id, _item, path, _identity in batch:
                content.append({"type": "text", "text": f"Image {candidate_id}"})
                content.append({"type": "image_url", "image_url": {"url": _image_to_data_url(path, max_side)}})
            payload, batch_retries = _request_batch_with_retries(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": content},
                ],
                model=model,
                max_tokens=int(settings.agent_vlm_max_tokens or 900),
                timeout=float(settings.agent_vlm_timeout_seconds or 45.0),
                max_retries=max(0, min(int(settings.agent_vlm_max_retries or 0), 3)),
                backoff_seconds=max(0.0, float(settings.agent_vlm_retry_backoff_seconds or 0.0)),
            )
            retries_used += batch_retries
            api_calls += 1 + batch_retries
            expected_ids = {candidate_id for candidate_id, _item, _path, _identity in batch}
            normalised, batch_contract_errors = _normalise_vlm_items(payload, expected_ids)
            contract_errors.extend(batch_contract_errors)
            for candidate_id, verdict in normalised.items():
                results[candidate_id] = verdict
                result_sources[candidate_id] = "api"
                key = cache_keys.get(candidate_id)
                if cache_enabled and key:
                    cache_updates[key] = {"created_at": time.time(), "verdict": verdict}
        except (urllib.error.URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError) as exc:
            logger.warning("OpenRouter VLM verification batch failed: %s", exc)
            errors.append(_clean(exc)[:180])
            max_retries = max(0, min(int(settings.agent_vlm_max_retries or 0), 3))
            retries_used += max_retries
            api_calls += 1 + max_retries

    if cache_enabled and cache_updates:
        try:
            with _CACHE_LOCK:
                latest_entries = _read_verdict_cache(cache_path, max(0, int(settings.agent_vlm_cache_ttl_seconds or 0)))
                latest_entries.update(cache_updates)
                _write_verdict_cache(cache_path, latest_entries, max(1, int(settings.agent_vlm_cache_max_entries or 5000)))
        except OSError as exc:
            logger.warning("Unable to persist Agent VLM cache: %s", exc)
            errors.append(f"cache write failed: {_clean(exc)[:140]}")

    if not results:
        summary = {
            "enabled": True,
            "method": "openrouter_vlm",
            "status": "fallback",
            "fallback_used": True,
            "evaluated": 0,
            "requested": len(candidates),
            "missing_images": missing_images,
            "cache_hits": cache_hits,
            "cache_misses": len(uncached_candidates),
            "api_calls": api_calls,
            "retries": retries_used,
            "contract_errors": contract_errors[:6],
            "errors": errors[:3],
        }
        return frames, summary

    by_id = {candidate_id: (item, identity) for candidate_id, item, _path, identity in candidates}
    reranked: List[Dict[str, Any]] = []
    remainder: List[Dict[str, Any]] = []

    for candidate_id, (item, identity) in by_id.items():
        verdict = results.get(candidate_id)
        if not verdict:
            remainder.append(item)
            continue
        updated = dict(item)
        light_score = float(updated.get("verification_score") or updated.get("agent_score") or updated.get("score") or 0.0)
        vlm_score = float(verdict["score"])
        combined = max(0.0, min(1.0, 0.72 * vlm_score + 0.28 * light_score))
        previous = updated.get("agent_verification") if isinstance(updated.get("agent_verification"), dict) else {}
        updated["vlm_score"] = round(vlm_score, 6)
        updated["verification_score"] = round(combined, 6)
        updated["agent_verification"] = {
            **previous,
            "score": round(combined, 6),
            "method": "openrouter_vlm",
            "vlm_model": model,
            "vlm_score": round(vlm_score, 6),
            "vlm_decision": verdict.get("decision"),
            "vlm_reason": verdict.get("reason"),
            "vlm_matched": verdict.get("matched", []),
            "vlm_missing": verdict.get("missing", []),
            "vlm_source": result_sources.get(candidate_id, "api"),
            "vlm_contract_version": VERDICT_CONTRACT_VERSION,
        }
        updated["reason"] = verdict.get("reason") or updated.get("reason")
        updated["agent_matched_checks"] = list(dict.fromkeys(list(updated.get("agent_matched_checks") or []) + list(verdict.get("matched") or [])))[:12]
        updated["agent_missing_checks"] = list(dict.fromkeys(list(verdict.get("missing") or []) + list(updated.get("agent_missing_checks") or [])))[:12]
        reranked.append(updated)

    evaluated_identities = {identity for _candidate_id, _item, _path, identity in candidates}
    tail = [item for item in frames if _frame_identity(item, "") not in evaluated_identities]
    reranked.sort(key=lambda item: float(item.get("verification_score") or 0.0), reverse=True)
    combined_frames = reranked + remainder + tail
    for index, item in enumerate(combined_frames, start=1):
        item["rank"] = index

    return combined_frames, {
        "enabled": True,
        "method": "openrouter_vlm",
        "status": "verified" if len(results) == len(candidates) else "partial",
        "fallback_used": len(results) < len(candidates),
        "model": model,
        "evaluated": len(results),
        "requested": len(candidates),
        "candidate_pool": pool_limit,
        "per_video_limit": per_video_limit,
        "missing_images": missing_images,
        "cache_enabled": cache_enabled,
        "cache_hits": cache_hits,
        "cache_misses": len(uncached_candidates),
        "api_calls": api_calls,
        "retries": retries_used,
        "contract_errors": contract_errors[:6],
        "errors": errors[:3],
    }
