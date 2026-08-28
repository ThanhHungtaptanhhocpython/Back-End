"""OpenRouter vision verifier for Agent Search candidates."""

from __future__ import annotations

import base64
import io
import json
import logging
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from PIL import Image

from src.config.settings import get_settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a strict visual verifier for keyframe retrieval.
You receive a natural-language target description, a search plan, and several candidate keyframe images.
Score each image by how well it visually matches the target.
Return strict JSON only, no markdown.

Rules:
- Score 1.0 only when the visible image strongly matches the important details.
- Penalize missing core entities, wrong scene, wrong action, wrong order, or wrong camera angle.
- Do not invent details. If uncertain, use a lower score and explain what is missing.
- Temporal descriptions may require nearby frames; score the current still image based on visible evidence only.

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


def _normalise_vlm_items(payload: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    values = payload.get("items")
    if not isinstance(values, list):
        return {}
    output: Dict[str, Dict[str, Any]] = {}
    for item in values:
        if not isinstance(item, dict):
            continue
        item_id = _clean(item.get("id"))
        if not item_id:
            continue
        try:
            score = float(item.get("score", 0.0))
        except (TypeError, ValueError):
            score = 0.0
        score = max(0.0, min(1.0, score))
        matched = item.get("matched") if isinstance(item.get("matched"), list) else []
        missing = item.get("missing") if isinstance(item.get("missing"), list) else []
        output[item_id] = {
            "score": score,
            "decision": _clean(item.get("decision")) or "uncertain",
            "reason": _clean(item.get("reason"))[:500],
            "matched": [_clean(value) for value in matched if _clean(value)][:8],
            "missing": [_clean(value) for value in missing if _clean(value)][:8],
        }
    return output


def _request_openrouter_vlm(messages: List[Dict[str, Any]], model: str, max_tokens: int, timeout: float) -> Dict[str, Any]:
    settings = get_settings()
    body = {
        "model": model,
        "temperature": 0,
        "max_tokens": max_tokens,
        "messages": messages,
    }
    data = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        settings.openrouter_base_url.rstrip("/") + "/chat/completions",
        data=data,
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


def _chunks(values: List[Any], size: int) -> Iterable[List[Any]]:
    for index in range(0, len(values), max(1, size)):
        yield values[index : index + max(1, size)]


def verify_frames_with_openrouter_vlm(frames: List[Dict[str, Any]], plan: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    settings = get_settings()
    if not settings.agent_vlm_enabled or not settings.openrouter_api_key or not frames:
        return frames, {"enabled": False, "method": "none", "evaluated": 0}

    candidate_limit = max(1, min(int(settings.agent_vlm_max_candidates or 12), len(frames), 50))
    batch_size = max(1, min(int(settings.agent_vlm_batch_size or 4), 8))
    max_side = max(256, min(int(settings.agent_vlm_image_max_side or 768), 1600))
    candidates: List[Tuple[str, Dict[str, Any], Path, str]] = []
    missing_images = 0

    for index, item in enumerate(frames[:candidate_limit], start=1):
        path = resolve_keyframe_path(item)
        if path is None:
            missing_images += 1
            continue
        candidate_id = f"c{index}"
        candidates.append((candidate_id, item, path, _frame_identity(item, candidate_id)))

    if not candidates:
        return frames, {"enabled": True, "method": "openrouter_vlm", "evaluated": 0, "missing_images": missing_images, "error": "No local keyframe images resolved."}

    results: Dict[str, Dict[str, Any]] = {}
    errors: List[str] = []
    model = settings.agent_vlm_model
    checklist = plan.get("must_have_checks") or []
    checklist_text = "; ".join(_clean(check.get("label") or check.get("query_en")) for check in checklist[:10] if isinstance(check, dict))

    for batch in _chunks(candidates, batch_size):
        text_lines = [
            f"Target description: {_clean(plan.get('original_query'))}",
            f"Main English query: {_clean(plan.get('visual_query'))}",
            f"Checklist: {checklist_text}",
            "Candidate images are attached in order with ids below. Score every candidate id exactly once.",
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
            payload = _request_openrouter_vlm(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": content},
                ],
                model=model,
                max_tokens=int(settings.agent_vlm_max_tokens or 900),
                timeout=float(settings.agent_vlm_timeout_seconds or 45.0),
            )
            results.update(_normalise_vlm_items(payload))
        except (urllib.error.URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError) as exc:
            logger.warning("OpenRouter VLM verification batch failed: %s", exc)
            errors.append(_clean(exc)[:180])

    if not results:
        summary = {"enabled": True, "method": "openrouter_vlm", "evaluated": 0, "missing_images": missing_images, "errors": errors[:3]}
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
        "model": model,
        "evaluated": len(results),
        "requested": len(candidates),
        "missing_images": missing_images,
        "errors": errors[:3],
    }