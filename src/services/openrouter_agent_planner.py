"""Optional OpenRouter planner for Agent Search query enrichment."""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from typing import Any, Dict, Iterable, List

from src.config.settings import get_settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a query planner for a multimodal keyframe retrieval engine.
Return strict JSON only. Do not include markdown.

Goal:
- Convert the user's natural-language description, often Vietnamese, into enriched English search queries.
- For a single scene, prefer one holistic visual query that describes the whole scene.
- For an ordered multi-event description, set intent_type to temporal_sequence and split it into short event_queries in chronological order. Do not send the whole long description as one visual query.
- Add at most two extra visual queries only when they are full-scene variants, not keyword fragments.
- Preserve concrete details: subjects, count, clothing colors, action, camera angle, sequence, objects, visible text.
- Do not infer colors. In Vietnamese, "bi do" means pumpkin/squash, not a red object; "do hai nguoi dieu khien" means controlled by two people, not red.
- Only make the lion dance red if the user explicitly says "lan mau do" or the accented phrase "lan do" meaning a red lion dance.
- Use OCR only for exact visible text, logos, signs, subtitles, or screen text.
- Use ASR only for spoken words, sound, dialogue, speech, interview audio, or narration.
- This system has no VLM verifier. The checklist is used for lightweight reranking, so make each check visible and concrete.

JSON schema:
{
  "profile": "llm_enriched",
  "intent": "short English intent",
  "intent_type": "single_scene or temporal_sequence",
  "event_queries": ["ordered event 1", "ordered event 2", "ordered event 3"],
  "visual_queries": ["one rich English scene query", "optional rich variant", "optional rich variant"],
  "ocr_queries": [],
  "asr_queries": [],
  "must_have_checks": [
    {"id": "short_snake_case", "label": "human readable", "query_en": "English visual/text/audio evidence", "weight": 1.0}
  ],
  "negative_checks": [
    {"id": "short_snake_case", "label": "what to avoid"}
  ],
  "rerank_focus": ["short ranking instruction"]
}
"""

USER_TEMPLATE = """User description:
{prompt}

Local fallback plan summary:
{local_summary}

Return strict JSON only."""


def _clean(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", _clean(value).lower()).strip("_")
    return slug[:48] or "check"


def _normalise_text(value: Any) -> str:
    text = unicodedata.normalize("NFD", str(value or "").lower())
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return text.replace("\u0111", "d")


def _prompt_explicitly_says_red_lion(prompt: str) -> bool:
    raw = str(prompt or "").lower()
    if re.search(r"l\u00e2n\s+m\u00e0u\s+\u0111\u1ecf|l\u00e2n\s+\u0111\u1ecf", raw):
        return True
    normalised = _normalise_text(prompt)
    return bool(re.search(r"\blan\s+mau\s+do\b", normalised))


def _sanitize_visual_query(query: str, prompt: str) -> str:
    cleaned = _clean(query)
    if not _prompt_explicitly_says_red_lion(prompt):
        cleaned = re.sub(r"\ba\s+red\s+lion\s+dance\b", "a lion dance", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\bred\s+lion\s+dance\b", "lion dance", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\bred\s+lion\s+costume\b", "lion dance costume", cleaned, flags=re.IGNORECASE)
    return _clean(cleaned)


def _strip_json_fences(value: str) -> str:
    text = _clean(value)
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _extract_json_object(value: str) -> Dict[str, Any]:
    text = _strip_json_fences(value)
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return {}
    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _clean_string_list(values: Any, limit: int, min_words: int = 1) -> List[str]:
    if not isinstance(values, list):
        return []
    output: List[str] = []
    seen = set()
    for value in values:
        cleaned = _clean(value)
        if not cleaned or len(cleaned.split()) < min_words:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        output.append(cleaned)
        if len(output) >= limit:
            break
    return output


def _normalise_checks(values: Any, limit: int = 12) -> List[Dict[str, Any]]:
    if not isinstance(values, list):
        return []
    checks: List[Dict[str, Any]] = []
    seen = set()
    for value in values:
        if not isinstance(value, dict):
            continue
        query_en = _clean(value.get("query_en") or value.get("query") or value.get("label"))
        label = _clean(value.get("label") or query_en)
        if not query_en or not label:
            continue
        check_id = _slug(value.get("id") or label)
        if check_id in seen:
            continue
        seen.add(check_id)
        try:
            weight = float(value.get("weight", 1.0))
        except (TypeError, ValueError):
            weight = 1.0
        weight = max(0.1, min(2.0, weight))
        checks.append({"id": check_id, "label": label, "query_en": query_en, "weight": weight})
        if len(checks) >= limit:
            break
    return checks


def _normalise_negative_checks(values: Any, limit: int = 6) -> List[Dict[str, str]]:
    if not isinstance(values, list):
        return []
    checks: List[Dict[str, str]] = []
    seen = set()
    for value in values:
        if isinstance(value, dict):
            label = _clean(value.get("label") or value.get("query_en") or value.get("id"))
            check_id = _slug(value.get("id") or label)
        else:
            label = _clean(value)
            check_id = _slug(label)
        if not label or check_id in seen:
            continue
        seen.add(check_id)
        checks.append({"id": check_id, "label": label})
        if len(checks) >= limit:
            break
    return checks


def _merge_unique(primary: Iterable[str], fallback: Iterable[str], limit: int) -> List[str]:
    merged: List[str] = []
    seen = set()
    for value in list(primary or []) + list(fallback or []):
        cleaned = _clean(value)
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        merged.append(cleaned)
        if len(merged) >= limit:
            break
    return merged


def _local_summary(local_plan: Dict[str, Any]) -> str:
    summary = {
        "profile": local_plan.get("profile"),
        "visual_queries": local_plan.get("visual_queries", [])[:3],
        "ocr_queries": local_plan.get("ocr_queries", [])[:2],
        "asr_queries": local_plan.get("asr_queries", [])[:2],
        "must_have_checks": local_plan.get("must_have_checks", [])[:8],
        "intent_type": local_plan.get("intent_type", "single_scene"),
        "event_queries": local_plan.get("event_queries", [])[:5],
    }
    return json.dumps(summary, ensure_ascii=False)


def _normalise_llm_plan(payload: Dict[str, Any], prompt: str, local_plan: Dict[str, Any]) -> Dict[str, Any]:
    intent_type = "temporal_sequence" if _clean(payload.get("intent_type")).lower() == "temporal_sequence" else "single_scene"
    event_queries = _clean_string_list(payload.get("event_queries"), limit=5, min_words=3)
    visual_queries = [_sanitize_visual_query(query, prompt) for query in _clean_string_list(payload.get("visual_queries"), limit=3, min_words=6)]
    visual_queries = [query for query in visual_queries if query]
    if not visual_queries and intent_type == "temporal_sequence" and event_queries:
        visual_queries = [event_queries[0]]
    if not visual_queries:
        return {}

    ocr_queries = _clean_string_list(payload.get("ocr_queries"), limit=3)
    asr_queries = _clean_string_list(payload.get("asr_queries"), limit=2, min_words=3)
    must_have = _normalise_checks(payload.get("must_have_checks"))
    for check in must_have:
        check["label"] = _sanitize_visual_query(check.get("label", ""), prompt)
        check["query_en"] = _sanitize_visual_query(check.get("query_en", ""), prompt)
    if not must_have:
        must_have = list(local_plan.get("must_have_checks") or [])[:10]
    negative = _normalise_negative_checks(payload.get("negative_checks")) or list(local_plan.get("negative_checks") or [])[:6]
    rerank_focus = _clean_string_list(payload.get("rerank_focus"), limit=6, min_words=3) or list(local_plan.get("rerank_focus") or [])[:6]

    profile = _clean(payload.get("profile")) or "llm_enriched"
    if not profile.startswith("llm"):
        profile = f"llm_{_slug(profile)}"

    return {
        "profile": profile,
        "intent": _clean(payload.get("intent")) or _clean(prompt),
        "intent_type": intent_type,
        "event_queries": event_queries,
        "planner_source": "openrouter",
        "must_have_checks": must_have,
        "negative_checks": negative,
        "rerank_focus": rerank_focus,
        "visual_queries": visual_queries,
        "ocr_queries": ocr_queries,
        "asr_queries": asr_queries,
        "local_fallback_profile": local_plan.get("profile", ""),
    }


def _plan_with_gateway(settings, user_prompt: str) -> str:
    """Run the planner prompt through the multi-provider Text chain.

    Returns the raw model text, or ``""`` when the gateway is disabled, has no
    usable Text provider, or every provider failed.
    """
    if not settings.ai_gateway_enabled:
        return ""
    try:
        from src.services.ai import gateway as ai_gateway
    except Exception:
        return ""
    if not ai_gateway.text_available(settings):
        return ""
    try:
        result, _attempts = ai_gateway.text_completion(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            settings=settings,
            max_tokens=settings.agent_llm_max_tokens,
            response_format={"type": "json_object"},
        )
    except Exception as exc:
        logger.warning("Agent LLM planner gateway exhausted; using local fallback: %s", exc)
        return ""
    return result.text or ""


def plan_agent_query_with_openrouter(prompt: str, local_plan: Dict[str, Any]) -> Dict[str, Any]:
    settings = get_settings()
    if not settings.agent_llm_enabled:
        return {}

    user_prompt = USER_TEMPLATE.format(prompt=_clean(prompt), local_summary=_local_summary(local_plan))

    try:
        raw_text = _plan_with_gateway(settings, user_prompt)
        if not raw_text:
            if not settings.openrouter_api_key:
                return {}
            from langchain_openai import ChatOpenAI

            model_name = settings.agent_llm_model or settings.openrouter_model
            llm = ChatOpenAI(
                model=model_name,
                temperature=0,
                api_key=settings.openrouter_api_key,
                max_tokens=settings.agent_llm_max_tokens,
                base_url=settings.openrouter_base_url,
                default_headers={
                    "HTTP-Referer": settings.openrouter_site_url,
                    "X-Title": settings.openrouter_app_name,
                },
            )
            response = llm.invoke([
                ("system", SYSTEM_PROMPT),
                ("user", user_prompt),
            ])
            raw_text = getattr(response, "content", response)

        payload = _extract_json_object(raw_text)
        plan = _normalise_llm_plan(payload, prompt, local_plan)
        if plan:
            return plan
        logger.warning("Agent LLM planner returned invalid JSON plan; using local fallback.")
    except Exception as exc:
        logger.warning("Agent LLM planner failed; using local fallback: %s", exc)
    return {}
