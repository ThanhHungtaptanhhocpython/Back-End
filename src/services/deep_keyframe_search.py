import hashlib
import logging
import os
import re
import unicodedata
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Tuple

from src.config.settings import get_settings
from src.services.reranker_service import reranker_service
from src.services.user_service import (
    getImageDataSingleTextSearch,
    getTextSearchASR,
    getTextSearchOCR,
)

logger = logging.getLogger(__name__)

VISUAL_HINTS = (
    "frame",
    "khung",
    "hinh",
    "canh",
    "clip",
    "video",
    "nguoi",
    "dang",
    "mac",
    "ao",
    "mau",
    "trong",
)
OCR_HINTS = ("chu tren", "co chu", "dong chu", "text", "ocr", "phu de", "caption", "logo", "bien hieu", "bang hieu", "\"", "'")
ASR_HINTS = ("noi", "noi ve", "nhac den", "hoi thoai", "mc", "phat bieu", "am thanh", "giong noi")
TEMPORAL_HINTS = ("sau do", "truoc do", "roi", "tiep theo", "sequence", "chuoi", "truoc tien")
QUERY_TRANSLATIONS = (
    ("mot nguoi dung duoi nuoc va roi den", "one person standing in water and shining a light"),
    ("mot nguoi dung duoi nuoc", "one person standing in water"),
    ("nguoi dung duoi nuoc", "person standing in water"),
    ("dung duoi nuoc", "standing in water"),
    ("duoi nuoc", "in water"),
    ("roi den", "shining a light"),
    ("chieu den", "shining a light"),
    ("nguoi nay keo luoi ca luc binh minh", "person pulling a fishing net at dawn"),
    ("keo luoi ca luc binh minh", "pulling a fishing net at dawn"),
    ("keo luoi ca", "pulling a fishing net"),
    ("luoi ca", "fishing net"),
    ("luc binh minh", "at dawn"),
    ("binh minh", "dawn"),
    ("mot nhom nguoi khac tien den dung may quay ghi hinh", "another group of people approaching and filming with a camera"),
    ("nhom nguoi khac tien den dung may quay ghi hinh", "another group of people approaching and filming with a camera"),
    ("mot nhom nguoi khac", "another group of people"),
    ("nhom nguoi khac", "another group of people"),
    ("tien den", "approaching"),
    ("dung may quay ghi hinh", "filming with a camera"),
    ("may quay ghi hinh", "video camera"),
    ("may quay", "camera"),
    ("ghi hinh", "filming"),
    ("nguoi nay", "person"),
    ("mot nguoi", "one person"),
    ("hai nguoi phu nu cho de an trong chuong", "two women feeding goats in a pen"),
    ("hai nguoi phu nu", "two women"),
    ("hai phu nu trong chuong trai", "two women in a farm pen"),
    ("hai phu nu", "two women"),
    ("nguoi phu nu", "woman"),
    ("phu nu", "women"),
    ("nguoi cho de an", "person feeding goats"),
    ("cho de an", "feeding goats"),
    ("de trong chuong trai", "goats in a farm pen"),
    ("chuong trai nuoi de", "goat farm pen"),
    ("chuong trai", "farm pen"),
    ("trong chuong", "in a pen"),
    ("chuong", "pen"),
    ("de", "goats"),
    ("ao thun trang", "white T-shirt"),
    ("ao do", "red garment"),
    ("ke soc tim", "purple striped"),
    ("mai che bang ton", "corrugated metal roof"),
    ("hang rao go", "wooden fence"),
    ("mim cuoi", "smiling"),
    ("trai", "farm"),
)


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _normalise_intent_text(value: Any) -> str:
    text = unicodedata.normalize("NFD", str(value or "").lower())
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return text.replace("\u0111", "d")


def _query_to_english(query: str) -> str:
    text = _clean_text(query).strip(" .,:;!?")
    if not text:
        return ""

    translated = _normalise_intent_text(text)
    translated = re.sub(r"\s*->\s*", " -> ", translated)
    for source, target in sorted(QUERY_TRANSLATIONS, key=lambda pair: len(pair[0]), reverse=True):
        translated = re.sub(rf"\b{re.escape(source)}\b", target, translated, flags=re.IGNORECASE)

    translated = re.sub(
        r"\b(tiep theo|sau do|truoc do|roi|canh|khung hinh|dang|duoc|co|la|voi|va|trong)\b",
        " ",
        translated,
        flags=re.IGNORECASE,
    )
    translated = _clean_text(translated).strip(" .,:;!?")
    return translated or text


def _first(source: Dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = source.get(key)
        if value not in (None, "", [], {}):
            return value
    return None


def _frame_identity(item: Dict[str, Any]) -> str:
    return str(
        _first(
            item,
            "global_frame_id",
            "frame_name",
            "frame_path",
            "video_id",
            "video_key",
            "faiss_id",
            "faiss_index",
            "vector_id",
            "id",
        )
    )


def _video_id(item: Dict[str, Any]) -> str:
    return str(_first(item, "video_id", "video_key", "videoKey", "unknown-video"))


def _timestamp(item: Dict[str, Any]) -> float:
    try:
        return float(_first(item, "timestamp", "time", "start_time") or 0)
    except (TypeError, ValueError):
        return 0.0


def _score(item: Dict[str, Any], rank: int) -> float:
    raw = _first(item, "final_score", "normalized_score", "score", "_score")
    try:
        return float(raw)
    except (TypeError, ValueError):
        return max(0.0, 1.0 - (rank - 1) * 0.03)


def _extract_quoted(text: str) -> List[str]:
    return [q.strip() for q in re.findall(r"[\"']([^\"']{2,80})[\"']", text) if q.strip()]


def _has_phrase(text: str, phrases: Tuple[str, ...]) -> bool:
    normalised = _normalise_intent_text(text)
    return any(phrase in normalised for phrase in phrases)


def _looks_like_command(sentence: str) -> bool:
    normalised = _normalise_intent_text(sentence)
    command_markers = (
        "hay tim sau",
        "tim sau",
        "tim ky",
        "dao sau",
        "deep search",
        "tu thu nhieu huong",
        "retrieval agent",
        "tra ve",
    )
    return any(marker in normalised for marker in command_markers)


def _extract_search_description(prompt: str) -> str:
    text = _clean_text(prompt)
    if not text:
        return ""

    if ":" in text and _looks_like_command(text.split(":", 1)[0]):
        text = text.split(":", 1)[1]

    sentences = re.split(r"(?<=[.!?])\s+|\n+", text)
    kept: List[str] = []
    for sentence in sentences:
        sentence = _clean_text(sentence)
        if not sentence:
            continue
        if _looks_like_command(sentence):
            continue
        kept.append(sentence)

    text = _clean_text(" ".join(kept) or text)
    text = re.sub(
        "(?i)^\\s*(toi|t\u00f4i)?\\s*(chua|ch\u01b0a|khong|kh\u00f4ng)?\\s*(tim|t\u00ecm)\\s*(duoc|\u0111\u01b0\u1ee3c)?\\s*",
        "",
        text,
    )
    text = re.sub(
        "(?i)^\\s*(canh|c\u1ea3nh|khung hinh|khung h\u00ecnh|frame|clip|video)\\s+(nay|n\u00e0y)?\\s*:?\\s*",
        "",
        text,
    )
    text = re.sub("(?i)\\s*(hay|h\u00e3y)\\s*(tim|t\u00ecm)\\s*(sau|k\u1ef9|ky).*?$", "", text)
    return _clean_text(text)


def _clean_temporal_event(text: str) -> str:
    event = _clean_text(text).strip(" .,:;!?-")
    event = re.sub(
        "(?i)^\\s*(ti\u1ebfp theo|tiep theo|sau \u0111\u00f3|sau do|r\u1ed3i|roi|tr\u01b0\u1edbc \u0111\u00f3|truoc do)\\s*(l\u00e0|la|\u0111\u01b0\u1ee3c|duoc)?\\s*(c\u1ea3nh|canh)?\\s*",
        "",
        event,
    )
    event = re.sub(
        "(?i)^\\s*(c\u1ea3nh|canh|khung h\u00ecnh|khung hinh|frame|clip|video)\\s+(n\u00e0y|nay)?\\s*:?\\s*",
        "",
        event,
    )
    return _clean_text(event).strip(" .,:;!?-")


def _temporal_events(text: str) -> List[str]:
    split_re = re.compile(
        "\\s*(?:->|(?:ti\u1ebfp theo|tiep theo)(?:\\s+l\u00e0\\s+c\u1ea3nh|\\s+la\\s+canh|\\s+l\u00e0|\\s+la)?|"
        "(?:sau \u0111\u00f3|sau do)(?:\\s+\u0111\u01b0\u1ee3c|\\s+duoc|\\s+l\u00e0|\\s+la)?|(?:r\u1ed3i)|(?:tr\u01b0\u1edbc \u0111\u00f3|truoc do))\\s*",
        flags=re.IGNORECASE,
    )
    return [event for event in (_clean_temporal_event(part) for part in split_re.split(text)) if event]


def expand_queries(prompt: str, max_queries: int = 8) -> List[Dict[str, str]]:
    base = _extract_search_description(prompt)
    if not base:
        base = _clean_text(prompt)

    queries: List[Dict[str, str]] = []

    def add(kind: str, query: str) -> None:
        query = _clean_text(query).strip(" .,:;!?")
        if not query:
            return
        query_en = _query_to_english(query)
        key = (kind, query.lower())
        existing = {(q["kind"], q["query"].lower()) for q in queries}
        if key not in existing:
            queries.append({"kind": kind, "query": query, "query_en": query_en})

    temporal_events = _temporal_events(base) if _has_phrase(base, TEMPORAL_HINTS) else []
    if len(temporal_events) >= 2:
        add("temporal", " -> ".join(temporal_events))
        for event in temporal_events:
            add("visual", event)
    else:
        add("visual", base)
        add("visual", re.sub(r"\b(chu|text|ocr|phu de|caption)\b.*", "", base, flags=re.IGNORECASE))

    normalised_base = _normalise_intent_text(base)
    if "de" in normalised_base and "chuong" in normalised_base:
        add("visual", "de trong chuong trai")
        add("visual", "chuong trai nuoi de")
    if "phu nu" in normalised_base and "de" in normalised_base:
        add("visual", "hai phu nu trong chuong trai")
        add("visual", "two women feeding goats in a farm pen")
    if "cho" in normalised_base and "de an" in normalised_base:
        add("visual", "nguoi cho de an")

    for quoted in _extract_quoted(prompt):
        add("ocr", quoted)
        add("asr", quoted)

    if _extract_quoted(prompt) or _has_phrase(prompt, OCR_HINTS):
        add("ocr", base)
    if _has_phrase(prompt, ASR_HINTS):
        add("asr", base)

    fragments = re.split(r"[,.;\n]+", base)
    for fragment in fragments:
        fragment = _clean_temporal_event(fragment)
        if len(fragment.split()) >= 4 and any(hint in _normalise_intent_text(fragment) for hint in VISUAL_HINTS):
            add("visual", fragment)

    return queries[:max_queries]


def _frame_order(item: Dict[str, Any]) -> float:
    for key in ("timestamp", "timestamp_s", "time", "start_time", "global_frame_id", "frame_id", "frame_key"):
        value = item.get(key)
        if value in (None, ""):
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            digits = re.findall(r"\d+", str(value))
            if digits:
                return float(digits[-1])
    return 0.0


def _round_robin_events(event_results: List[List[Dict[str, Any]]], limit: int) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    seen = set()
    max_len = max((len(results) for results in event_results), default=0)
    for idx in range(max_len):
        for results in event_results:
            if idx >= len(results):
                continue
            item = results[idx]
            identity = _frame_identity(item)
            if identity in seen:
                continue
            seen.add(identity)
            merged.append(item)
            if len(merged) >= limit:
                return merged
    return merged


def _run_beit3_temporal_query(text: str, per_query: int) -> List[Dict[str, Any]]:
    events = [part.strip() for part in text.split("->") if part.strip()]
    if len(events) < 2:
        return getImageDataSingleTextSearch(text, per_query)

    event_results: List[List[Dict[str, Any]]] = []
    for event_idx, event in enumerate(events):
        results = []
        for rank, item in enumerate(getImageDataSingleTextSearch(event, per_query) or [], start=1):
            if not isinstance(item, dict):
                continue
            copied = dict(item)
            copied["_temporal_event"] = event_idx
            copied["_temporal_rank"] = rank
            copied["_temporal_query"] = event
            results.append(copied)
        event_results.append(results)

    grouped: Dict[str, List[List[Dict[str, Any]]]] = defaultdict(lambda: [[] for _ in events])
    for event_idx, results in enumerate(event_results):
        for item in results:
            grouped[_video_id(item)][event_idx].append(item)

    sequences: List[Tuple[float, List[Dict[str, Any]]]] = []
    for candidates_by_event in grouped.values():
        if any(not candidates for candidates in candidates_by_event):
            continue

        beam: List[Tuple[float, List[Dict[str, Any]]]] = []
        for item in sorted(candidates_by_event[0], key=lambda row: row.get("_temporal_rank", 999))[:30]:
            beam.append((_score(item, int(item.get("_temporal_rank") or 1)), [item]))

        for next_candidates in candidates_by_event[1:]:
            next_beam: List[Tuple[float, List[Dict[str, Any]]]] = []
            ordered_next = sorted(next_candidates, key=_frame_order)
            for score, seq in beam:
                last_order = _frame_order(seq[-1])
                for item in ordered_next:
                    order = _frame_order(item)
                    if order <= last_order:
                        continue
                    rank = int(item.get("_temporal_rank") or 1)
                    gap = max(0.0, order - last_order)
                    gap_penalty = min(gap / 100000.0, 0.08)
                    next_beam.append((score + _score(item, rank) - gap_penalty, seq + [item]))
            next_beam.sort(key=lambda pair: pair[0], reverse=True)
            beam = next_beam[:30]
            if not beam:
                break
        sequences.extend(beam)

    sequences.sort(key=lambda pair: pair[0], reverse=True)
    merged: List[Dict[str, Any]] = []
    seen = set()
    for score, sequence in sequences:
        for item in sequence:
            identity = _frame_identity(item)
            if identity in seen:
                continue
            enriched = dict(item)
            enriched["temporal_score"] = round(float(score), 6)
            seen.add(identity)
            merged.append(enriched)
            if len(merged) >= per_query:
                return merged

    return merged or _round_robin_events(event_results, per_query)


def _run_query(query: Dict[str, str], per_query: int) -> List[Dict[str, Any]]:
    kind = query["kind"]
    text = query.get("query_en") or query["query"]
    try:
        if kind == "ocr":
            return getTextSearchOCR(text, topk=per_query)
        if kind == "asr":
            return getTextSearchASR(text, topk=per_query)
        if kind == "temporal":
            return _run_beit3_temporal_query(text, per_query)
        return getImageDataSingleTextSearch(text, per_query)
    except Exception as exc:
        logger.warning("Deep search query failed (%s): %s", kind, exc)
        return []


def _merge_results(query_results: Iterable[Tuple[Dict[str, str], List[Dict[str, Any]]]]) -> List[Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    source_weight = {"visual": 1.0, "ocr": 0.92, "asr": 0.82, "temporal": 1.08}

    for query, results in query_results:
        kind = query["kind"]
        for rank, item in enumerate(results or [], start=1):
            if not isinstance(item, dict):
                continue
            identity = _frame_identity(item)
            entry = merged.setdefault(
                identity,
                {
                    "item": dict(item),
                    "score": 0.0,
                    "sources": set(),
                    "queries": [],
                    "best_rank": rank,
                },
            )
            rank_score = 1.0 / (rank + 4)
            model_score = max(0.0, min(1.0, _score(item, rank)))
            entry["score"] += source_weight.get(kind, 1.0) * (rank_score + model_score * 0.12)
            entry["sources"].add(kind)
            entry["queries"].append(query["query"])
            entry["best_rank"] = min(entry["best_rank"], rank)

    return sorted(
        merged.values(),
        key=lambda entry: (entry["score"], -entry["best_rank"]),
        reverse=True,
    )


def _resolve_keyframe_path(item: Dict[str, Any]) -> str:
    """Resolve a BEiT3 metadata path to the local keyframe file."""
    raw_path = str(_first(item, "frame_path", "image_path", "keyframe_path") or "").replace("\\", "/")
    if not raw_path:
        return ""
    if os.path.isabs(raw_path):
        return raw_path
    return str(get_settings().get_keyframes_root() / raw_path)


def _image_content_key(item: Dict[str, Any]) -> str:
    """Use image bytes, not video/frame IDs, to suppress duplicate corpus frames."""
    path = _resolve_keyframe_path(item)
    try:
        digest = hashlib.sha1()
        with open(path, "rb") as image_file:
            for chunk in iter(lambda: image_file.read(1024 * 1024), b""):
                digest.update(chunk)
        return f"sha1:{digest.hexdigest()}"
    except OSError:
        return _frame_identity(item)


def _vqa_query_for(item: Dict[str, Any], queries: List[Dict[str, str]], matched_values: Iterable[str] = ()) -> str:
    temporal_query = _clean_text(item.get("_temporal_query"))
    if temporal_query:
        return _query_to_english(temporal_query)

    matched_queries = set(matched_values or item.get("deep_queries") or [])
    for query in queries:
        if query.get("query") in matched_queries:
            return _clean_text(query.get("query_en") or query.get("query"))
    return _clean_text((item.get("deep_queries") or [""])[0])


def _rerank_with_vqa(ranked: List[Dict[str, Any]], queries: List[Dict[str, str]]) -> int:
    """Validate the best KIS candidates against the action that retrieved them."""
    settings = get_settings()
    if not settings.kis_vqa_rerank:
        return 0

    try:
        limit = max(1, min(int(settings.kis_vqa_rerank_candidates), 60))
    except (TypeError, ValueError):
        limit = 24

    validated = 0
    for entry in ranked[:limit]:
        item = entry["item"]
        image_path = _resolve_keyframe_path(item)
        query = _vqa_query_for(item, queries, entry.get("queries") or [])
        if not image_path or not os.path.exists(image_path) or not query:
            continue
        score = reranker_service.score_image(image_path, f"Does this image show {query}? Answer yes or no.")
        if score <= 0:
            continue
        item["vqa_score"] = round(float(score), 6)
        item["vqa_query"] = query
        entry["score"] += 0.25 * float(score)
        validated += 1

    if validated:
        ranked.sort(key=lambda entry: (entry["score"], -entry["best_rank"]), reverse=True)
    return validated


def _dedup_ranked(ranked: List[Dict[str, Any]], topk: int, min_seconds: float = 8.0) -> List[Dict[str, Any]]:
    accepted: List[Dict[str, Any]] = []
    times_by_video: Dict[str, List[float]] = defaultdict(list)
    seen_content = set()

    for entry in ranked:
        item = entry["item"]
        content_key = _image_content_key(item)
        if content_key in seen_content:
            continue
        video = _video_id(item)
        ts = _timestamp(item)
        if any(abs(ts - previous) < min_seconds for previous in times_by_video[video]):
            continue
        seen_content.add(content_key)
        times_by_video[video].append(ts)

        item["deep_score"] = round(float(entry["score"]), 6)
        item["deep_sources"] = sorted(entry["sources"])
        item["deep_queries"] = list(dict.fromkeys(entry["queries"]))[:4]
        item["reason"] = _reason_for(item)
        accepted.append(item)
        if len(accepted) >= topk:
            break

    return accepted


def _reason_for(item: Dict[str, Any]) -> str:
    sources = ", ".join(item.get("deep_sources") or [])
    query = (item.get("deep_queries") or [""])[0]
    video = _video_id(item)
    frame = _first(item, "frame_name", "frame_id", "frame_key", "global_frame_id")
    return f"Matched by {sources or 'visual'} search for '{query}' near {video}/{frame}."


def _atomic_vqa_questions(query: str) -> List[str]:
    """Turn one event into small visual checks that BLIP can answer reliably."""
    text = _clean_text(query).lower()
    questions: List[str] = []

    if "standing in water" in text or "standing in the water" in text:
        questions.append("Is a person standing in water?")
    if "shining a light" in text or "flashlight" in text or "holding a light" in text:
        questions.append("Is a person holding or shining a light?")
    if "pulling a fishing net" in text or "pulling a net" in text:
        questions.append("Is a person pulling a fishing net?")
    if "at dawn" in text or "sunrise" in text:
        questions.append("Is this scene at dawn or sunrise?")
    if "group of people" in text:
        questions.append("Is there a group of people?")
    if "filming" in text or "video camera" in text:
        questions.append("Is someone filming with a camera?")

    if not questions:
        clauses = [part.strip(" .") for part in re.split(r"\s+(?:and|while)\s+", text) if part.strip(" .")]
        questions = [f"Does this image show {clause}?" for clause in clauses[:2]]
    return list(dict.fromkeys(questions))[:3]


def _ordered_evidence(candidates_by_event: List[List[Dict[str, Any]]]) -> Tuple[List[Tuple[int, Dict[str, Any]]], bool]:
    """Choose a high-scoring, chronologically ordered frame for every event."""
    if not candidates_by_event or any(not candidates for candidates in candidates_by_event):
        evidence = [(idx, candidates[0]) for idx, candidates in enumerate(candidates_by_event) if candidates]
        return evidence, False

    beam: List[Tuple[float, List[Tuple[int, Dict[str, Any]]]]] = [
        (_score(item, rank), [(0, item)])
        for rank, item in enumerate(candidates_by_event[0][:8], start=1)
    ]
    for event_idx, candidates in enumerate(candidates_by_event[1:], start=1):
        expanded: List[Tuple[float, List[Tuple[int, Dict[str, Any]]]]] = []
        for accumulated, sequence in beam:
            last_order = _frame_order(sequence[-1][1])
            for rank, item in enumerate(candidates[:8], start=1):
                if _frame_order(item) <= last_order:
                    continue
                expanded.append((accumulated + _score(item, rank), sequence + [(event_idx, item)]))
        expanded.sort(key=lambda row: row[0], reverse=True)
        beam = expanded[:40]
        if not beam:
            evidence = [(idx, event_candidates[0]) for idx, event_candidates in enumerate(candidates_by_event)]
            return evidence, False
    return beam[0][1], True


def _event_video_search(events: List[str], topk: int, per_query: int) -> Dict[str, Any]:
    """KIS multi-scene retrieval: recall frames, rank videos, then validate evidence."""
    settings = get_settings()
    try:
        recall_k = max(per_query, min(int(settings.kis_event_recall_k), 1000))
    except (TypeError, ValueError):
        recall_k = max(per_query, 300)

    queries = [
        {"kind": "visual", "query": event, "query_en": _query_to_english(event)}
        for event in events
    ]
    event_results: List[List[Dict[str, Any]]] = []
    total_candidates = 0
    for event_idx, query in enumerate(queries):
        results: List[Dict[str, Any]] = []
        for rank, item in enumerate(getImageDataSingleTextSearch(query["query_en"], recall_k) or [], start=1):
            if not isinstance(item, dict):
                continue
            copied = dict(item)
            copied["_event_index"] = event_idx
            copied["_event_rank"] = rank
            results.append(copied)
        event_results.append(results)
        total_candidates += len(results)

    grouped: Dict[str, List[List[Dict[str, Any]]]] = defaultdict(lambda: [[] for _ in events])
    for event_idx, results in enumerate(event_results):
        for item in results:
            grouped[_video_id(item)][event_idx].append(item)

    preliminary: List[Dict[str, Any]] = []
    for video_id, candidates_by_event in grouped.items():
        for candidates in candidates_by_event:
            candidates.sort(key=lambda item: _score(item, int(item.get("_event_rank") or 1)), reverse=True)
        coverage = sum(bool(candidates) for candidates in candidates_by_event)
        evidence, ordered = _ordered_evidence(candidates_by_event)
        retrieval_scores = [_score(item, int(item.get("_event_rank") or 1)) for _, item in evidence]
        average_retrieval = sum(retrieval_scores) / len(retrieval_scores) if retrieval_scores else 0.0
        preliminary.append({
            "video_id": video_id,
            "coverage": coverage,
            "candidates": candidates_by_event,
            "evidence": evidence,
            "ordered": ordered,
            "average_retrieval": average_retrieval,
            "video_score": (coverage / len(events)) * 4.0 + average_retrieval + (0.75 if ordered else 0.0),
        })
    preliminary.sort(key=lambda row: row["video_score"], reverse=True)

    requested_videos = max(1, (topk + len(events) - 1) // len(events))
    try:
        rerank_videos = min(max(requested_videos, int(settings.kis_video_rerank_videos)), 16)
    except (TypeError, ValueError):
        rerank_videos = min(max(requested_videos, 8), 16)
    try:
        frames_per_event = max(1, min(int(settings.kis_vqa_frames_per_event), 4))
    except (TypeError, ValueError):
        frames_per_event = 2
    try:
        vqa_threshold = float(settings.kis_vqa_threshold)
    except (TypeError, ValueError):
        vqa_threshold = 0.55

    preferred = [record for record in preliminary if record["coverage"] == len(events) and record["ordered"]]
    fallback = [record for record in preliminary if record not in preferred]
    selected_videos: List[Dict[str, Any]] = []
    seen_signatures = set()
    for record in preferred + fallback:
        signature = tuple(_image_content_key(item) for _, item in record["evidence"])
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        selected_videos.append(record)
        if len(selected_videos) >= rerank_videos:
            break

    vqa_cache: Dict[Tuple[str, str], float] = {}
    for record in selected_videos:
        preliminary_evidence = {event_idx: item for event_idx, item in record["evidence"]}
        scored_by_event: List[List[Tuple[float, float, Dict[str, Any]]]] = []
        for event_idx, candidates in enumerate(record["candidates"]):
            if not candidates:
                scored_by_event.append([])
                continue
            event_query = queries[event_idx]["query_en"]
            questions = _atomic_vqa_questions(event_query)
            candidate_pool = list(candidates[:frames_per_event])
            ordered_candidate = preliminary_evidence.get(event_idx)
            if ordered_candidate is not None and all(
                _frame_identity(item) != _frame_identity(ordered_candidate) for item in candidate_pool
            ):
                candidate_pool.append(ordered_candidate)

            scored_candidates: List[Tuple[float, float, Dict[str, Any]]] = []
            for item in candidate_pool:
                image_path = _resolve_keyframe_path(item)
                atomic_scores: List[float] = []
                if image_path and os.path.exists(image_path):
                    for question in questions:
                        cache_key = (image_path, question)
                        if cache_key not in vqa_cache:
                            vqa_cache[cache_key] = reranker_service.score_image(image_path, question)
                        atomic_scores.append(vqa_cache[cache_key])
                vqa_score = min(atomic_scores) if atomic_scores else 0.0
                retrieval_score = _score(item, int(item.get("_event_rank") or 1))
                scored_candidates.append((0.45 * retrieval_score + 0.55 * vqa_score, vqa_score, item))
            scored_candidates.sort(key=lambda row: row[0], reverse=True)
            scored_by_event.append(scored_candidates)

        ordered_beam: List[Tuple[float, List[Tuple[int, Dict[str, Any], float]]]] = []
        if scored_by_event and all(scored_by_event):
            ordered_beam = [
                (combined, [(0, item, vqa_score)])
                for combined, vqa_score, item in scored_by_event[0]
            ]
            for event_idx, scored_candidates in enumerate(scored_by_event[1:], start=1):
                expanded: List[Tuple[float, List[Tuple[int, Dict[str, Any], float]]]] = []
                for accumulated, sequence in ordered_beam:
                    last_order = _frame_order(sequence[-1][1])
                    for combined, vqa_score, item in scored_candidates:
                        if _frame_order(item) <= last_order:
                            continue
                        expanded.append((accumulated + combined, sequence + [(event_idx, item, vqa_score)]))
                expanded.sort(key=lambda row: row[0], reverse=True)
                ordered_beam = expanded[:40]
                if not ordered_beam:
                    break

        if ordered_beam:
            validated_evidence = ordered_beam[0][1]
            ordered = True
        else:
            validated_evidence = [
                (event_idx, scored_candidates[0][2], scored_candidates[0][1])
                for event_idx, scored_candidates in enumerate(scored_by_event)
                if scored_candidates
            ]
            ordered = False

        vqa_scores = [score for _, _, score in validated_evidence]
        validated_coverage = sum(score >= vqa_threshold for score in vqa_scores)
        average_vqa = sum(vqa_scores) / len(vqa_scores) if vqa_scores else 0.0
        record["evidence"] = validated_evidence
        record["ordered"] = ordered
        record["validated_coverage"] = validated_coverage
        record["average_vqa"] = average_vqa
        record["video_score"] = (
            (record["coverage"] / len(events)) * 4.0
            + record["average_retrieval"]
            + (validated_coverage / len(events)) * 2.0
            + average_vqa * 2.0
            + (1.5 if ordered else 0.0)
        )
    selected_videos.sort(
        key=lambda row: (row["ordered"], row["validated_coverage"], row["video_score"]),
        reverse=True,
    )
    fully_validated = [
        record for record in selected_videos
        if record["ordered"] and record["validated_coverage"] == len(events)
    ]
    final_videos = fully_validated or selected_videos

    frames: List[Dict[str, Any]] = []
    video_results: List[Dict[str, Any]] = []
    seen_content = set()
    for record in final_videos[:requested_videos]:
        evidence_summary = [
            {
                "event_index": event_idx,
                "event_query": events[event_idx],
                "frame_id": _first(item, "frame_id", "frame_key", "global_frame_id"),
                "timestamp": _timestamp(item),
                "vqa_score": round(float(vqa_score), 6),
            }
            for event_idx, item, vqa_score in record["evidence"]
        ]
        video_results.append({
            "video_id": record["video_id"],
            "coverage": record["coverage"],
            "event_count": len(events),
            "validated_coverage": record["validated_coverage"],
            "ordered": record["ordered"],
            "video_score": round(float(record["video_score"]), 6),
            "fully_validated": record["validated_coverage"] == len(events) and record["ordered"],
            "evidence_frames": evidence_summary,
        })
        for event_idx, item, vqa_score in record["evidence"]:
            content_key = _image_content_key(item)
            if content_key in seen_content:
                continue
            seen_content.add(content_key)
            enriched = dict(item)
            enriched.update({
                "event_index": event_idx,
                "event_query": events[event_idx],
                "event_query_en": queries[event_idx]["query_en"],
                "event_vqa_score": round(float(vqa_score), 6),
                "video_score": round(float(record["video_score"]), 6),
                "event_coverage": record["coverage"],
                "validated_coverage": record["validated_coverage"],
                "event_count": len(events),
                "temporal_ordered": record["ordered"],
                "fully_validated": record["validated_coverage"] == len(events) and record["ordered"],
                "evidence_frames": evidence_summary,
                "deep_sources": ["visual", "video_aggregation", "vqa"],
                "deep_queries": [events[event_idx]],
                "reason": (
                    f"Video {record['video_id']} covers {record['coverage']}/{len(events)} events; "
                    f"VQA validated {record['validated_coverage']}/{len(events)}."
                ),
            })
            frames.append(enriched)
            if len(frames) >= topk:
                break
        if len(frames) >= topk:
            break

    return {
        "answer": (
            (f"Tim thay {len(fully_validated)} video dat du {len(events)}/{len(events)} su kien va dung thu tu; "
             if fully_validated else
             f"Chua co video nao dat du {len(events)}/{len(events)} su kien; dang tra cac ung vien mot phan de kiem tra; ")
            + f"chon {len(video_results)} video va {len(frames)} keyframe bang chung."
        ),
        "queries_used": queries,
        "frames": frames,
        "video_results": video_results,
        "total_candidates": total_candidates,
    }

def deep_keyframe_search(prompt: str, topk: int = 20, per_query: int = 30) -> Dict[str, Any]:
    prompt = _clean_text(prompt)
    if not prompt:
        return {
            "answer": "Prompt rong, khong the tim sau.",
            "queries_used": [],
            "frames": [],
            "total_candidates": 0,
        }

    base = _extract_search_description(prompt)
    events = _temporal_events(base) if _has_phrase(base, TEMPORAL_HINTS) else []
    if len(events) >= 2:
        return _event_video_search(events, topk=topk, per_query=per_query)

    queries = expand_queries(prompt)
    query_results = [(query, _run_query(query, per_query)) for query in queries]
    ranked = _merge_results(query_results)
    validated_count = _rerank_with_vqa(ranked, queries)
    frames = _dedup_ranked(ranked, topk=topk)

    return {
        "answer": (f"Da thu {len(queries)} huong tim, xac thuc anh cho {validated_count} ung vien, "
                   f"va chon {len(frames)} keyframe it trung lap nhat."),
        "queries_used": queries,
        "frames": frames,
        "total_candidates": len(ranked),
    }