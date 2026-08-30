import re
from typing import Any, Dict, List

import logging
from langchain_core.tools import tool

from src.services.user_service import (
    GetImageDataTrakeSearch,
    getImageDataQAndASearch,
    getImageDataSingleTextSearch,
    getTextSearchASR,
    getTextSearchOCR,
)

logger = logging.getLogger(__name__)

_MAX_TOOL_RESULTS = 6
_MAX_TEXT_LEN = 220


def _short_text(value: Any, limit: int = _MAX_TEXT_LEN) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return f"{text[:limit]}..."


def _pick_first(source: Dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = source.get(key)
        if value not in (None, "", [], {}):
            return value
    return None


def _compact_result(res: Dict[str, Any]) -> Dict[str, Any]:
    compact: Dict[str, Any] = {}
    for key in ("video_id", "video_key", "frame_id", "frame_key", "frame_name", "timestamp", "score", "answer"):
        value = res.get(key)
        if value not in (None, "", [], {}):
            compact[key] = _short_text(value) if isinstance(value, str) else value

    watch_url = _pick_first(res, "watch_url", "youtube_url", "video_url", "link", "url")
    if watch_url:
        compact["watch_url"] = watch_url

    ocr_text = _pick_first(res, "ocr_text", "ocr")
    if ocr_text:
        compact["ocr_text"] = _short_text(ocr_text)

    objects = _pick_first(res, "od_classes", "objects", "detected_objects")
    if isinstance(objects, list) and objects:
        compact["objects"] = [_short_text(obj, 40) for obj in objects[:8]]

    media_info = res.get("media_info")
    if isinstance(media_info, dict):
        title = media_info.get("title")
        if title:
            compact["title"] = _short_text(title, 120)
        media_watch_url = media_info.get("watch_url")
        if media_watch_url and "watch_url" not in compact:
            compact["watch_url"] = media_watch_url

    error = res.get("error")
    if error:
        compact["error"] = _short_text(error, 240)

    # TRAKE returns nested frames. Keep locators, never base64 images.
    frames = res.get("frames")
    if isinstance(frames, list) and frames:
        keyframes = []
        for frame in frames[:8]:
            if not isinstance(frame, dict):
                continue
            keyframe = {
                key: frame[key]
                for key in ("folder_key", "video_key", "frame_key", "frame_name", "timestamp")
                if frame.get(key) not in (None, "", [], {})
            }
            if keyframe:
                keyframes.append(keyframe)
        if keyframes:
            compact["video_id"] = compact.get("video_id") or keyframes[0].get("video_key")
            compact["keyframes"] = keyframes

    return compact


def _compact_results(results: List[Dict[str, Any]], limit: int = _MAX_TOOL_RESULTS) -> List[Dict[str, Any]]:
    return [_compact_result(res) for res in (results or [])[:limit]]




def _temporal_queries(query: str) -> List[Dict[str, str]]:
    """Convert a natural-language temporal description into ordered events."""
    cleaned = " ".join(str(query or "").split()).strip(" .")
    if not cleaned:
        return []

    # "A, before that B" means B happened before A.
    before_pattern = r"\btr\u01b0\u1edbc \u0111\u00f3\b"
    before_parts = re.split(before_pattern, cleaned, maxsplit=1, flags=re.IGNORECASE)
    if len(before_parts) == 2 and all(part.strip(" ,.;:") for part in before_parts):
        current, previous = before_parts
        return [
            {"query": previous.strip(" ,.;:")},
            {"query": current.strip(" ,.;:")},
        ]

    ordered_connectors = (
        r"\b(?:sau \u0111\u00f3|ti\u1ebfp theo|r\u1ed3i|"
        r"sau c\u00f9ng|cu\u1ed1i c\u00f9ng)\b"
    )
    parts = re.split(ordered_connectors, cleaned, flags=re.IGNORECASE)
    events = [part.strip(" ,.;:") for part in parts if part.strip(" ,.;:")]
    return [{"query": event} for event in events]

@tool
def vector_search_tool(query: str, top_k: int = 20) -> List[Dict[str, Any]]:
    """
    Tim kiem semantic vector (Faiss + BEiT-3) dua tren text query.
    Su dung tool nay khi cau hoi yeu cau tim kiem noi dung hinh anh, mau sac, hanh dong chung chung trong video.
    Vi du: 'nguoi dan ong mac ao do dang chay', 'chiec xe mau xanh'.
    """
    logger.info("Agent called vector_search_tool with query: %s", query)
    try:
        results = getImageDataSingleTextSearch(query, top_k)
        return _compact_results(results)
    except Exception as e:
        return [{"error": _short_text(e, 240)}]


@tool
def ocr_search_tool(query: str, top_k: int = 20) -> List[Dict[str, Any]]:
    """
    Tim kiem van ban xuat hien tren man hinh video (OCR - Optical Character Recognition).
    Su dung tool nay khi nguoi dung tim kiem cac tu ngu cu the, van ban, bien bao, logo co trong khung hinh.
    Vi du: 'chu tren bang', 'bien so xe', 'dong chu tin tuc'.
    """
    logger.info("Agent called ocr_search_tool with query: %s", query)
    try:
        results = getTextSearchOCR(query, top_k)
        return _compact_results(results)
    except Exception as e:
        return [{"error": _short_text(e, 240)}]


@tool
def asr_search_tool(query: str, top_k: int = 20) -> List[Dict[str, Any]]:
    """
    Tim kiem giong noi/loi thoai trong video (ASR - Automatic Speech Recognition).
    Su dung tool nay khi cau hoi lien quan den loi noi, hoi thoai, nguoi ta dang noi gi.
    Vi du: 'ho dang noi chuyen ve gi', 'MC noi tu khoa'.
    """
    logger.info("Agent called asr_search_tool with query: %s", query)
    try:
        results = getTextSearchASR(query, top_k)
        return _compact_results(results)
    except Exception as e:
        return [{"error": _short_text(e, 240)}]


@tool
def temporal_search_tool(query: str, top_k: int = 20) -> List[Dict[str, Any]]:
    """
    Tim kiem chuoi su kien theo thoi gian (Temporal Search / TRAKE).
    Su dung tool nay khi cau hoi co yeu to thoi gian, truoc/sau, hoac mot chuoi cac hanh dong lien tiep.
    Vi du: 'nguoi dan ong chay sau do nga', 'truoc tien..., sau do...'.
    """
    logger.info("Agent called temporal_search_tool with query: %s", query)
    try:
        events = _temporal_queries(query)
        if not events:
            return [{"error": "Temporal query is empty."}]
        logger.info("Temporal query parsed into %d ordered event(s): %s", len(events), events)
        results = GetImageDataTrakeSearch(events, top_results=top_k)
        return _compact_results(results)
    except Exception as e:
        return [{"error": _short_text(e, 240)}]


@tool
def video_qa_tool(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Tim kiem ket hop tra loi cau hoi truc quan (Video QA) su dung mo hinh VQA.
    Su dung tool nay khi cau hoi yeu cau giai thich chi tiet ve mot phan canh,
    dem so luong, hoac tra loi mot cau hoi cu the dua tren hinh anh.
    Vi du: 'Co bao nhieu nguoi trong khung canh nay?', 'Nguoi dan ong dang cam gi?'.
    Luu y: Tool nay chay cham, chi nen goi khi thuc su can thiet.
    """
    logger.info("Agent called video_qa_tool with query: %s", query)
    try:
        results = getImageDataQAndASearch(query, top_k)
        return _compact_results(results)
    except Exception as e:
        return [{"error": _short_text(e, 240)}]


agent_tools = [
    vector_search_tool,
    ocr_search_tool,
    asr_search_tool,
    temporal_search_tool,
    video_qa_tool,
]

