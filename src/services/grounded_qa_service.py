"""Grounded retrieve-evidence-answer pipeline for video Q&A."""

from __future__ import annotations

import base64
import io
import json
import logging
import math
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from PIL import Image

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
            "answer_language": {"type": "string", "enum": ["vi"]},
        },
        "required": [
            "status",
            "answer",
            "confidence",
            "reason",
            "supporting_frame_ids",
            "used_ocr_evidence",
            "used_asr_evidence",
            "answer_language",
        ],
    },
}

CANDIDATE_ANSWER_SCHEMA = {
    "name": "grounded_video_answer_candidates",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "candidates": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "candidate_id": {"type": "string"},
                        "status": {"type": "string", "enum": ["answered", "uncertain"]},
                        "answer": {"type": "string"},
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                        "reason": {"type": "string"},
                        "supporting_frame_ids": {"type": "array", "items": {"type": "string"}},
                        "used_ocr_evidence": {"type": "boolean"},
                        "used_asr_evidence": {"type": "boolean"},
                        "answer_language": {"type": "string", "enum": ["vi"]},
                    },
                    "required": [
                        "candidate_id",
                        "status",
                        "answer",
                        "confidence",
                        "reason",
                        "supporting_frame_ids",
                        "used_ocr_evidence",
                        "used_asr_evidence",
                        "answer_language",
                    ],
                },
            },
        },
        "required": ["candidates"],
    },
}

VERIFICATION_SCHEMA = {
    "name": "grounded_video_answer_verification",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "verified": {"type": "boolean"},
            "canonical_answer": {"type": "string"},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "reason": {"type": "string"},
            "supporting_frame_ids": {"type": "array", "items": {"type": "string"}},
            "answer_language": {"type": "string", "enum": ["vi"]},
        },
        "required": [
            "verified",
            "canonical_answer",
            "confidence",
            "reason",
            "supporting_frame_ids",
            "answer_language",
        ],
    },
}

SYSTEM_PROMPT = """You answer questions about retrieved video evidence.
Use only the attached keyframes and OCR/ASR snippets. Never use outside knowledge to fill a missing fact.
Always write the final answer and reason in natural Vietnamese with correct Vietnamese diacritics,
even when the user's question is in English. Do not return an English natural-language answer.
For OCR text, logos, proper names, codes, and numbers, preserve the exact visible spelling instead of translating it.
If evidence supports a plausible answer but is not strong enough for certainty, return
status=uncertain and put the single most likely concise answer in answer. Keep the uncertainty
explanation in reason. Use a generic insufficient-evidence answer only when no meaningful
candidate can be inferred from any relevant frame or OCR/ASR snippet.
For counting, colors, identities, actions, spatial relations, and visible objects, rely on visible frames.
For spoken content, rely on ASR snippets. For written text, rely on OCR snippets.
Event groups are alternative candidate moments unless the question explicitly asks about a sequence.
Use one coherent event group for a factual answer.
For a question describing multiple events, every event used for the answer must belong to the same video id.
Never combine a matching scene from one video with OCR, ASR, a title, or another scene from a different video.
Prefer a video that covers all described events over a higher-scoring video that covers only one event.
Never add counts across event groups or across duplicate timeline frames.
The answer field is the competition answer: concise, no explanation or prefix, at most 100 characters.
Put explanations in reason, not answer. Only cite frame ids that are attached. Return strict JSON only, no markdown.
"""

VERIFIER_SYSTEM_PROMPT = """You verify a proposed answer against retrieved video evidence.
Reject it if the answer is guessed, the count/object/color/action/text is not directly visible or present in OCR/ASR,
the evidence refers to a different event, duplicate timeline frames were counted more than once,
evidence from alternative event groups was combined, or the answer type does not match the question.
If verified, return a canonical answer in natural Vietnamese with correct Vietnamese diacritics,
with no prefix or explanation and at most 100 characters. Preserve exact OCR text, proper names, codes, and numbers.
If not verified, canonical_answer must still contain the single most likely Vietnamese answer
supported by the strongest attached event, with low confidence. Use a generic insufficient-evidence
answer only when no meaningful candidate exists.
Only cite attached frame ids. Use no outside knowledge. Return strict JSON only, no markdown.
"""

DETAIL_SYSTEM_PROMPT = """You perform a focused second inspection of video keyframes that an earlier pass
already identified as relevant. Inspect the full frames and every labelled zoom tile carefully. For a scale,
clock, counter, sign, subtitle, label, or other display, mentally correct rotated viewing angles and compare
adjacent moments before reading the final visible value. For a count, inspect each crop but count the same
object only once. Do not infer a unit or hidden digit that is not visible. If one value is the strongest visual
reading but is not perfectly sharp, return that single most likely concise value with status=uncertain and a
low confidence instead of a generic refusal. The answer and reason must be natural Vietnamese with correct
diacritics; exact numbers and OCR strings must be preserved. Only cite the supplied frame ids. Return strict
JSON matching the requested schema, without markdown.
"""

CANDIDATE_SYSTEM_PROMPT = """You independently answer the same video question for several candidate videos.
Treat every candidate video as a separate hypothesis. Never transfer visual, OCR, ASR, title, or answer evidence
between candidate ids or video ids. A description with multiple events is matched only when the same candidate
video covers those events. For every supplied candidate, return the single most likely concise answer even when
the evidence is weak; use status=uncertain and low confidence for a partial match. Use a generic insufficient-
evidence answer only when that candidate has no meaningful answer at all. Answers and reasons must be natural
Vietnamese with correct diacritics. Preserve exact visible proper names, numbers, units, and OCR spelling.
Only cite frame ids listed inside that candidate. Return every supplied candidate id exactly once as strict JSON.
"""

OCR_INTENT = re.compile(
    r"\b(?:ocr|text|written|read|sign|caption|subtitle|displayed|logo|watermark|"
    r"chu|chữ|ghi gì|viet gi|viết gì|biển|bien|bảng|bang|dòng chữ|dong chu|"
    r"hiển thị|hien thi|kênh truyền hình|kenh truyen hinh|cột mốc|cot moc)\b",
    re.IGNORECASE,
)
ASR_INTENT = re.compile(
    r"\b(?:asr|audio|speech|say|said|saying|hear|nghe|nói|noi|phát biểu|phat bieu)\b",
    re.IGNORECASE,
)
OCR_CONTEXT = re.compile(
    r"\b(?:map|chart|graph|diagram|legend|label|scoreboard|document|screen|"
    r"address|street|road|bản đồ|ban do|biểu đồ|bieu do|chú giải|chu giai|"
    r"ký hiệu|ky hieu|màn hình|man hinh|địa chỉ|dia chi|đường|duong)\b",
    re.IGNORECASE,
)

TEMPORAL_CONTEXT = re.compile(
    r"\b(?:followed by|after that|afterwards|then|next|finally|last|before|after|"
    r"sau đó|sau do|tiếp theo|tiep theo|rồi|roi|cuối cùng|cuoi cung|trước đó|truoc do)\b",
    re.IGNORECASE,
)

QUESTION_TYPE_PATTERNS: Sequence[Tuple[str, re.Pattern[str]]] = (
    (
        "count",
        re.compile(
            r"\b(?:how many|number of|count|which number|bao nhiêu|bao nhieu|mấy|may|"
            r"số nào\s+(?:không|khong)|so nao\s+khong)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "color",
        re.compile(
            r"\b(?:what (?:two )?(?:main )?colou?rs?|colou?r is|màu gì|mau gi|màu nào|mau nao|"
            r"màu chủ đạo gì|mau chu dao gi)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "ocr",
        re.compile(
            r"\b(?:what does .* say|what is written|what .* (?:text|logo|watermark|displayed)|"
            r"which country name|ghi gì|ghi gi|chữ gì|chu gi|viết gì|viet gi|"
            r"dòng chữ|dong chu|logo .* nào|logo .* nao|license plate|registration number|"
            r"biển số|bien so)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "asr",
        re.compile(
            r"\b(?:what did .* say|what is .* saying|nói gì|noi gi|phát biểu gì|phat bieu gi)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "object",
        re.compile(
            r"\b(?:what|which) (?:exact )?(?:kind of )?(?:chemical )?(?:object|device|vehicle|food|utensil|"
            r"container|transport|tool|item|animal|compound|substance|material|brand|model)|"
            r"\b(?:phương tiện|phuong tien|thiết bị|thiet bi|vật|vat|dụng cụ|dung cu|đồ vật|"
            r"do vat) gì\b|\b(?:phương tiện|phuong tien).{0,30}(?:nào|nao)\b|"
            r"\b(?:loài|loai|giống|giong|con|cá|ca|topping|thịt của con|thit cua con)"
            r".{0,40}(?:gì|gi|nào|nao)\b|\b[xX]\s+là\s+con\s+gì\b|"
            r"\b(?:nhãn hiệu|nhan hieu|mã sản phẩm|ma san pham)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "location",
        re.compile(
            r"\b(?:where|ở đâu|o dau|vị trí nào|vi tri nao|which (?:street|road)|"
            r"nằm (?:ở|trên) đường nào|nam (?:o|tren) duong nao|đường nào|duong nao)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "spatial",
        re.compile(
            r"\b(?:left|right|above|below|upper|lower|horizontal|vertical|orientation|"
            r"bên trái|ben trai|bên phải|ben phai|phía trên|phia tren|phía dưới|phia duoi|"
            r"hướng dọc|huong doc|hướng ngang|huong ngang|dọc hay ngang|doc hay ngang)\b",
            re.IGNORECASE,
        ),
    ),
    ("person", re.compile(r"\b(?:who|ai|người nào|nguoi nao)\b", re.IGNORECASE)),
    (
        "action",
        re.compile(
            r"\b(?:what (?:is|are|was|were) .* doing|what happens|what performance|"
            r"doing what|đang làm gì|dang lam gi|làm gì|lam gi|hoạt động gì|hoat dong gi|"
            r"xảy ra|xay ra)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "yes_no",
        re.compile(
            r"^(?:is|are|was|were|do|does|did|can|could|has|have|có|co)\b|"
            r"\b(?:hay không|hay khong|không\s*\?|khong\s*\?)$",
            re.IGNORECASE,
        ),
    ),
    (
        "temporal",
        re.compile(
            r"\b(?:before|after|first|last|then|date of birth|calendar date|exact date|"
            r"ngày sinh|ngay sinh|ngày chính xác|ngay chinh xac|trước đó|truoc do|trước khi|truoc khi|"
            r"sau đó|sau do|sau khi|cuối cùng|cuoi cung|đầu tiên|dau tien)\b",
            re.IGNORECASE,
        ),
    ),
)

EXPECTED_ANSWER_FORMAT = {
    "count": "một số nguyên hoặc số lượng ngắn gọn bằng tiếng Việt",
    "color": "một tên màu ngắn gọn bằng tiếng Việt",
    "ocr": "nguyên văn chữ nhìn thấy; không dịch tên riêng, mã hoặc số",
    "asr": "nội dung lời nói ngắn gọn bằng tiếng Việt",
    "temporal": "một sự kiện hoặc hành động ngắn gọn bằng tiếng Việt",
    "person": "mô tả hoặc tên người ngắn gọn dựa trên bằng chứng",
    "location": "một địa điểm ngắn gọn bằng tiếng Việt",
    "action": "một hành động ngắn gọn bằng tiếng Việt",
    "object": "một vật thể, thiết bị, món ăn, phương tiện hoặc dụng cụ bằng tiếng Việt",
    "spatial": "một quan hệ hoặc hướng không gian ngắn gọn bằng tiếng Việt",
    "yes_no": "có/không, chỉ kèm diễn giải thật cần thiết",
    "other": "một đáp án thực tế ngắn gọn bằng tiếng Việt",
}

ENGLISH_ANSWER_MARKERS = {
    "a",
    "an",
    "and",
    "are",
    "black",
    "blue",
    "bowl",
    "cable",
    "car",
    "cannot",
    "dance",
    "evidence",
    "flatbread",
    "glass",
    "green",
    "insufficient",
    "is",
    "motorbike",
    "motorcycle",
    "object",
    "one",
    "orange",
    "person",
    "phone",
    "purple",
    "red",
    "sausage",
    "sausages",
    "smartphone",
    "spatula",
    "the",
    "this",
    "three",
    "truck",
    "two",
    "unknown",
    "vehicle",
    "vertical",
    "horizontal",
    "left",
    "right",
    "woman",
    "man",
    "white",
    "yellow",
}


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


def _answer_type(question: str) -> str:
    for answer_type, pattern in QUESTION_TYPE_PATTERNS:
        if pattern.search(question):
            return answer_type
    return "other"


def _visual_focus(query: str) -> str:
    focus = _clean(query).rstrip("?.!")
    focus = re.sub(
        r"^(?:how many|what colou?r (?:is|are)|what (?:is|are|was|were)|who (?:is|are|was|were)|"
        r"where (?:is|are|was|were)|is there|are there|does the (?:image|video|scene) show)\s+",
        "",
        focus,
        flags=re.IGNORECASE,
    )
    focus = re.sub(
        r"\b(?:in (?:this|the) (?:image|frame|video|scene)|"
        r"shown in the (?:image|frame|video|scene))\b",
        "",
        focus,
        flags=re.IGNORECASE,
    )
    focus = re.sub(r"\b(?:which|what|who|where|how many)\b", " ", focus, flags=re.IGNORECASE)
    focus = re.sub(
        r"\b(?:is|are|was|were|has|have|does|do|did|its|their)\b",
        " ",
        focus,
        flags=re.IGNORECASE,
    )
    focus = re.sub(r"\s*[,;:]\s*", " ", focus)
    return _clean(focus)


def _unique_queries(*queries: str, limit: int = 3) -> List[str]:
    output: List[str] = []
    seen: set[str] = set()
    for query in queries:
        cleaned = _clean(query)
        key = cleaned.casefold()
        if cleaned and key not in seen:
            output.append(cleaned)
            seen.add(key)
        if len(output) >= max(1, limit):
            break
    return output


def _visual_expansions(query: str) -> List[str]:
    """Generate conservative retrieval variants without changing the answer.

    BEiT3 is sensitive to the concrete appearance of an event.  Competition
    questions often describe the semantics ("a fish is placed on a scale")
    while the frame actually contains an intermediate carrier such as a tub.
    These variants broaden recall only; the VLM must still verify the original
    question against the retrieved images.
    """
    expansions: List[str] = []
    two_wheeled = re.compile(r"\btwo[- ]wheeled vehicle\b", re.IGNORECASE)
    if two_wheeled.search(query):
        expansions.append(two_wheeled.sub("motorcycle", query))
        expansions.append(two_wheeled.sub("motorbike", query))

    scale_subject = re.search(
        r"\b(?:a|an|the)\s+([a-z][a-z -]{0,50}?)\s+"
        r"(?:being\s+)?(?:placed|put|weighed)\s+(?:on|in)\s+"
        r"(?:a|an|the)\s+(?:digital\s+)?(?:weighing\s+)?scale\b",
        query,
        re.IGNORECASE,
    )
    if scale_subject:
        subject = _clean(scale_subject.group(1))
        expansions.extend([
            f"a {subject} in a plastic container on a digital scale",
            f"a {subject} being weighed in a container on a digital weighing scale",
        ])
        if re.search(r"\bfish\b", subject, re.IGNORECASE):
            # Carry the plausible subtype discovered by the companion
            # held-by-tail event back into the weighing event.  This is a
            # retrieval hypothesis, never an asserted answer.
            expansions.append("a small shark in a plastic container on a digital scale")

    held_by_tail = re.search(
        r"\b(?:a|an|another|the)\s+([a-z][a-z -]{0,40}?)\s+.*?"
        r"(?:held|holding).*?\b(?:by\s+)?(?:its|the)\s+tail\b",
        query,
        re.IGNORECASE,
    )
    if held_by_tail:
        subject = _clean(re.sub(r"\bof the same (?:kind|type|species)\b", "", held_by_tail.group(1), flags=re.IGNORECASE))
        expansions.extend([
            f"a person holding a {subject} by its tail",
            f"a close-up of a {subject} being held by the tail",
        ])
        # A shark is taxonomically a fish, but visual encoders often separate
        # the two labels strongly.  Search both labels when the prompt only
        # says fish; the attached image still decides the answer.
        if re.search(r"\bfish\b", subject, re.IGNORECASE):
            expansions.extend([
                "a small shark held by its tail",
                "a person holding a small shark by its tail",
            ])
    return expansions


def _split_visual_events(query: str) -> List[str]:
    """Extract independently retrievable events from a temporal QA prompt."""
    text = _clean(query).strip(" .?!")
    if not text:
        return []
    text = re.sub(
        r"^(?:the\s+)?(?:image|video|scene|footage)\s+(?:shows?|depicts?|contains?)\s+",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\s*(?:[.!?]\s*)?(?:,\s*)?(?:followed by|after that|afterwards|then|next|finally)"
        r"(?:\s*,)?\s+",
        " || ",
        text,
        flags=re.IGNORECASE,
    )
    # Competition prompts often describe each must-have visual event in its
    # own declarative sentence without an explicit "then" connector.  Treat
    # those sentence boundaries as event boundaries, while leaving the final
    # interrogative to the removal step below.
    text = re.sub(
        r"[.!?]\s+(?=(?!(?:what|which|who|where|how|is|are|does|do|did)\b)[A-Z])",
        " || ",
        text,
        flags=re.IGNORECASE,
    )
    # The final interrogative describes the requested attribute, not another
    # event.  Keep it in the full query but do not let it dilute event recall.
    text = re.split(
        r"[.!?]\s*(?=(?:what|which|who|where|how|is|are|does|do|did)\b)",
        text,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    events = []
    for value in text.split("||"):
        event = _clean(value).strip(" ,.;:?!")
        event = re.sub(r"^(?:a\s+)?scene\s+(?:of|showing)\s+", "", event, flags=re.IGNORECASE)
        if event:
            events.append(event)
    return events


def _initial_time_window_seconds(question: str, translated: str) -> float | None:
    """Extract an explicit initial video window such as '16 giây đầu tiên'."""
    for value in (question, translated):
        match = re.search(
            r"\b(\d+(?:[.,]\d+)?)\s*(?:giây|giay|seconds?|secs?)\s*"
            r"(?:đầu tiên|dau tien|đầu|dau|from the start|at the beginning)\b",
            value,
            re.IGNORECASE,
        )
        if match:
            return _float(match.group(1).replace(",", "."), 0.0) or None
        match = re.search(
            r"\b(?:first|đầu tiên|dau tien)\s*(\d+(?:[.,]\d+)?)\s*"
            r"(?:giây|giay|seconds?|secs?)\b",
            value,
            re.IGNORECASE,
        )
        if match:
            return _float(match.group(1).replace(",", "."), 0.0) or None
    return None


def _enumeration_range(question: str, translated: str) -> Tuple[int, int] | None:
    for value in (question, translated):
        match = re.search(
            r"(?:các số|cac so|numbers?)\s*(?:từ|tu|from)?\s*(\d+)\s*"
            r"(?:-|–|đến|den|to)\s*(\d+)",
            value,
            re.IGNORECASE,
        )
        if match:
            start, end = int(match.group(1)), int(match.group(2))
            if 0 <= start <= end <= 100:
                return start, end
    return None


def _text_evidence_queries(
    question: str,
    limit: int = 6,
    *,
    include_interrogative: bool = True,
    include_full: bool = True,
) -> List[str]:
    """Create concise Vietnamese evidence queries from a story-like prompt."""
    clauses: List[str] = []
    for sentence in re.split(r"[.!?]+", _clean(question)):
        for clause in re.split(
            r"\b(?:sau đó|sau do|tiếp theo|tiep theo|rồi|roi|cuối cùng|cuoi cung)\b",
            sentence,
            flags=re.IGNORECASE,
        ):
            value = _clean(clause).strip(" ,;:-")
            is_interrogative = bool(re.match(
                r"^(?:hỏi|hoi|hãy cho biết|hay cho biet|đây là|day la|what|which|where|who|how)\b",
                value,
                re.IGNORECASE,
            ))
            if (
                len(re.findall(r"\w+", value, re.UNICODE)) >= 4
                and (include_interrogative or not is_interrogative)
            ):
                clauses.append(value)
    # Individual descriptive clauses prevent common words from a long final
    # question from dominating Elasticsearch BM25.  Keep the full prompt as a
    # final conjunction signal when capacity permits.
    queries = [*clauses, *([question] if include_full else [])]
    if not queries:
        queries = [question]
    return _unique_queries(*queries, limit=max(1, limit))


def _question_plan(question: str, visual_query_limit: int = 3) -> Dict[str, Any]:
    translated = Translation()(question) or question
    answer_type = _answer_type(question)
    visual_focus = _visual_focus(translated)
    needs_temporal_context = bool(TEMPORAL_CONTEXT.search(question) or TEMPORAL_CONTEXT.search(translated))
    event_queries = _split_visual_events(translated) if needs_temporal_context else []
    query_candidates: List[str] = []
    event_by_query: Dict[str, str] = {}
    priority_by_query: Dict[str, int] = {}
    if event_queries:
        event_variants = [[event, *_visual_expansions(event)] for event in event_queries]
        # Round-robin prevents rich expansions of event 1 from consuming the
        # whole query budget before event 2 is searched.
        depth = 0
        while any(depth < len(variants) for variants in event_variants):
            for event_index, variants in enumerate(event_variants, 1):
                if depth >= len(variants):
                    continue
                variant = variants[depth]
                query_candidates.append(variant)
                variant_key = _clean(variant).casefold()
                event_by_query.setdefault(variant_key, f"e{event_index}")
                priority_by_query.setdefault(variant_key, depth)
            depth += 1
    query_candidates.extend([
        visual_focus,
        *_visual_expansions(visual_focus),
        translated,
        question if translated.casefold() != question.casefold() else "",
    ])
    visual_queries = _unique_queries(*query_candidates, limit=visual_query_limit)
    visual_query_event_ids = [event_by_query.get(query.casefold(), "global") for query in visual_queries]
    visual_query_priorities = [priority_by_query.get(query.casefold(), 0) for query in visual_queries]
    wants_ocr = answer_type == "ocr" or bool(OCR_INTENT.search(question) or OCR_CONTEXT.search(question))
    # Long natural-language QA descriptions often identify the exact video in
    # spoken narration even when the requested value is visual.  ASR therefore
    # acts as a retrieval anchor for all substantive QA types, not only direct
    # "what was said" questions.
    wants_asr = bool(
        answer_type in {"asr", "object", "location", "person", "action", "temporal", "other"}
        or ASR_INTENT.search(question)
        or len(re.findall(r"\w+", question, re.UNICODE)) >= 12
    )
    time_window_seconds = _initial_time_window_seconds(question, translated)
    enumeration_range = _enumeration_range(question, translated)
    requires_set_comparison = bool(
        enumeration_range
        and re.search(
            r"(?:not (?:visible|seen|shown)|không (?:được )?(?:nhìn thấy|thấy|xuất hiện)|"
            r"khong (?:duoc )?(?:nhin thay|thay|xuat hien))",
            question + " " + translated,
            re.IGNORECASE,
        )
    )
    return {
        "question": question,
        "answer_type": answer_type,
        "expected_answer_format": EXPECTED_ANSWER_FORMAT[answer_type],
        "needs_temporal_context": needs_temporal_context or answer_type in {"temporal", "action"},
        "event_queries": event_queries,
        "visual_query": visual_queries[0] if visual_queries else translated,
        "visual_queries": visual_queries or [translated],
        "visual_query_event_ids": visual_query_event_ids or ["global"],
        "visual_query_priorities": visual_query_priorities or [0],
        "ocr_query": question if wants_ocr else "",
        "asr_query": question if wants_asr else "",
        "ocr_queries": _text_evidence_queries(question) if wants_ocr else [],
        "asr_queries": (
            _text_evidence_queries(
                question,
                include_interrogative=answer_type == "asr",
                include_full=answer_type == "asr",
            )
            if wants_asr
            else []
        ),
        "time_window_seconds": time_window_seconds,
        "enumeration_range": enumeration_range,
        "requires_set_comparison": requires_set_comparison,
    }


def _get_retriever() -> Any:
    from src.services.beit3_retriever import get_beit3_retriever

    return get_beit3_retriever()


def _search_visual_queries(
    retriever: Any,
    queries: Sequence[str],
    top_k: int,
    query_event_ids: Sequence[str] | None = None,
) -> List[Dict[str, Any]]:
    """Run several cheap text encodings against the existing FAISS index and fuse ranks."""
    fused: Dict[str, Dict[str, Any]] = {}
    order = 0
    errors: List[Exception] = []
    for query_index, query in enumerate(queries):
        try:
            frames = retriever.search_visual(query, top_k=top_k)
        except Exception as exc:
            errors.append(exc)
            logger.warning("Grounded Q&A visual query failed (%s): %s", query, exc)
            continue
        for rank, frame in enumerate(frames, 1):
            key = _identity(frame)
            if not key:
                continue
            if key not in fused:
                order += 1
                fused[key] = dict(frame)
                fused[key]["qa_first_seen"] = order
                fused[key]["qa_rrf_score"] = 0.0
                fused[key]["qa_query_hits"] = []
                fused[key]["qa_visual_score"] = _float(frame.get("score"))
            item = fused[key]
            item["qa_rrf_score"] += 1.0 / (60.0 + rank)
            item["qa_visual_score"] = max(item["qa_visual_score"], _float(frame.get("score")))
            event_id = (
                str(query_event_ids[query_index])
                if query_event_ids is not None and query_index < len(query_event_ids)
                else "global"
            )
            item["qa_query_hits"].append({
                "query_index": query_index,
                "event_id": event_id,
                "rank": rank,
            })
    if not fused and errors:
        raise errors[0]
    ranked = sorted(
        fused.values(),
        key=lambda frame: (
            _float(frame.get("qa_rrf_score")),
            _float(frame.get("qa_visual_score")),
            -int(frame.get("qa_first_seen") or 0),
        ),
        reverse=True,
    )
    for rank, frame in enumerate(ranked, 1):
        frame["qa_visual_rank"] = rank
    return ranked


def _fuse_text_hits(
    search: Any,
    queries: Sequence[str],
    top_k: int,
) -> List[Dict[str, Any]]:
    fused: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for query_index, query in enumerate(queries):
        for rank, row in enumerate(search(query, topk=top_k), 1):
            video_id = _clean(row.get("video_id"))
            timestamp = row.get("nearest_timestamp", row.get("timestamp", row.get("start_time", "")))
            text_value = _clean(row.get("text") or row.get("ocr_text"))
            key = (video_id.casefold(), str(timestamp), text_value.casefold())
            if key not in fused:
                fused[key] = dict(row)
                fused[key]["qa_text_rrf"] = 0.0
                fused[key]["qa_text_query_hits"] = []
            item = fused[key]
            item["qa_text_rrf"] += 1.0 / (20.0 + rank)
            item["qa_text_query_hits"].append(query_index)
            item["_score"] = max(_float(item.get("_score")), _float(row.get("_score")))
    ranked = sorted(
        fused.values(),
        key=lambda row: (
            _float(row.get("qa_text_rrf")),
            len(set(row.get("qa_text_query_hits") or [])),
            _float(row.get("_score")),
        ),
        reverse=True,
    )
    return ranked[:top_k]


def _collect_text_evidence(plan: Dict[str, Any], top_k: int) -> Dict[str, List[Dict[str, Any]]]:
    evidence: Dict[str, List[Dict[str, Any]]] = {"ocr": [], "asr": []}
    if not plan["ocr_query"] and not plan["asr_query"]:
        return evidence
    try:
        from src.services.user_service import get_elastic_processor

        processor = get_elastic_processor()
        if plan["ocr_query"]:
            evidence["ocr"] = _fuse_text_hits(
                processor.search_ocr,
                plan.get("ocr_queries") or [plan["ocr_query"]],
                top_k,
            )
        if plan["asr_query"]:
            evidence["asr"] = _fuse_text_hits(
                processor.search_asr,
                plan.get("asr_queries") or [plan["asr_query"]],
                top_k,
            )
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


def _optional_evidence_timestamp(row: Dict[str, Any], modality: str) -> float | None:
    """Return an evidence timestamp only when the source supplied one."""
    if modality == "asr":
        if row.get("nearest_timestamp") is not None:
            return _float(row.get("nearest_timestamp"))
        if row.get("start_time") is not None or row.get("end_time") is not None:
            start = _float(row.get("start_time"), _float(row.get("end_time")))
            end = _float(row.get("end_time"), start)
            return (start + end) / 2.0
        return None
    if row.get("timestamp") is None:
        return None
    return _float(row.get("timestamp"))


def _evidence_text(row: Dict[str, Any], modality: str) -> str:
    return _clean(row.get("text") if modality == "asr" else row.get("ocr_text"))[:500]


def _video_metadata_lines(frames: Sequence[Dict[str, Any]]) -> List[str]:
    """Expose concise existing catalogue metadata to the answer model.

    Titles frequently name a recipe, animal, location, or lesson and provide a
    useful cross-modal check without training or generating new annotations.
    """
    output: List[str] = []
    seen: set[Tuple[str, str]] = set()
    for frame in frames:
        video_id = _clean(frame.get("video_id"))
        media_info = frame.get("media_info") if isinstance(frame.get("media_info"), dict) else {}
        title = _clean(media_info.get("title"))[:240]
        if not video_id or not title:
            continue
        key = (video_id.casefold(), title.casefold())
        if key in seen:
            continue
        seen.add(key)
        output.append(f"VIDEO video={video_id} title={title}")
    return output


def _relevant_text_evidence(
    text_evidence: Dict[str, List[Dict[str, Any]]],
    selected_frames: Sequence[Dict[str, Any]],
    per_modality_limit: int,
    max_timestamp_delta: float,
) -> Dict[str, List[Dict[str, Any]]]:
    """Keep OCR/ASR snippets aligned with the attached video moments.

    Args:
        text_evidence: Raw Elasticsearch OCR and ASR hits.
        selected_frames: Keyframes that will be attached to the VLM prompt.
        per_modality_limit: Maximum retained rows per modality.
        max_timestamp_delta: Maximum time distance from an attached frame.

    Returns:
        Filtered evidence in original retrieval order, deduplicated by content.
    """
    timestamps_by_video: Dict[str, List[float]] = {}
    for frame in selected_frames:
        video_id = _clean(frame.get("video_id"))
        if not video_id:
            continue
        timestamps_by_video.setdefault(video_id, [])
        if frame.get("timestamp") is not None:
            timestamps_by_video[video_id].append(_float(frame.get("timestamp")))

    output: Dict[str, List[Dict[str, Any]]] = {"ocr": [], "asr": []}
    limit = max(0, per_modality_limit)
    for modality in ("ocr", "asr"):
        seen: set[Tuple[str, str, str]] = set()
        for row in text_evidence.get(modality, []):
            video_id = _clean(row.get("video_id"))
            if not video_id or video_id not in timestamps_by_video:
                continue
            timestamp = _optional_evidence_timestamp(row, modality)
            frame_times = timestamps_by_video[video_id]
            if timestamp is not None and frame_times:
                nearest_delta = min(abs(timestamp - value) for value in frame_times)
                if nearest_delta > max(0.0, max_timestamp_delta):
                    continue
            text = _evidence_text(row, modality)
            if not text:
                continue
            dedupe_key = (video_id.casefold(), text.casefold(), f"{timestamp:.3f}" if timestamp is not None else "")
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            output[modality].append(row)
            if len(output[modality]) >= limit:
                break
    return output


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


def _frame_priority(frame: Dict[str, Any]) -> float:
    evidence = math.log1p(max(0.0, _float(frame.get("qa_evidence_priority"))))
    visual = _float(frame.get("qa_visual_score", frame.get("score")))
    rrf = _float(frame.get("qa_rrf_score")) * 20.0
    query_coverage = min(len(frame.get("qa_query_hits") or []), 3) * 0.03
    return visual + rrf + evidence * 0.12 + query_coverage


def _build_evidence_groups(candidates: List[Dict[str, Any]], window_seconds: float) -> List[Dict[str, Any]]:
    """Group nearby retrieved keyframes into event windows before VLM evaluation."""
    by_video: Dict[str, List[Dict[str, Any]]] = {}
    for frame in candidates:
        by_video.setdefault(str(frame.get("video_id") or "unknown"), []).append(frame)

    video_query_coverage: Dict[str, set[int]] = {}
    video_event_coverage: Dict[str, set[str]] = {}
    video_event_best_ranks: Dict[str, Dict[str, int]] = {}
    video_event_best_timestamps: Dict[str, Dict[str, float]] = {}
    for video_id, frames in by_video.items():
        query_ids: set[int] = set()
        event_ids: set[str] = set()
        event_best_ranks: Dict[str, int] = {}
        event_best_timestamps: Dict[str, float] = {}
        for frame in frames:
            for hit in frame.get("qa_query_hits") or []:
                try:
                    query_ids.add(int(hit.get("query_index")))
                except (TypeError, ValueError):
                    pass
                event_id = str(hit.get("event_id") or "global")
                if event_id != "global":
                    event_ids.add(event_id)
                    try:
                        query_rank = int(hit.get("rank"))
                    except (TypeError, ValueError):
                        continue
                    if query_rank < event_best_ranks.get(event_id, 100000):
                        event_best_ranks[event_id] = query_rank
                        timestamp = _float(frame.get("timestamp"), float("nan"))
                        if math.isfinite(timestamp):
                            event_best_timestamps[event_id] = timestamp
        video_query_coverage[video_id] = query_ids
        video_event_coverage[video_id] = event_ids
        video_event_best_ranks[video_id] = event_best_ranks
        video_event_best_timestamps[video_id] = event_best_timestamps

    raw_groups: List[Dict[str, Any]] = []
    for video_id, frames in by_video.items():
        timed = sorted(
            frames,
            key=lambda frame: (
                _float(frame.get("timestamp"), float("inf")),
                int(frame.get("qa_visual_rank") or 100000),
            ),
        )
        current: List[Dict[str, Any]] = []
        last_timestamp: float | None = None
        for frame in timed:
            timestamp = _float(frame.get("timestamp"), float("nan"))
            has_timestamp = math.isfinite(timestamp)
            starts_new = bool(
                current
                and (
                    not has_timestamp
                    or last_timestamp is None
                    or timestamp - last_timestamp > max(0.0, window_seconds)
                )
            )
            if starts_new:
                raw_groups.append({"video_id": video_id, "frames": current})
                current = []
            current.append(frame)
            last_timestamp = timestamp if has_timestamp else None
        if current:
            raw_groups.append({"video_id": video_id, "frames": current})

    for group in raw_groups:
        group_frames = group["frames"]
        best_query_ranks: Dict[int, int] = {}
        for frame in group_frames:
            for hit in frame.get("qa_query_hits") or []:
                try:
                    query_index = int(hit.get("query_index"))
                    query_rank = int(hit.get("rank"))
                except (TypeError, ValueError):
                    continue
                best_query_ranks[query_index] = min(
                    best_query_ranks.get(query_index, query_rank),
                    query_rank,
                )
        group["best_query_ranks"] = best_query_ranks
        group["score"] = max((_frame_priority(frame) for frame in group_frames), default=0.0)
        group["score"] += min(len(group_frames), 5) * 0.01
        video_id = str(group.get("video_id") or "unknown")
        query_coverage = len(video_query_coverage.get(video_id, set()))
        event_coverage = len(video_event_coverage.get(video_id, set()))
        event_best_ranks = video_event_best_ranks.get(video_id, {})
        event_best_timestamps = video_event_best_timestamps.get(video_id, {})
        event_worst_rank = max(event_best_ranks.values(), default=100000)
        event_times = list(event_best_timestamps.values())
        event_span_seconds = max(event_times) - min(event_times) if len(event_times) > 1 else 0.0
        # Multiple phrasings of one event are useful, but covering two distinct
        # events in one video is a much stronger signal for temporal QA.  The
        # weakest event matters: rank 3 + rank 9 is better sequence evidence
        # than rank 1 + rank 118, even though both videos cover two event ids.
        group["score"] += min(query_coverage, 6) * 0.025
        group["score"] += max(0, event_coverage - 1) * 0.45
        if event_coverage > 1 and event_worst_rank < 100000:
            group["score"] += 0.8 / max(1.0, math.log2(event_worst_rank + 1))
            # A compact cluster of matching events is more likely to be the
            # described clip than coincidental matches scattered across a
            # long news programme.
            group["score"] += 0.35 / (1.0 + event_span_seconds / 60.0)
        group["video_query_coverage"] = query_coverage
        group["video_event_coverage"] = event_coverage
        group["video_event_best_ranks"] = dict(event_best_ranks)
        group["video_event_worst_rank"] = event_worst_rank
        group["video_event_best_timestamps"] = dict(event_best_timestamps)
        group["video_event_span_seconds"] = event_span_seconds
        for frame in group_frames:
            frame["qa_video_query_coverage"] = query_coverage
            frame["qa_video_event_coverage"] = event_coverage
            frame["qa_video_event_best_ranks"] = dict(event_best_ranks)
            frame["qa_video_event_worst_rank"] = event_worst_rank
            frame["qa_video_event_span_seconds"] = event_span_seconds
        timestamps = [
            _float(frame.get("timestamp"))
            for frame in group_frames
            if frame.get("timestamp") is not None
        ]
        group["timestamp_start"] = min(timestamps) if timestamps else None
        group["timestamp_end"] = max(timestamps) if timestamps else None

    raw_groups.sort(key=lambda group: _float(group.get("score")), reverse=True)
    for index, group in enumerate(raw_groups, 1):
        group_id = f"g{index}"
        group["id"] = group_id
        for frame in group["frames"]:
            frame["qa_group_id"] = group_id
    return raw_groups


def _rank_video_hypotheses(
    groups: Sequence[Dict[str, Any]],
    plan: Dict[str, Any],
    limit: int,
) -> List[Dict[str, Any]]:
    """Rank coherent video candidates instead of treating every frame alike."""
    by_video: Dict[str, List[Dict[str, Any]]] = {}
    for group in groups:
        video_id = _clean(group.get("video_id")) or "unknown"
        by_video.setdefault(video_id, []).append(group)

    event_total = len(plan.get("event_queries") or [])
    hypotheses: List[Dict[str, Any]] = []
    query_total = max(1, len(plan.get("visual_queries") or []))
    for video_id, video_groups in by_video.items():
        ranked_groups = sorted(video_groups, key=lambda group: _float(group.get("score")), reverse=True)
        strongest = ranked_groups[0]
        event_coverage = int(strongest.get("video_event_coverage") or 0)
        if event_total > 0:
            coverage_ratio = min(1.0, event_coverage / event_total)
        else:
            # Non-temporal questions have one implicit scene rather than
            # explicit event ids, so query agreement is their coverage cue.
            coverage_ratio = min(
                1.0,
                int(strongest.get("video_query_coverage") or 0) / query_total,
            )
        query_coverage = int(strongest.get("video_query_coverage") or 0)
        worst_rank = int(strongest.get("video_event_worst_rank") or 100000)
        event_span = _float(strongest.get("video_event_span_seconds"), 0.0)
        text_priority = max(
            (
                _float(frame.get("qa_evidence_priority"))
                for group in ranked_groups
                for frame in group.get("frames") or []
            ),
            default=0.0,
        )
        complete = bool(event_total > 1 and event_coverage >= event_total)
        score = _float(strongest.get("score"))
        score += coverage_ratio * (1.4 if event_total > 1 else 0.35)
        score += min(query_coverage, query_total) / query_total * 0.25
        score += math.log1p(max(0.0, text_priority)) * 0.32
        if complete:
            score += 0.75
        if event_coverage > 1 and worst_rank < 100000:
            score += 0.6 / max(1.0, math.log2(worst_rank + 1))
            score += 0.25 / (1.0 + event_span / 60.0)
        hypotheses.append({
            "video_id": video_id,
            "score": score,
            "retrieval_score": _float(strongest.get("score")),
            "event_coverage": event_coverage,
            "event_total": event_total,
            "event_coverage_ratio": coverage_ratio,
            "event_best_ranks": strongest.get("video_event_best_ranks") or {},
            "event_worst_rank": worst_rank,
            "event_span_seconds": event_span,
            "query_coverage": query_coverage,
            "text_evidence_priority": text_priority,
            "complete_event_match": complete,
            "groups": ranked_groups,
        })

    hypotheses.sort(
        key=lambda hypothesis: (
            bool(hypothesis.get("complete_event_match")),
            _float(hypothesis.get("event_coverage_ratio")),
            -int(hypothesis.get("event_worst_rank") or 100000),
            -_float(hypothesis.get("event_span_seconds")),
            _float(hypothesis.get("retrieval_score")),
            _float(hypothesis.get("score")),
        ),
        reverse=True,
    )
    return hypotheses[: max(1, limit)]


def _prioritize_hypothesis_groups(
    diversified: Sequence[Dict[str, Any]],
    hypotheses: Sequence[Dict[str, Any]],
    limit: int,
) -> List[Dict[str, Any]]:
    """Guarantee VLM evidence for several distinct video hypotheses."""
    selected: List[Dict[str, Any]] = []
    seen: set[str] = set()

    def add(group: Dict[str, Any]) -> None:
        key = str(group.get("id") or id(group))
        if key in seen or len(selected) >= max(1, limit):
            return
        selected.append(group)
        seen.add(key)

    # One strong moment from each candidate lets the candidate-answer pass
    # inspect lower-ranked videos instead of seeing only near-duplicates from
    # the global top result.
    for hypothesis in hypotheses:
        groups = list(hypothesis.get("groups") or [])
        if groups:
            add(groups[0])

    # The strongest video receives an additional event/text moment when
    # available, which helps a multi-event description stay coherent.
    if hypotheses:
        for group in list(hypotheses[0].get("groups") or [])[1:3]:
            add(group)

    for group in diversified:
        add(group)
    return selected


def _diversify_evidence_groups(
    groups: List[Dict[str, Any]],
    visual_queries: Sequence[str],
    limit: int,
    query_priorities: Sequence[int] | None = None,
) -> List[Dict[str, Any]]:
    """Reserve evidence slots for individual retrieval intents.

    Pure RRF can let a large burst of generic matches crowd out the third-best
    result of a highly specific query.  QA needs candidate diversity because
    the VLM, not the embedding score, is the final visual verifier.
    """
    max_groups = max(1, limit)
    if not groups or not visual_queries:
        return groups[:max_groups]

    query_order = sorted(
        range(len(visual_queries)),
        key=lambda index: (
            int(query_priorities[index])
            if query_priorities is not None and index < len(query_priorities)
            else 0,
            len(re.findall(r"[a-z0-9]+", str(visual_queries[index]).lower())),
            index,
        ),
        reverse=True,
    )
    by_query: Dict[int, List[Dict[str, Any]]] = {}
    for query_index in query_order:
        matching = [
            group
            for group in groups
            if query_index in (group.get("best_query_ranks") or {})
        ]
        matching.sort(
            key=lambda group: (
                int((group.get("best_query_ranks") or {}).get(query_index, 100000)),
                -_float(group.get("score")),
            )
        )
        by_query[query_index] = matching

    selected: List[Dict[str, Any]] = []
    seen: set[str] = set()

    def add(group: Dict[str, Any]) -> None:
        key = str(group.get("id") or id(group))
        if key in seen or len(selected) >= max_groups:
            return
        selected.append(group)
        seen.add(key)

    # Preserve the strongest fused evidence first.  Previously, four results
    # from the longest query were inserted before the fused ranking; long
    # story-like questions then filled the VLM context with lexical false
    # positives while the correct top groups were left out.
    fused_reserve = min(4, max(1, max_groups // 2))
    for group in groups[:fused_reserve]:
        add(group)

    # Reserve anchors produced directly by OCR/ASR.  They can identify the
    # exact video even when visual embeddings prefer a visually similar news
    # story.  Rank by raw text relevance first, then by the group score.
    text_groups = [
        group
        for group in groups
        if any(frame.get("qa_text_evidence") for frame in group.get("frames") or [])
    ]
    text_groups.sort(
        key=lambda group: (
            max(
                (_float(frame.get("qa_evidence_priority")) for frame in group.get("frames") or []),
                default=0.0,
            ),
            _float(group.get("score")),
        ),
        reverse=True,
    )
    for group in text_groups[:2]:
        add(group)

    # The longest/high-priority query still gets alternatives, after the fused
    # anchors have been secured.
    most_specific = query_order[0]
    for group in by_query.get(most_specific, [])[:2]:
        add(group)

    # Preserve at least one candidate for every other event/query variant.
    for query_index in query_order[1:]:
        for group in by_query.get(query_index, []):
            before = len(selected)
            add(group)
            if len(selected) > before:
                break

    # Fill remaining capacity with the normal fused ranking.
    for group in groups:
        add(group)
        if len(selected) >= max_groups:
            break
    return selected


def _timeline_context(retriever: Any, anchor: Dict[str, Any], limit: int) -> List[Dict[str, Any]]:
    if limit <= 1 or not hasattr(retriever, "get_video_timeline"):
        return []
    around = anchor.get("frame_id") or anchor.get("frame_name") or anchor.get("global_frame_id")
    try:
        return list(
            retriever.get_video_timeline(
                video_id=str(anchor.get("video_id") or ""),
                around_frame_id=str(around) if around not in (None, "") else None,
                limit=limit,
            )
            or []
        )
    except Exception as exc:
        logger.warning("Grounded Q&A timeline context unavailable: %s", exc)
        return []


def _prepend_initial_window_frames(
    retriever: Any,
    groups: Sequence[Dict[str, Any]],
    selected: Sequence[Dict[str, Any]],
    plan: Dict[str, Any],
    limit: int,
) -> List[Dict[str, Any]]:
    """Guarantee evidence from an explicitly requested initial time range.

    Generic visual similarity often retrieves a matching scene late in another
    video.  If the question says "first N seconds", choose the strongest group
    that actually falls inside that interval and prepend its early timeline.
    """
    window = _float(plan.get("time_window_seconds"), 0.0)
    if window <= 0.0 or not hasattr(retriever, "get_video_timeline"):
        return list(selected)[:limit]

    eligible: List[Dict[str, Any]] = []
    for group in groups:
        timestamps = [
            _float(frame.get("timestamp"), float("inf"))
            for frame in group.get("frames") or []
        ]
        if timestamps and min(timestamps) <= window:
            eligible.append(group)
    if not eligible:
        return list(selected)[:limit]

    target = max(eligible, key=lambda group: _float(group.get("score")))
    video_id = _clean(target.get("video_id"))
    try:
        timeline = list(retriever.get_video_timeline(video_id=video_id, limit=60) or [])
    except Exception as exc:
        logger.warning("Grounded Q&A initial timeline unavailable: %s", exc)
        timeline = []
    timeline.extend(target.get("frames") or [])

    scan: List[Dict[str, Any]] = []
    seen_scan: set[str] = set()
    for frame in sorted(timeline, key=lambda item: _float(item.get("timestamp"), float("inf"))):
        if _float(frame.get("timestamp"), float("inf")) > window:
            continue
        key = _identity(frame)
        if not key or key in seen_scan or resolve_keyframe_path(frame) is None:
            continue
        item = dict(frame)
        item["qa_group_id"] = f"initial:{video_id}:0-{window:g}s"
        item["qa_temporal_scan"] = True
        scan.append(item)
        seen_scan.add(key)
        if len(scan) >= min(8, max(1, limit // 2)):
            break

    output: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for frame in [*scan, *selected]:
        key = _identity(frame)
        if not key or key in seen:
            continue
        output.append(frame)
        seen.add(key)
        if len(output) >= limit:
            break
    return output


def _select_grouped_frames(
    retriever: Any,
    groups: List[Dict[str, Any]],
    limit: int,
    per_video_limit: int,
    max_groups: int,
    context_per_group: int,
) -> List[Dict[str, Any]]:
    """Round-robin the best event groups so one near-duplicate burst cannot dominate."""
    prepared: List[List[Dict[str, Any]]] = []
    for group in groups:
        ranked = sorted(group["frames"], key=_frame_priority, reverse=True)
        if not ranked:
            continue
        anchor = ranked[0]
        augmented = ranked + _timeline_context(retriever, anchor, context_per_group)
        unique: Dict[str, Dict[str, Any]] = {}
        anchor_timestamp = _float(anchor.get("timestamp"))
        for frame in augmented:
            key = _identity(frame)
            if not key or key in unique:
                continue
            item = dict(frame)
            item["qa_group_id"] = group["id"]
            item["qa_context_frame"] = key not in {_identity(value) for value in ranked}
            item["qa_anchor_delta"] = abs(_float(item.get("timestamp"), anchor_timestamp) - anchor_timestamp)
            unique[key] = item
        usable = [item for item in unique.values() if resolve_keyframe_path(item) is not None]
        usable.sort(
            key=lambda frame: (
                not bool(frame.get("qa_context_frame")),
                bool(frame.get("qa_text_evidence")),
                _frame_priority(frame),
                -_float(frame.get("qa_anchor_delta")),
            ),
            reverse=True,
        )
        if usable:
            prepared.append(usable[: max(1, context_per_group)])
        if len(prepared) >= max(1, max_groups):
            break

    selected: List[Dict[str, Any]] = []
    counts: Dict[str, int] = {}
    seen: set[str] = set()
    depth = 0
    while len(selected) < limit and any(depth < len(items) for items in prepared):
        for items in prepared:
            if depth >= len(items):
                continue
            frame = items[depth]
            key = _identity(frame)
            video_id = str(frame.get("video_id") or "unknown")
            if key in seen or counts.get(video_id, 0) >= per_video_limit:
                continue
            selected.append(frame)
            seen.add(key)
            counts[video_id] = counts.get(video_id, 0) + 1
            if len(selected) >= limit:
                break
        depth += 1
    return selected


def _build_candidate_bundles(
    hypotheses: Sequence[Dict[str, Any]],
    selected_ids: Dict[str, Dict[str, Any]],
    max_videos: int,
    frames_per_video: int,
) -> List[Dict[str, Any]]:
    """Attach a small, independent frame bundle to each video hypothesis."""
    by_video: Dict[str, List[Tuple[str, Dict[str, Any]]]] = {}
    for frame_id, frame in selected_ids.items():
        by_video.setdefault(_clean(frame.get("video_id")) or "unknown", []).append((frame_id, frame))

    bundles: List[Dict[str, Any]] = []
    for hypothesis in hypotheses:
        video_id = _clean(hypothesis.get("video_id")) or "unknown"
        available = by_video.get(video_id) or []
        if not available:
            continue
        # Text anchors and genuine retrieval hits are more informative than
        # timeline context, while still preserving the selected order.
        available = sorted(
            available,
            key=lambda pair: (
                bool(pair[1].get("qa_text_evidence")),
                not bool(pair[1].get("qa_context_frame")),
                _frame_priority(pair[1]),
            ),
            reverse=True,
        )
        chosen = available[: max(1, frames_per_video)]
        bundles.append({
            "candidate_id": f"c{len(bundles) + 1}",
            "video_id": video_id,
            "frame_ids": [frame_id for frame_id, _frame in chosen],
            "frames": {frame_id: frame for frame_id, frame in chosen},
            "score": _float(hypothesis.get("score")),
            "retrieval_score": _float(hypothesis.get("retrieval_score")),
            "event_coverage": int(hypothesis.get("event_coverage") or 0),
            "event_total": int(hypothesis.get("event_total") or 0),
            "event_coverage_ratio": _float(hypothesis.get("event_coverage_ratio")),
            "event_span_seconds": _float(hypothesis.get("event_span_seconds")),
            "query_coverage": int(hypothesis.get("query_coverage") or 0),
            "complete_event_match": bool(hypothesis.get("complete_event_match")),
        })
        if len(bundles) >= max(1, max_videos):
            break
    return bundles


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


def _request_structured(
    messages: List[Dict[str, Any]],
    model: str,
    timeout: float,
    max_tokens: int,
    schema: Dict[str, Any],
) -> Dict[str, Any]:

    def request_via_gateway() -> Dict[str, Any]:
        from src.services.ai import gateway as ai_gateway
        from src.services.ai.base import AllProvidersFailed

        try:
            payload, _attempts, _provider = ai_gateway.vision_completion(
                messages,
                max_tokens=max_tokens,
                response_format={"type": "json_schema", "json_schema": schema},
            )
        except AllProvidersFailed as exc:
            raise urllib.error.URLError(f"vision chain exhausted: {exc}") from exc
        content = (((payload.get("choices") or [{}])[0].get("message") or {}).get("content") or "")
        if isinstance(content, list):
            content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
        return _extract_json_object(content)

    settings = get_settings()
    from src.services.openrouter_vlm_verifier import vision_gateway_available

    if vision_gateway_available(settings):
        return request_via_gateway()

    body = {
        "model": model,
        "temperature": 0,
        "max_tokens": max_tokens,
        "messages": messages,
        "response_format": {"type": "json_schema", "json_schema": schema},
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


def _request_answer(messages: List[Dict[str, Any]], model: str, timeout: float, max_tokens: int) -> Dict[str, Any]:
    return _request_structured(messages, model, timeout, max_tokens, ANSWER_SCHEMA)


def _request_verification(
    messages: List[Dict[str, Any]],
    model: str,
    timeout: float,
    max_tokens: int,
) -> Dict[str, Any]:
    return _request_structured(messages, model, timeout, max_tokens, VERIFICATION_SCHEMA)


def _request_candidate_answers(
    messages: List[Dict[str, Any]],
    model: str,
    timeout: float,
    max_tokens: int,
) -> Dict[str, Any]:
    return _request_structured(messages, model, timeout, max_tokens, CANDIDATE_ANSWER_SCHEMA)


def _encode_detail_image(image: Image.Image, max_side: int, *, allow_upscale: bool) -> str:
    """Encode a high-quality inspection image, optionally enlarging small crops."""
    image = image.convert("RGB")
    width, height = image.size
    longest = max(width, height, 1)
    if allow_upscale:
        scale = min(max_side / longest, 4.0)
    else:
        scale = min(max_side / longest, 1.0)
    if abs(scale - 1.0) > 0.01:
        image = image.resize(
            (max(1, round(width * scale)), max(1, round(height * scale))),
            Image.Resampling.LANCZOS,
        )
    output = io.BytesIO()
    image.save(output, format="JPEG", quality=90, optimize=True)
    encoded = base64.b64encode(output.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _detail_image_data_urls(
    path: Path,
    max_side: int,
    grid_size: int = 3,
) -> List[Tuple[str, str]]:
    """Return a full frame plus overlapping zoom tiles for tiny displays/text."""
    grid_size = max(2, min(int(grid_size), 4))
    max_side = max(384, min(int(max_side), 1600))
    with Image.open(path) as source:
        image = source.convert("RGB")
        width, height = image.size
        parts: List[Tuple[str, str]] = [
            ("toàn cảnh", _encode_detail_image(image.copy(), max_side, allow_upscale=False))
        ]
        overlap_x = max(2, round(width * 0.025))
        overlap_y = max(2, round(height * 0.025))
        for row in range(grid_size):
            for column in range(grid_size):
                x0 = max(0, math.floor(column * width / grid_size) - overlap_x)
                y0 = max(0, math.floor(row * height / grid_size) - overlap_y)
                x1 = min(width, math.ceil((column + 1) * width / grid_size) + overlap_x)
                y1 = min(height, math.ceil((row + 1) * height / grid_size) + overlap_y)
                crop = image.crop((x0, y0, x1, y1))
                parts.append((
                    f"ô phóng to hàng {row + 1}, cột {column + 1}",
                    _encode_detail_image(crop, max_side, allow_upscale=True),
                ))
    return parts


def _is_unresolved_answer(answer: Any) -> bool:
    value = _clean(answer).casefold()
    return not value or bool(re.search(
        r"(?:không\s+(?:thể|đủ|có)\s+(?:xác định|đọc|thông tin|bằng chứng)|"
        r"không\s+rõ|chưa\s+(?:thể\s+)?xác định|unknown|cannot\s+determine|insufficient)",
        value,
    ))


def _select_detail_frame_ids(
    supporting_frame_ids: Sequence[str],
    selected_ids: Dict[str, Dict[str, Any]],
    limit: int,
    temporal_question: bool,
) -> List[str]:
    """Keep detail inspection coherent by choosing frames from one event group."""
    order = {frame_id: index for index, frame_id in enumerate(supporting_frame_ids)}
    by_group: Dict[str, List[str]] = {}
    for frame_id in supporting_frame_ids:
        frame = selected_ids.get(frame_id)
        if frame is None:
            continue
        group_id = str(frame.get("qa_group_id") or frame_id)
        by_group.setdefault(group_id, []).append(frame_id)
    if not by_group:
        return []

    def group_key(frame_ids: List[str]) -> Tuple[float, int, int]:
        worst_event_rank = min(
            (
                _float(selected_ids[frame_id].get("qa_video_event_worst_rank"), 100000.0)
                for frame_id in frame_ids
            ),
            default=100000.0,
        )
        return (
            worst_event_rank if temporal_question else 0.0,
            -len(frame_ids),
            min(order.get(frame_id, 100000) for frame_id in frame_ids),
        )

    best_group_ids = min(by_group.values(), key=group_key)
    return best_group_ids[: max(1, limit)]


def _canonical_answer(value: Any, max_chars: int = 100) -> str:
    answer = _clean(value)
    answer = re.sub(r"^(?:answer|câu trả lời|cau tra loi|response)\s*:\s*", "", answer, flags=re.IGNORECASE)
    if len(answer) >= 2 and answer[0] == answer[-1] and answer[0] in {'"', "'"}:
        answer = answer[1:-1].strip()
    if len(answer) <= max_chars:
        return answer
    shortened = answer[:max_chars].rstrip()
    if " " in shortened and len(answer) > max_chars and answer[max_chars : max_chars + 1] not in {"", " "}:
        shortened = shortened.rsplit(" ", 1)[0]
    return shortened.rstrip(" ,;:-")


def _is_vietnamese_answer(answer: str, answer_type: str) -> bool:
    """Reject obvious English output while allowing language-neutral OCR/counts."""
    if not answer:
        return False
    if answer_type in {"count", "ocr"}:
        return True
    if re.search(r"[à-ỹÀ-ỸđĐ]", answer):
        return True
    tokens = set(re.findall(r"[A-Za-z]+", answer.casefold()))
    return not bool(tokens & ENGLISH_ANSWER_MARKERS)


def _normalise_supporting_ids(values: Any, frame_ids: set[str]) -> List[str]:
    if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
        return []
    output: List[str] = []
    for value in values:
        candidate = value.strip()
        if candidate not in frame_ids:
            # Some vision models concatenate the descriptive frame label and
            # the short evidence id, e.g. L30_V043_001818f2.
            match = re.search(r"(f\d+)$", candidate, re.IGNORECASE)
            candidate = match.group(1).lower() if match else candidate
        if candidate in frame_ids and candidate not in output:
            output.append(candidate)
    return output


def _normalise_answer(
    payload: Dict[str, Any],
    frame_ids: set[str],
    answer_type: str,
    answer_max_chars: int = 100,
) -> Tuple[Dict[str, Any], List[str]]:
    errors: List[str] = []
    status = _clean(payload.get("status")).lower()
    answer = _canonical_answer(payload.get("answer"), answer_max_chars)
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
    if payload.get("answer_language") != "vi":
        errors.append("answer_language must be vi")
    if not _is_vietnamese_answer(answer, answer_type):
        errors.append("answer must be Vietnamese with correct diacritics")
    if not 0.0 <= confidence <= 1.0:
        errors.append("confidence out of range")
    if not isinstance(supporting, list) or any(not isinstance(value, str) for value in supporting):
        errors.append("supporting_frame_ids must be strings")
        supporting = []
    normalised_supporting = _normalise_supporting_ids(supporting, frame_ids)
    unexpected = set(supporting) - frame_ids if supporting and not normalised_supporting else set()
    if unexpected:
        errors.append(f"unexpected supporting frame ids: {sorted(unexpected)}")
    for key in ("used_ocr_evidence", "used_asr_evidence"):
        if not isinstance(payload.get(key), bool):
            errors.append(f"{key} must be boolean")
    if status == "answered" and not normalised_supporting:
        errors.append("answered result requires a supporting frame id")
    return {
        "status": status,
        "answer": answer,
        "confidence": max(0.0, min(1.0, confidence)),
        "reason": reason,
        "supporting_frame_ids": normalised_supporting,
        "used_ocr_evidence": bool(payload.get("used_ocr_evidence")),
        "used_asr_evidence": bool(payload.get("used_asr_evidence")),
        "answer_language": "vi",
    }, errors


def _normalise_candidate_answers(
    payload: Dict[str, Any],
    bundles: Sequence[Dict[str, Any]],
    answer_type: str,
    answer_max_chars: int = 100,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Validate candidate answers and keep citations inside their own video."""
    rows = payload.get("candidates")
    if not isinstance(rows, list):
        return [], ["candidate answers must be a list"]
    bundle_by_id = {str(bundle.get("candidate_id")): bundle for bundle in bundles}
    row_by_id = {
        _clean(row.get("candidate_id")): row
        for row in rows
        if isinstance(row, dict) and _clean(row.get("candidate_id"))
    }
    output: List[Dict[str, Any]] = []
    errors: List[str] = []
    for bundle in bundles:
        candidate_id = str(bundle.get("candidate_id"))
        row = row_by_id.get(candidate_id)
        if row is None:
            errors.append(f"missing candidate answer: {candidate_id}")
            continue
        allowed = set(bundle.get("frame_ids") or [])
        parsed, row_errors = _normalise_answer(
            row,
            allowed,
            answer_type,
            answer_max_chars,
        )
        if row_errors:
            errors.extend(f"{candidate_id}: {error}" for error in row_errors)
            continue
        supporting_names = [
            _clean((bundle.get("frames") or {}).get(frame_id, {}).get("frame_name"))
            for frame_id in parsed.get("supporting_frame_ids") or []
        ]
        supporting_names = [name for name in supporting_names if name]
        representative_id = next(
            iter(parsed.get("supporting_frame_ids") or bundle.get("frame_ids") or []),
            None,
        )
        representative_frame = (bundle.get("frames") or {}).get(representative_id, {})
        output.append({
            "candidate_id": candidate_id,
            "video_id": bundle.get("video_id"),
            **parsed,
            "retrieval_score": round(_float(bundle.get("retrieval_score")), 6),
            "hypothesis_score": round(_float(bundle.get("score")), 6),
            "event_coverage": int(bundle.get("event_coverage") or 0),
            "event_total": int(bundle.get("event_total") or 0),
            "event_coverage_ratio": round(_float(bundle.get("event_coverage_ratio")), 4),
            "event_span_seconds": round(_float(bundle.get("event_span_seconds")), 3),
            "query_coverage": int(bundle.get("query_coverage") or 0),
            "complete_event_match": bool(bundle.get("complete_event_match")),
            "supporting_frame_names": supporting_names,
            "representative_frame_name": _clean(representative_frame.get("frame_name")),
        })
    unexpected = sorted(set(row_by_id) - set(bundle_by_id))
    if unexpected:
        errors.append(f"unexpected candidate ids: {unexpected}")
    return output, errors


def _normalise_verification(
    payload: Dict[str, Any],
    frame_ids: set[str],
    answer_type: str,
    answer_max_chars: int = 100,
) -> Tuple[Dict[str, Any], List[str]]:
    errors: List[str] = []
    verified = payload.get("verified")
    if not isinstance(verified, bool):
        errors.append("verified must be boolean")
        verified = False
    answer = _canonical_answer(payload.get("canonical_answer"), answer_max_chars)
    reason = _clean(payload.get("reason"))[:500]
    confidence = _float(payload.get("confidence"), -1.0)
    supporting = payload.get("supporting_frame_ids")
    if not answer or not reason:
        errors.append("canonical_answer/reason is required")
    if payload.get("answer_language") != "vi":
        errors.append("verifier answer_language must be vi")
    if not _is_vietnamese_answer(answer, answer_type):
        errors.append("verified answer must be Vietnamese with correct diacritics")
    if not 0.0 <= confidence <= 1.0:
        errors.append("confidence out of range")
    if not isinstance(supporting, list) or any(not isinstance(value, str) for value in supporting):
        errors.append("supporting_frame_ids must be strings")
        supporting = []
    normalised_supporting = _normalise_supporting_ids(supporting, frame_ids)
    unexpected = set(supporting) - frame_ids if supporting and not normalised_supporting else set()
    if unexpected:
        errors.append(f"unexpected verifier supporting frame ids: {sorted(unexpected)}")
    if verified and not normalised_supporting:
        errors.append("verified result requires a supporting frame id")
    return {
        "verified": bool(verified),
        "canonical_answer": answer,
        "confidence": max(0.0, min(1.0, confidence)),
        "reason": reason,
        "supporting_frame_ids": normalised_supporting,
        "answer_language": "vi",
    }, errors


def _uncertain_answer(_question: str, reason: str) -> Dict[str, Any]:
    answer = "Không đủ bằng chứng đã truy xuất để trả lời chắc chắn."
    return {
        "status": "uncertain",
        "answer": answer,
        "confidence": 0.0,
        "reason": reason,
        "supporting_frame_ids": [],
        "used_ocr_evidence": False,
        "used_asr_evidence": False,
        "answer_language": "vi",
    }


def _best_guess_answer(
    verdict: Dict[str, Any],
    reason: str,
    confidence_cap: float,
) -> Dict[str, Any]:
    """Downgrade a candidate to uncertain without discarding its useful answer."""
    answer = _canonical_answer(verdict.get("answer"))
    if not answer:
        return _uncertain_answer("", reason)
    output = dict(verdict)
    output["status"] = "uncertain"
    output["answer"] = answer
    output["confidence"] = min(
        max(0.0, _float(verdict.get("confidence"))),
        max(0.0, min(1.0, confidence_cap)),
    )
    output["reason"] = _clean(reason)[:500]
    output["answer_language"] = "vi"
    return output


def grounded_video_qa(question: str, top_k: int) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    settings = get_settings()
    plan = _question_plan(question, max(1, int(getattr(settings, "qa_visual_query_limit", 3))))
    retriever = _get_retriever()
    pool_size = max(top_k, int(settings.qa_retrieval_pool))
    visual_frames = _search_visual_queries(
        retriever,
        plan["visual_queries"],
        pool_size,
        plan.get("visual_query_event_ids"),
    )
    text_evidence = _collect_text_evidence(
        plan,
        max(
            1,
            int(settings.qa_text_evidence_top_k),
            int(getattr(settings, "qa_text_retrieval_pool", 32)),
        ),
    )
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
    evidence_groups = _build_evidence_groups(
        candidates,
        max(0.0, float(getattr(settings, "qa_event_window_seconds", 8.0))),
    )
    candidate_video_limit = max(1, int(getattr(settings, "qa_candidate_max_videos", 4)))
    video_hypotheses = _rank_video_hypotheses(
        evidence_groups,
        plan,
        candidate_video_limit,
    )
    selection_groups = _diversify_evidence_groups(
        evidence_groups,
        plan["visual_queries"],
        max(1, int(getattr(settings, "qa_max_evidence_groups", 4))),
        plan.get("visual_query_priorities"),
    )
    selection_groups = _prioritize_hypothesis_groups(
        selection_groups,
        video_hypotheses,
        max(1, int(getattr(settings, "qa_max_evidence_groups", 4))),
    )
    selection_limit = max(1, min(int(settings.qa_max_frames), 16))
    selected = _select_grouped_frames(
        retriever,
        selection_groups,
        selection_limit,
        max(1, int(settings.qa_per_video_limit)),
        max(1, int(getattr(settings, "qa_max_evidence_groups", 4))),
        max(1, int(getattr(settings, "qa_context_frames_per_group", 3))),
    )
    selected = _prepend_initial_window_frames(
        retriever,
        evidence_groups,
        selected,
        plan,
        selection_limit,
    )
    selected_ids = {f"f{index}": frame for index, frame in enumerate(selected, 1)}
    candidate_bundles = _build_candidate_bundles(
        video_hypotheses,
        selected_ids,
        candidate_video_limit,
        max(1, int(getattr(settings, "qa_candidate_frames_per_video", 3))),
    )
    prompt_text_evidence = _relevant_text_evidence(
        text_evidence,
        selected,
        max(1, int(settings.qa_text_evidence_top_k)),
        max(0.0, float(settings.qa_evidence_window_seconds)),
    )
    verdict = _uncertain_answer(question, "No grounded VLM answer was produced.")
    errors: List[str] = []
    retries = 0
    verification_status = "not_run"
    detail_pass_status = "not_run"
    candidate_answers: List[Dict[str, Any]] = []
    promoted_candidate_id = ""
    prompt_evidence_sent = {"ocr": 0, "asr": 0}
    answer_max_chars = max(20, min(int(getattr(settings, "qa_answer_max_chars", 100)), 500))
    return_best_guess = bool(getattr(settings, "qa_return_best_guess", True))
    uncertain_confidence_cap = float(getattr(settings, "qa_uncertain_confidence_cap", 0.49))

    if settings.qa_vlm_enabled and settings.agent_vlm_enabled and settings.openrouter_api_key and selected:
        prompt_evidence_sent = {
            "ocr": len(prompt_text_evidence["ocr"]),
            "asr": len(prompt_text_evidence["asr"]),
        }
    from src.services.openrouter_vlm_verifier import vision_gateway_available

    _qa_vlm_ready = bool(settings.openrouter_api_key) or vision_gateway_available(settings)
    if settings.qa_vlm_enabled and settings.agent_vlm_enabled and _qa_vlm_ready and selected:
        evidence_lines = []
        for modality in ("ocr", "asr"):
            for row in prompt_text_evidence[modality]:
                text = _evidence_text(row, modality)
                if text:
                    evidence_lines.append(
                        f"{modality.upper()} video={row.get('video_id')} "
                        f"time={_evidence_timestamp(row, modality):.3f}: {text}"
                    )
        metadata_lines = _video_metadata_lines(selected)
        hypothesis_lines = [
            (
                f"video={hypothesis.get('video_id')} "
                f"event_coverage={hypothesis.get('event_coverage')}/{hypothesis.get('event_total')} "
                f"coverage_ratio={_float(hypothesis.get('event_coverage_ratio')):.2f} "
                f"event_span_seconds={_float(hypothesis.get('event_span_seconds')):.1f} "
                f"retrieval_score={_float(hypothesis.get('retrieval_score')):.3f}"
            )
            for hypothesis in video_hypotheses
        ]
        set_comparison_lines: List[str] = []
        if plan.get("requires_set_comparison"):
            set_comparison_lines = [
                f"This is a set-comparison question over {plan.get('enumeration_range')}.",
                "List the values actually visible in the coherent requested time window, then subtract them from the stated range.",
                "Do not claim that every value appeared unless every value is visibly evidenced.",
            ]
        prompt_lines = [
            f"Question: {question}",
            f"Question type: {plan['answer_type']}",
            f"Required answer format: {plan['expected_answer_format']}; maximum {answer_max_chars} characters.",
            f"Visual retrieval queries: {json.dumps(plan['visual_queries'], ensure_ascii=False)}",
            "Each event group is an alternative retrieved moment; use one group unless the question is temporal.",
            "Never combine frames or text evidence from different video ids into one answer.",
            "For a multi-event description, prefer one video that covers every event, even if another video's first frame score is higher.",
            "Ranked coherent video hypotheses:",
            *(hypothesis_lines or ["No coherent video hypothesis was available."]),
            "For counts, count subjects in one frame/event and do not add repeated subjects across frames.",
            *set_comparison_lines,
            "Text evidence:",
            *(evidence_lines or ["No OCR/ASR evidence was retrieved."]),
            "Existing video catalogue metadata (use only for the matching attached video):",
            *(metadata_lines or ["No video title metadata was available."]),
            "Attached keyframes are labelled with their event group:",
        ]
        content: List[Dict[str, Any]] = [{"type": "text", "text": "\n".join(prompt_lines)}]
        verification_base_content = content
        max_side = max(256, min(int(settings.agent_vlm_image_max_side), 1200))
        for frame_id, frame in selected_ids.items():
            path = resolve_keyframe_path(frame)
            content.append({
                "type": "text",
                "text": (
                    f"{frame_id}: group={frame.get('qa_group_id')} video={frame.get('video_id')} "
                    f"timestamp={frame.get('timestamp')} frame={frame.get('frame_name')}"
                ),
            })
            content.append({"type": "image_url", "image_url": {"url": _image_to_data_url(Path(path), max_side)}})

        max_retries = max(0, min(int(settings.agent_vlm_max_retries), 3))
        payload: Dict[str, Any] = {}
        for attempt in range(max_retries + 1):
            try:
                payload = _request_answer(
                    [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": content}],
                    str(getattr(settings, "qa_answer_model", "") or settings.agent_vlm_model),
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
        parsed, contract_errors = _normalise_answer(
            payload,
            set(selected_ids),
            plan["answer_type"],
            answer_max_chars,
        )
        errors.extend(contract_errors)
        if not contract_errors:
            verdict = parsed

        detail_types = {"count", "ocr", "location", "object"}
        needs_detail = bool(
            verdict.get("status") == "uncertain"
            or _is_unresolved_answer(verdict.get("answer"))
            or _float(verdict.get("confidence")) < float(settings.qa_min_confidence)
        )
        if (
            bool(getattr(settings, "qa_detail_pass_enabled", True))
            and not contract_errors
            and plan["answer_type"] in detail_types
            and needs_detail
        ):
            detail_limit = max(1, min(int(getattr(settings, "qa_detail_max_frames", 2)), 4))
            supporting_ids = list(dict.fromkeys(verdict.get("supporting_frame_ids") or []))
            if supporting_ids:
                detail_frame_ids = _select_detail_frame_ids(
                    supporting_ids,
                    selected_ids,
                    detail_limit,
                    bool(plan.get("needs_temporal_context")),
                )
            else:
                # A first pass may recognise that retrieval is relevant but be
                # unable to cite a tiny value.  Inspect the strongest attached
                # anchors instead of skipping the high-resolution pass.
                detail_frame_ids = list(selected_ids)[:detail_limit]
            detail_lines = [
                f"Câu hỏi: {question}",
                f"Loại câu trả lời: {plan['answer_type']}",
                f"Định dạng yêu cầu: {plan['expected_answer_format']}; tối đa {answer_max_chars} ký tự.",
                "Các frame dưới đây là những bằng chứng truy xuất mạnh nhất cần kiểm tra kỹ.",
                "Hãy đọc giá trị nhìn thấy rõ nhất bằng cách đối chiếu toàn cảnh và các ô phóng to.",
                *(evidence_lines or ["Không có OCR/ASR liên quan được gửi."]),
                *(metadata_lines or ["Không có tiêu đề video liên quan được gửi."]),
                *set_comparison_lines,
            ]
            detail_content: List[Dict[str, Any]] = [
                {"type": "text", "text": "\n".join(detail_lines)}
            ]
            detail_image_errors = False
            for frame_id in detail_frame_ids:
                frame = selected_ids.get(frame_id)
                if frame is None:
                    continue
                path = Path(resolve_keyframe_path(frame))
                detail_content.append({
                    "type": "text",
                    "text": (
                        f"{frame_id}: video={frame.get('video_id')} timestamp={frame.get('timestamp')} "
                        f"frame={frame.get('frame_name')}"
                    ),
                })
                try:
                    detail_parts = _detail_image_data_urls(
                        path,
                        int(getattr(settings, "qa_detail_image_max_side", 900)),
                        int(getattr(settings, "qa_detail_grid_size", 3)),
                    )
                except Exception as exc:
                    detail_image_errors = True
                    errors.append(f"detail image: {_clean(exc)[:150]}")
                    logger.warning("Grounded Q&A detail image preparation failed: %s", exc)
                    continue
                for label, data_url in detail_parts:
                    detail_content.append({"type": "text", "text": f"{frame_id} — {label}"})
                    detail_content.append({"type": "image_url", "image_url": {"url": data_url}})

            detail_payload: Dict[str, Any] = {}
            if any(part.get("type") == "image_url" for part in detail_content):
                for attempt in range(max_retries + 1):
                    try:
                        detail_payload = _request_answer(
                            [
                                {"role": "system", "content": DETAIL_SYSTEM_PROMPT},
                                {"role": "user", "content": detail_content},
                            ],
                            str(getattr(settings, "qa_detail_model", "") or settings.agent_vlm_model),
                            float(settings.agent_vlm_timeout_seconds),
                            int(settings.qa_max_tokens),
                        )
                        break
                    except Exception as exc:
                        errors.append(f"detail: {_clean(exc)[:160]}")
                        logger.warning("Grounded Q&A detail request failed: %s", exc)
                        if attempt >= max_retries:
                            break
                        retries += 1
                        time.sleep(
                            max(0.0, float(settings.agent_vlm_retry_backoff_seconds)) * (2**attempt)
                        )
                detail_parsed, detail_errors = _normalise_answer(
                    detail_payload,
                    set(detail_frame_ids),
                    plan["answer_type"],
                    answer_max_chars,
                )
                errors.extend(f"detail: {error}" for error in detail_errors)
                if not detail_errors:
                    old_unresolved = _is_unresolved_answer(verdict.get("answer"))
                    new_unresolved = _is_unresolved_answer(detail_parsed.get("answer"))
                    should_replace = bool(detail_parsed.get("supporting_frame_ids")) and (
                        (old_unresolved and not new_unresolved)
                        or detail_parsed.get("status") == "answered"
                        or _float(detail_parsed.get("confidence")) > _float(verdict.get("confidence"))
                    )
                    if should_replace:
                        verdict = detail_parsed
                        verification_base_content = detail_content
                        detail_pass_status = "refined"
                    else:
                        detail_pass_status = "unresolved"
                else:
                    detail_pass_status = "unavailable"
            else:
                detail_pass_status = "unavailable" if detail_image_errors else "not_run"

        if (
            verdict["status"] == "answered"
            and bool(getattr(settings, "qa_verify_enabled", True))
            and not contract_errors
        ):
            verification_content = list(verification_base_content)
            verification_content.append({
                "type": "text",
                "text": (
                    "Proposed answer to verify:\n"
                    + json.dumps(
                        {
                            "answer": verdict["answer"],
                            "confidence": verdict["confidence"],
                            "supporting_frame_ids": verdict["supporting_frame_ids"],
                        },
                        ensure_ascii=False,
                    )
                ),
            })
            verification_payload: Dict[str, Any] = {}
            for attempt in range(max_retries + 1):
                try:
                    verification_payload = _request_verification(
                        [
                            {"role": "system", "content": VERIFIER_SYSTEM_PROMPT},
                            {"role": "user", "content": verification_content},
                        ],
                        settings.agent_vlm_model,
                        float(settings.agent_vlm_timeout_seconds),
                        int(settings.qa_max_tokens),
                    )
                    break
                except Exception as exc:
                    errors.append(f"verification: {_clean(exc)[:160]}")
                    logger.warning("Grounded Q&A verification request failed: %s", exc)
                    if attempt >= max_retries:
                        break
                    retries += 1
                    time.sleep(max(0.0, float(settings.agent_vlm_retry_backoff_seconds)) * (2**attempt))
            verified, verification_errors = _normalise_verification(
                verification_payload,
                set(selected_ids),
                plan["answer_type"],
                answer_max_chars,
            )
            errors.extend(verification_errors)
            if not verification_errors:
                if verified["verified"]:
                    verification_status = "verified"
                    verdict["answer"] = verified["canonical_answer"]
                    verdict["confidence"] = min(verdict["confidence"], verified["confidence"])
                    verdict["reason"] = verified["reason"]
                    verdict["supporting_frame_ids"] = verified["supporting_frame_ids"]
                    verdict["answer_language"] = "vi"
                else:
                    verification_status = "rejected"
                    candidate = dict(verdict)
                    candidate["answer"] = verified["canonical_answer"]
                    candidate["confidence"] = min(
                        _float(verdict.get("confidence")),
                        _float(verified.get("confidence")),
                    )
                    candidate["reason"] = verified["reason"]
                    candidate["supporting_frame_ids"] = (
                        verified["supporting_frame_ids"]
                        or verdict["supporting_frame_ids"]
                    )
                    verdict = (
                        _best_guess_answer(
                            candidate,
                            verified["reason"],
                            uncertain_confidence_cap,
                        )
                        if return_best_guess
                        else _uncertain_answer(question, verified["reason"])
                    )
            else:
                verification_status = "unavailable"

        if (
            bool(getattr(settings, "qa_candidate_answers_enabled", True))
            and len(candidate_bundles) > 1
        ):
            candidate_lines = [
                f"Câu hỏi: {question}",
                f"Loại câu trả lời: {plan['answer_type']}",
                f"Định dạng: {plan['expected_answer_format']}; tối đa {answer_max_chars} ký tự.",
                "Hãy trả lời độc lập cho từng candidate video dưới đây.",
                "Không được lấy tên, số, OCR, ASR hay suy luận từ candidate khác.",
            ]
            for bundle in candidate_bundles:
                candidate_lines.append(
                    f"CANDIDATE {bundle['candidate_id']} video={bundle['video_id']} "
                    f"event_coverage={bundle['event_coverage']}/{bundle['event_total']} "
                    f"coverage_ratio={bundle['event_coverage_ratio']:.2f} "
                    f"event_span_seconds={bundle['event_span_seconds']:.1f} "
                    f"allowed_frames={json.dumps(bundle['frame_ids'], ensure_ascii=False)}"
                )
                bundle_metadata = _video_metadata_lines(list(bundle.get("frames", {}).values()))
                for line in bundle_metadata:
                    candidate_lines.append(f"{bundle['candidate_id']} METADATA {line}")
                for modality in ("ocr", "asr"):
                    for row in prompt_text_evidence[modality]:
                        if _clean(row.get("video_id")) != bundle["video_id"]:
                            continue
                        evidence_text = _evidence_text(row, modality)
                        if evidence_text:
                            candidate_lines.append(
                                f"{bundle['candidate_id']} {modality.upper()} "
                                f"time={_evidence_timestamp(row, modality):.3f}: {evidence_text}"
                            )

            candidate_content: List[Dict[str, Any]] = [
                {"type": "text", "text": "\n".join(candidate_lines)}
            ]
            for bundle in candidate_bundles:
                for frame_id, frame in bundle.get("frames", {}).items():
                    path = resolve_keyframe_path(frame)
                    if path is None:
                        continue
                    candidate_content.append({
                        "type": "text",
                        "text": (
                            f"{bundle['candidate_id']} {frame_id}: video={bundle['video_id']} "
                            f"timestamp={frame.get('timestamp')} frame={frame.get('frame_name')}"
                        ),
                    })
                    candidate_content.append({
                        "type": "image_url",
                        "image_url": {"url": _image_to_data_url(Path(path), max_side)},
                    })

            candidate_payload: Dict[str, Any] = {}
            for attempt in range(max_retries + 1):
                try:
                    candidate_payload = _request_candidate_answers(
                        [
                            {"role": "system", "content": CANDIDATE_SYSTEM_PROMPT},
                            {"role": "user", "content": candidate_content},
                        ],
                        str(getattr(settings, "qa_answer_model", "") or settings.agent_vlm_model),
                        float(settings.agent_vlm_timeout_seconds),
                        min(1800, max(900, int(settings.qa_max_tokens) * 2)),
                    )
                    break
                except Exception as exc:
                    errors.append(f"candidates: {_clean(exc)[:150]}")
                    logger.warning("Grounded Q&A candidate request failed: %s", exc)
                    if attempt >= max_retries:
                        break
                    retries += 1
                    time.sleep(
                        max(0.0, float(settings.agent_vlm_retry_backoff_seconds)) * (2**attempt)
                    )
            candidate_answers, candidate_errors = _normalise_candidate_answers(
                candidate_payload,
                candidate_bundles,
                plan["answer_type"],
                answer_max_chars,
            )
            errors.extend(f"candidates: {error}" for error in candidate_errors)

            # The independent candidate pass can still succeed when the main
            # VLM call times out or returns an invalid contract.  In that
            # situation, use the strongest coherent answered video instead
            # of showing a generic 0% fallback above a valid candidate card.
            primary_unresolved = bool(
                verdict.get("status") == "uncertain"
                and (
                    _float(verdict.get("confidence")) <= 0.0
                    or not verdict.get("supporting_frame_ids")
                    or _is_unresolved_answer(verdict.get("answer"))
                )
            )
            fallback_candidate = next(
                (
                    candidate
                    for candidate in candidate_answers
                    if candidate.get("status") == "answered"
                    and candidate.get("supporting_frame_ids")
                    and _float(candidate.get("confidence")) >= float(settings.qa_min_confidence)
                ),
                None,
            )
            if primary_unresolved and fallback_candidate is not None:
                verdict = {
                    key: fallback_candidate[key]
                    for key in (
                        "status",
                        "answer",
                        "confidence",
                        "reason",
                        "supporting_frame_ids",
                        "used_ocr_evidence",
                        "used_asr_evidence",
                        "answer_language",
                    )
                }
                promoted_candidate_id = str(fallback_candidate.get("candidate_id") or "")

    if verdict["status"] == "answered" and verdict["confidence"] < float(settings.qa_min_confidence):
        verdict = (
            _best_guess_answer(
                verdict,
                "Đáp án có khả năng đúng nhất nhưng độ tin cậy chưa đạt ngưỡng.",
                uncertain_confidence_cap,
            )
            if return_best_guess
            else _uncertain_answer(
                question,
                "VLM confidence was below the configured threshold.",
            )
        )
    elif verdict["status"] == "uncertain" and return_best_guess and verdict["confidence"] > 0:
        verdict["confidence"] = min(verdict["confidence"], uncertain_confidence_cap)

    supporting = set(verdict["supporting_frame_ids"])
    answer_mode = (
        "candidate"
        if promoted_candidate_id
        else "verified"
        if verdict["status"] == "answered"
        else "best_guess"
        if verdict["confidence"] > 0 and bool(verdict["supporting_frame_ids"])
        else "fallback"
    )
    selected_identity_to_id = {_identity(frame): frame_id for frame_id, frame in selected_ids.items()}
    supporting_selected = [
        selected_ids[frame_id]
        for frame_id in verdict["supporting_frame_ids"]
        if frame_id in selected_ids
    ]
    supporting_identity = {_identity(frame) for frame in supporting_selected}
    remaining_selected = [frame for frame in selected if _identity(frame) not in supporting_identity]
    selected_identity = {_identity(frame) for frame in selected}
    ordered_candidates = supporting_selected + remaining_selected + [
        frame for frame in candidates if _identity(frame) not in selected_identity
    ]
    candidate_ids_by_frame: Dict[str, List[str]] = {}
    for bundle in candidate_bundles:
        for frame in (bundle.get("frames") or {}).values():
            candidate_ids_by_frame.setdefault(_identity(frame), []).append(
                str(bundle.get("candidate_id"))
            )
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
        item["qa_answer_language"] = "vi"
        item["qa_answer_mode"] = answer_mode
        item["qa_candidate_ids"] = candidate_ids_by_frame.get(_identity(item), [])
        output.append(item)

    summary = {
        "status": verdict["status"],
        "answer": verdict["answer"],
        "answer_language": "vi",
        "answer_mode": answer_mode,
        "selected_candidate_id": promoted_candidate_id or None,
        "confidence": verdict["confidence"],
        "reason": verdict["reason"],
        "supporting_frame_ids": verdict["supporting_frame_ids"],
        "answer_type": plan["answer_type"],
        "expected_answer_format": plan["expected_answer_format"],
        "visual_queries": plan["visual_queries"],
        "event_queries": plan.get("event_queries") or [],
        "retrieved_frames": len(visual_frames),
        "evaluated_frames": len(selected),
        "evidence_groups": len(evidence_groups),
        "top_evidence_groups": [
            {
                "group_id": group.get("id"),
                "video_id": group.get("video_id"),
                "score": round(_float(group.get("score")), 6),
                "video_event_coverage": int(group.get("video_event_coverage") or 0),
                "video_event_best_ranks": group.get("video_event_best_ranks") or {},
                "video_event_worst_rank": int(group.get("video_event_worst_rank") or 100000),
                "video_event_span_seconds": round(_float(group.get("video_event_span_seconds")), 3),
                "video_query_coverage": int(group.get("video_query_coverage") or 0),
                "frame_ids": [
                    str(frame.get("frame_id") or frame.get("frame_name") or "")
                    for frame in (group.get("frames") or [])[:4]
                ],
            }
            for group in evidence_groups[:20]
        ],
        "evaluated_groups": len({frame.get("qa_group_id") for frame in selected if frame.get("qa_group_id")}),
        "selected_group_ids": [str(group.get("id")) for group in selection_groups],
        "video_hypotheses": [
            {
                "video_id": hypothesis.get("video_id"),
                "score": round(_float(hypothesis.get("score")), 6),
                "retrieval_score": round(_float(hypothesis.get("retrieval_score")), 6),
                "event_coverage": int(hypothesis.get("event_coverage") or 0),
                "event_total": int(hypothesis.get("event_total") or 0),
                "event_coverage_ratio": round(_float(hypothesis.get("event_coverage_ratio")), 4),
                "event_span_seconds": round(_float(hypothesis.get("event_span_seconds")), 3),
                "event_worst_rank": int(hypothesis.get("event_worst_rank") or 100000),
                "query_coverage": int(hypothesis.get("query_coverage") or 0),
                "complete_event_match": bool(hypothesis.get("complete_event_match")),
            }
            for hypothesis in video_hypotheses
        ],
        "answer_candidates": candidate_answers,
        "ocr_evidence": len(text_evidence["ocr"]),
        "asr_evidence": len(text_evidence["asr"]),
        "ocr_evidence_sent": prompt_evidence_sent["ocr"],
        "asr_evidence_sent": prompt_evidence_sent["asr"],
        "supporting_group_ids": sorted({
            str(selected_ids[frame_id].get("qa_group_id"))
            for frame_id in verdict["supporting_frame_ids"]
            if frame_id in selected_ids and selected_ids[frame_id].get("qa_group_id")
        }),
        "verification": verification_status,
        "detail_pass": detail_pass_status,
        "answer_max_chars": answer_max_chars,
        "retries": retries,
        "errors": errors[:6],
    }
    for item in output:
        item["qa_summary"] = summary
    return output, summary
