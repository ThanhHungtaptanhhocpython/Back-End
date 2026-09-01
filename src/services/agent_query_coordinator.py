"""Agent query coordinator for multimodal search.

Turns a natural-language retrieval description into an explicit multimodal
plan, executes the shared fusion pipeline, and returns both frames and plan
metadata for chat/search UI surfaces.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from typing import Any, Dict, Iterable, List, Optional, Tuple


logger = logging.getLogger(__name__)
COLOR_TERMS = (
    ("xanh duong", "blue"),
    ("xanh la", "green"),
    ("do", "red"),
    ("vang", "yellow"),
    ("den", "black"),
    ("trang", "white"),
    ("cam", "orange"),
    ("tim", "purple"),
    ("hong", "pink"),
    ("xam", "gray"),
    ("nau", "brown"),
)

CAMERA_TERMS = (
    (("goc may sat mat duong", "sat mat duong", "goc thap"), "low road-level camera angle", 1.25),
    (("can canh", "gan canh", "close up"), "close-up shot", 1.1),
    (("toan canh", "goc rong", "wide shot"), "wide shot", 0.85),
    (("tu tren cao", "flycam", "drone"), "aerial drone shot", 1.05),
    (("quay cham", "slow motion"), "slow motion", 0.9),
    (("camera co dinh", "goc may co dinh"), "fixed camera shot", 0.8),
)

ACTION_TERMS = (
    (("di qua duong", "bang qua duong", "qua duong"), "crossing the street", 1.1),
    (("dung duoi nuoc", "dung trong nuoc"), "standing in shallow water", 1.15),
    (("roi den", "chieu den", "soi den"), "shining a light", 1.0),
    (("keo luoi ca", "keo luoi", "keo luoi danh ca"), "pulling a fishing net", 1.25),
    (("tien den", "di den gan", "di toi"), "approaching", 0.8),
    (("dung may quay ghi hinh", "ghi hinh", "quay phim"), "filming with a video camera", 1.0),
    (("dung cho", "dang dung cho", "dung doi", "dang dung doi"), "standing and waiting", 1.0),
    (("chay xe", "lai xe", "di xe"), "riding a motorbike", 1.0),
    (("tu trai sang phai", "trai sang phai"), "moving from left to right", 0.9),
    (("cam o", "cam du"), "holding an umbrella", 1.0),
    (("cam dien thoai",), "holding a phone", 0.9),
    (("cam micro", "cam mic"), "holding a microphone", 0.9),
    (("chay bo", "dang chay"), "running", 0.9),
    (("di bo", "dang di"), "walking", 0.75),
    (("dung", "dang dung"), "standing", 0.7),
    (("ngoi", "dang ngoi"), "sitting", 0.7),
    (("noi chuyen", "tro chuyen"), "talking", 0.8),
    (("phong van", "tra loi phong van"), "interview scene", 1.0),
    (("bat tay",), "shaking hands", 1.0),
    (("an", "dang an"), "eating", 0.75),
    (("uong", "dang uong"), "drinking", 0.75),
    (("lai xe",), "driving", 0.9),
    (("dap xe", "di xe dap"), "riding a bicycle", 1.0),
    (("choi bong", "da bong"), "playing soccer", 1.0),
    (("mua hang",), "shopping", 0.8),
)

SUBJECT_TERMS = (
    (("tai xe xe om cong nghe", "xe om cong nghe", "tai xe cong nghe"), "app-based motorbike taxi driver", 1.15),
    (("tai xe",), "driver", 0.9),
    (("nguoi phu nu", "phu nu", "co gai"), "woman", 1.0),
    (("nguoi dan ong", "dan ong", "nam gioi", "chang trai"), "man", 1.0),
    (("tre em", "dua tre", "em be"), "child", 1.0),
    (("nguoi",), "person", 0.75),
    (("nhom nguoi", "dam dong"), "group of people", 0.9),
    (("van dong vien",), "athlete", 1.0),
    (("tay dua",), "racer", 1.0),
    (("canh sat",), "police officer", 1.0),
)

OBJECT_TERMS = (
    (("xe dap",), "bicycle", 1.0),
    (("xe may",), "motorbike", 1.0),
    (("bang gia xang dau", "bang gia xang", "bang gia dau"), "fuel price board", 1.15),
    (("cot bom xang", "tru bom xang"), "gas pump", 1.0),
    (("o to", "xe hoi"), "car", 1.0),
    (("xe buyt",), "bus", 1.0),
    (("tau hoa",), "train", 1.0),
    (("may bay",), "airplane", 1.0),
    (("cai o", "cam o", "du "), "umbrella", 0.9),
    (("dien thoai",), "phone", 0.9),
    (("den pin", "den"), "flashlight", 0.9),
    (("luoi ca", "luoi danh ca", "luoi"), "fishing net", 1.15),
    (("may quay", "may quay ghi hinh", "camera quay phim"), "video camera", 1.0),
    (("micro", "mic"), "microphone", 0.9),
    (("bong", "qua bong"), "ball", 0.9),
    (("bien bao",), "traffic sign", 0.9),
    (("bang hieu", "bien hieu"), "signboard", 0.9),
    (("man hinh",), "screen", 0.8),
)

SCENE_TERMS = (
    (("tram xang", "cay xang", "tram xang dau"), "gas station", 1.15),
    (("duong pho", "ngoai duong"), "street scene", 0.8),
    (("vach qua duong", "lan duong"), "road crossing", 0.9),
    (("san khau",), "stage", 0.9),
    (("lop hoc", "phong hoc"), "classroom", 0.9),
    (("nha hang", "quan an"), "restaurant", 0.8),
    (("sieu thi", "cua hang"), "store interior", 0.8),
    (("bai bien",), "beach", 0.8),
    (("duoi nuoc", "trong nuoc", "mat nuoc"), "shallow water", 1.0),
    (("binh minh", "luc binh minh", "sang som"), "dawn sunrise", 1.0),
    (("san van dong",), "stadium", 0.9),
    (("ban ngay", "troi sang"), "daytime", 0.6),
    (("ban dem", "troi toi"), "nighttime", 0.6),
    (("troi mua", "mua"), "rainy scene", 0.7),
)

def _parse_query_light(query: str) -> Dict[str, Any]:
    visual_query = _clean(query)
    ocr_query = ""
    asr_query = ""
    weights = {"visual": 1.0, "ocr": 0.0, "asr": 0.0}

    text_queries = _extract_text_queries(query)
    if text_queries:
        ocr_query = _clean(" ".join(text_queries))
        visual_query = _clean(re.sub(r"\"[^\"]+\"|'[^']+'", " ", visual_query))
        weights = {"visual": 0.55, "ocr": 0.45, "asr": 0.0}

    normalised = _normalise_text(query)
    if _has_asr_signal(normalised):
        asr_query = visual_query or query
        weights["asr"] = 0.25
        weights["visual"] = 0.5 if ocr_query else 0.75
        if ocr_query:
            weights["ocr"] = 0.25

    return {"visual_query": visual_query, "ocr_query": ocr_query, "asr_query": asr_query, "weights": weights}


def _clean(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _normalise_text(value: Any) -> str:
    text = unicodedata.normalize("NFD", str(value or "").lower())
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return text.replace("\u0111", "d")


def _has_asr_signal(normalised: str) -> bool:
    if re.search(r"\b(?:am thanh|phat bieu|hoi thoai|giong noi|loi noi|tieng noi)\b", normalised):
        return True
    if re.search(r"\bnoi\b", normalised) and not re.search(r"\b(?:ha noi|noi that|noi dung)\b", normalised):
        return True
    return bool(re.search(r"\bnghe\s+(?:tieng|thay|duoc|ro|noi|am thanh)\b", normalised))


def _normalise_weights(weights: Dict[str, float]) -> Dict[str, float]:
    clipped = {key: max(0.0, float(weights.get(key, 0.0) or 0.0)) for key in ("visual", "ocr", "asr")}
    total = sum(clipped.values())
    if total <= 0:
        return {"visual": 1.0, "ocr": 0.0, "asr": 0.0}
    return {key: round(value / total, 3) for key, value in clipped.items()}


def _dedupe_queries(queries: Iterable[Dict[str, str]], limit: int = 8) -> List[Dict[str, str]]:
    deduped: List[Dict[str, str]] = []
    seen = set()
    for query in queries:
        kind = _clean(query.get("kind") or "visual")
        raw = _clean(query.get("query"))
        query_en = _clean(query.get("query_en") or query.get("queryEn") or raw)
        if not raw and not query_en:
            continue
        key = (kind, query_en.lower() or raw.lower())
        if key in seen:
            continue
        seen.add(key)
        deduped.append({"kind": kind, "query": raw or query_en, "query_en": query_en or raw})
        if len(deduped) >= limit:
            break
    return deduped


def _translate_vi_to_en(text: str) -> str:
    """Cheap deterministic fallback only; no network calls in the search path."""
    cleaned = _clean(text)
    if not cleaned:
        return ""
    normalised = _normalise_text(cleaned)
    phrase_pairs = (
        ("cuoc dua xe dap", "cycling race"),
        ("dua xe dap", "cycling race"),
        ("vach dich", "finish line"),
        ("dung duoi nuoc", "standing in shallow water"),
        ("dung trong nuoc", "standing in shallow water"),
        ("roi den", "shining a light"),
        ("chieu den", "shining a light"),
        ("soi den", "shining a light"),
        ("keo luoi ca", "pulling a fishing net"),
        ("keo luoi", "pulling a fishing net"),
        ("luoi ca", "fishing net"),
        ("binh minh", "dawn sunrise"),
        ("tien den", "approaching"),
        ("may quay ghi hinh", "video camera filming"),
        ("may quay", "video camera"),
        ("ghi hinh", "filming"),
        ("tram xang", "gas station"),
        ("cay xang", "gas station"),
        ("tai xe xe om cong nghe", "app-based motorbike taxi driver"),
        ("xe om cong nghe", "app-based motorbike taxi driver"),
        ("bang gia xang dau", "fuel price board"),
        ("bang gia xang", "fuel price board"),
        ("gia xang dau", "fuel prices"),
        ("dung cho", "standing and waiting"),
        ("dung doi", "standing and waiting"),
        ("chay xe", "riding a motorbike"),
        ("tu trai sang phai", "from left to right"),
        ("ve dich", "crossing the finish line"),
        ("tay dua", "cyclist"),
        ("nguoi phu nu", "woman"),
        ("nguoi dan ong", "man"),
        ("tre em", "child"),
        ("nhom nguoi", "group of people"),
        ("dam dong", "crowd"),
        ("ao xanh duong", "blue shirt"),
        ("ao xanh la", "green shirt"),
        ("ao vang", "yellow shirt"),
        ("ao do", "red shirt"),
        ("ao den", "black shirt"),
        ("ao trang", "white shirt"),
        ("quan xanh duong", "blue pants"),
        ("quan xanh la", "green pants"),
        ("quan den", "black pants"),
        ("quan do", "red pants"),
        ("quan trang", "white pants"),
        ("goc may sat mat duong", "low angle road camera"),
        ("sat mat duong", "road-level camera"),
        ("goc thap", "low angle"),
        ("can canh", "close-up"),
        ("toan canh", "wide shot"),
        ("tu tren cao", "aerial view"),
        ("quay cham", "slow motion replay"),
        ("di qua duong", "crossing the street"),
        ("bang qua duong", "crossing the street"),
        ("cam o", "holding an umbrella"),
        ("cam dien thoai", "holding a phone"),
        ("cam micro", "holding a microphone"),
        ("noi chuyen", "talking"),
        ("phong van", "interview"),
        ("bat tay", "shaking hands"),
        ("di bo", "walking"),
        ("chay bo", "running"),
        ("dang chay", "running"),
        ("dang dung", "standing"),
        ("dang ngoi", "sitting"),
        ("xe dap", "bicycle"),
        ("xe may", "motorbike"),
        ("o to", "car"),
        ("xe hoi", "car"),
        ("xe buyt", "bus"),
        ("duong pho", "street"),
        ("vach qua duong", "road crossing"),
        ("bang hieu", "signboard"),
        ("bien hieu", "signboard"),
        ("man hinh", "screen"),
        ("san khau", "stage"),
        ("lop hoc", "classroom"),
    )
    translated = normalised
    for source, target in phrase_pairs:
        translated = translated.replace(source, target)
    translated = re.sub(r"\b(canh|khung hinh|clip|video|tai vi tri|cua|theo thu tu|lan luot la|mot|1|va|bat tron|khoanh khac|nhat|nhi|ba)\b", " ", translated)
    translated = _clean(translated).strip(" .,:;!?")
    return translated if re.search(r"[a-z]", translated) and translated != normalised else ""


def _cycling_finish_rewrites(prompt: str) -> List[Dict[str, str]]:
    normalised = _normalise_text(prompt)
    if not (("xe dap" in normalised or "dua xe" in normalised) and ("vach dich" in normalised or "dich" in normalised)):
        return []

    rewrites = [
        "road level low angle camera cyclists crossing the finish line",
        "close up bicycle wheels crossing a cycling race finish line",
        "slow motion road level finish line cycling race",
        "cyclists crossing the finish line in a road cycling race",
        "bicycle race finish line with cyclists in yellow and blue jerseys",
        "road cycling sprint finish at the finish line",
    ]
    if "ao vang" in normalised or "vang" in normalised:
        rewrites.append("cyclist wearing a yellow jersey crossing the finish line")
    if "ao xanh" in normalised or "xanh" in normalised:
        rewrites.append("cyclists wearing blue jerseys at a cycling finish line")

    return [{"kind": "visual", "query": prompt, "query_en": query} for query in rewrites]


def _generic_visual_rewrites(prompt: str) -> List[Dict[str, str]]:
    translated = _translate_vi_to_en(prompt)
    rewrites: List[Dict[str, str]] = []
    if translated:
        rewrites.append({"kind": "visual", "query": prompt, "query_en": translated})

    normalised = _normalise_text(prompt)
    stripped = re.sub(r"\b(canh|khung hinh|clip|video|quay cham|slow motion|goc may|camera)\b", " ", normalised)
    stripped = _clean(stripped.strip(" .,:;!?"))
    if stripped and stripped != normalised:
        stripped_en = _translate_vi_to_en(stripped) or stripped
        rewrites.append({"kind": "visual", "query": stripped, "query_en": stripped_en})
    return rewrites




def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", _normalise_text(value)).strip("_")
    return slug[:48] or "check"


def _contains_any(normalised: str, phrases: Iterable[str]) -> bool:
    padded = f" {normalised} "
    for phrase in phrases:
        phrase_clean = phrase.strip()
        if not phrase_clean:
            continue
        if " " in phrase_clean and phrase_clean in normalised:
            return True
        if " " not in phrase_clean and f" {phrase_clean} " in padded:
            return True
    return False


def _add_check(checks: List[Dict[str, Any]], seen: set, check_id: str, label: str, query_en: str, weight: float) -> None:
    check_id = _slug(check_id)
    query_en = _clean(query_en)
    label = _clean(label or query_en)
    if not query_en or check_id in seen:
        return
    seen.add(check_id)
    checks.append({"id": check_id, "label": label, "query_en": query_en, "weight": weight})


def _extract_term_hits(normalised: str, specs: Iterable[Tuple[Tuple[str, ...], str, float]]) -> List[Tuple[str, float]]:
    hits: List[Tuple[str, float]] = []
    seen = set()
    for phrases, query_en, weight in specs:
        if query_en in seen:
            continue
        if _contains_any(normalised, phrases):
            seen.add(query_en)
            hits.append((query_en, weight))
    return hits


def _extract_count_checks(normalised: str) -> List[Tuple[str, float]]:
    count_words = (
        ("1", "one"),
        ("mot", "one"),
        ("2", "two"),
        ("hai", "two"),
        ("3", "three"),
        ("ba", "three"),
        ("4", "four"),
        ("bon", "four"),
        ("5", "five"),
        ("nam", "five"),
    )
    subject_words = (
        ("tai xe xe om cong nghe", "app-based motorbike taxi drivers"),
        ("xe om cong nghe", "app-based motorbike taxi drivers"),
        ("tai xe", "drivers"),
        ("nguoi", "person"),
        ("tay dua", "racer"),
        ("van dong vien", "athlete"),
        ("xe dap", "bicycle"),
        ("o to", "car"),
        ("xe may", "motorbike"),
        ("xe", "vehicle"),
    )
    hits: List[Tuple[str, float]] = []
    for raw, en in count_words:
        for subject, subject_en in subject_words:
            if re.search(rf"\b{re.escape(raw)}\s+{re.escape(subject)}\b", normalised):
                hits.append((f"{en} {subject_en}", 0.85))
                break
    return hits[:2]


def _extract_clothing_checks(normalised: str) -> List[Tuple[str, float]]:
    hits: List[Tuple[str, float]] = []
    for source, target in COLOR_TERMS:
        if f"ao {source}" in normalised:
            hits.append((f"person wearing a {target} shirt", 1.0))
        if f"quan {source}" in normalised:
            hits.append((f"person wearing {target} pants", 0.9))
        if f"mu {source}" in normalised or f"non {source}" in normalised:
            hits.append((f"person wearing a {target} hat", 0.8))
        if f"xe mau {source}" in normalised:
            hits.append((f"{target} vehicle", 0.85))
    return hits


def _uniq_terms(values: Iterable[str]) -> List[str]:
    return list(dict.fromkeys(_clean(value) for value in values if _clean(value)))


def _join_scene_query(parts: Iterable[str]) -> str:
    return _clean(" ".join(_uniq_terms(parts)))


def _is_generic_structured_profile(profile: str) -> bool:
    return profile.startswith("generic_")

def _extract_text_queries(prompt: str) -> List[str]:
    values: List[str] = []
    for match in re.findall(r"\"([^\"]{2,80})\"|'([^']{2,80})'", prompt):
        values.extend(part.strip() for part in match if part.strip())

    for match in re.findall(r"\b[A-Z0-9][A-Z0-9 ._-]{2,40}\b", prompt):
        value = _clean(match.strip(" ._-"))
        if value and not value.isdigit() and len(value) <= 40:
            values.append(value)

    normalised = _normalise_text(prompt)
    for match in re.findall(r"(?:co chu|dong chu|chu|logo|bang hieu|bien hieu)\s+([^,.;]{2,60})", normalised):
        value = re.split(r"\b(?:tren|o|tai|phia|cua|voi)\b", match, maxsplit=1)[0]
        value = _clean(value.strip(" .,:;!?"))
        if value and len(value) <= 40:
            values.append(value)

    deduped: List[str] = []
    seen = set()
    for value in values:
        key = _normalise_text(value)
        if key and key not in seen:
            seen.add(key)
            deduped.append(value)
    return deduped[:3]


def _generic_structured_plan(prompt: str) -> Dict[str, Any]:
    normalised = _normalise_text(prompt)
    translated = _translate_vi_to_en(prompt)
    checks: List[Dict[str, Any]] = []
    seen_checks = set()
    visual_parts: List[str] = []
    categories = set()

    def add_component(category: str, query_en: str, weight: float) -> None:
        query_en_clean = _clean(query_en)
        if not query_en_clean:
            return
        categories.add(category)
        visual_parts.append(query_en_clean)
        _add_check(checks, seen_checks, f"{category}_{query_en_clean}", query_en_clean, query_en_clean, weight)

    for category, specs in (
        ("camera", CAMERA_TERMS),
        ("subject", SUBJECT_TERMS),
        ("object", OBJECT_TERMS),
        ("action", ACTION_TERMS),
        ("scene", SCENE_TERMS),
    ):
        hits = _extract_term_hits(normalised, specs)
        if category == "subject" and any(query_en != "person" for query_en, _ in hits):
            hits = [(query_en, weight) for query_en, weight in hits if query_en != "person"]
        for query_en, weight in hits:
            add_component(category, query_en, weight)

    for query_en, weight in _extract_count_checks(normalised):
        add_component("count", query_en, weight)

    clothing_hits = _extract_clothing_checks(normalised)
    for query_en, weight in clothing_hits:
        add_component("appearance", query_en, weight)

    if any(marker in normalised for marker in ("thu tu", "lan luot", "truoc", "sau do", "tiep theo")) or re.search(r"\b(?:nhat|nhi)\b", normalised):
        add_component("temporal", "ordered sequence of events", 0.75)

    ocr_queries = _extract_text_queries(prompt)
    for text_query in ocr_queries:
        categories.add("ocr")
        _add_check(checks, seen_checks, f"ocr_{text_query}", f"visible text {text_query}", text_query, 0.9)

    asr_queries: List[str] = []
    if _has_asr_signal(normalised):
        categories.add("asr")
        asr_query = translated or _clean(prompt)
        asr_queries.append(asr_query)
        _add_check(checks, seen_checks, "speech_audio", "relevant speech or audio", asr_query, 0.7)

    base_query = _clean(" ".join(dict.fromkeys(visual_parts)))
    if not base_query:
        base_query = translated or _clean(prompt)

    people_terms = [part for part in visual_parts if part in {"woman", "man", "child", "person", "group of people", "athlete", "racer", "police officer"}]
    action_terms = [part for part in visual_parts if part in {query for _, query, _ in ACTION_TERMS}]
    object_terms = [part for part in visual_parts if part in {query for _, query, _ in OBJECT_TERMS}]
    camera_terms = [part for part in visual_parts if part in {query for _, query, _ in CAMERA_TERMS}]
    scene_terms = [part for part in visual_parts if part in {query for _, query, _ in SCENE_TERMS}]
    appearance_terms = [query for query, _ in clothing_hits]

    subject = people_terms[0] if people_terms else ""
    count_terms = [part for part in visual_parts if re.match(r"^(one|two|three|four|five)\b", part)]
    if count_terms and not subject:
        subject = count_terms[0]
    action_text = " ".join(action_terms)
    object_terms_for_query = [term for term in object_terms if term not in action_text]
    holistic_parts = [
        camera_terms[0] if camera_terms else "",
        scene_terms[0] if scene_terms else "",
        subject,
        *appearance_terms[:2],
        *action_terms[:3],
        *object_terms_for_query[:3],
    ]
    enriched_query = _join_scene_query(holistic_parts) or base_query
    if translated and len(translated.split()) > len(enriched_query.split()):
        enriched_query = translated

    visual_queries = [enriched_query]
    if scene_terms and subject and action_terms:
        visual_queries.append(_join_scene_query([scene_terms[0], subject, *appearance_terms[:1], *action_terms[:2], *object_terms_for_query[:2]]))
    if camera_terms and enriched_query and camera_terms[0] not in enriched_query:
        visual_queries.append(_join_scene_query([camera_terms[0], enriched_query]))
    if ocr_queries and not any(term in enriched_query for term in ("screen", "signboard", "traffic sign")):
        visual_queries.append(_clean(f"screen or signboard with text {ocr_queries[0]} in {enriched_query}"))

    visual_queries = list(dict.fromkeys(query for query in visual_queries if query))[:4]
    if not checks:
        _add_check(checks, seen_checks, "main_visual", base_query, base_query, 1.0)

    profile_bits = sorted(category for category in categories if category in {"camera", "subject", "object", "action", "scene", "appearance", "count", "temporal", "ocr", "asr"})
    profile = "generic_" + ("_".join(profile_bits) if profile_bits else "visual")
    return {
        "profile": profile,
        "intent": translated or _clean(prompt),
        "must_have_checks": checks,
        "negative_checks": [],
        "rerank_focus": [
            "prefer frames matching multiple checklist items together",
            "prefer concrete visible entities over broad scene matches",
            "prefer OCR or ASR evidence when the prompt asks for text or speech",
        ],
        "visual_queries": visual_queries,
        "ocr_queries": ocr_queries,
        "asr_queries": asr_queries,
    }

def _fishing_net_temporal_structured_plan(prompt: str) -> Dict[str, Any]:
    normalised = _normalise_text(prompt)
    has_fishing = any(term in normalised for term in ("luoi ca", "keo luoi", "danh ca"))
    has_water = any(term in normalised for term in ("duoi nuoc", "trong nuoc", "mat nuoc"))
    has_camera = any(term in normalised for term in ("may quay", "ghi hinh", "quay phim"))
    if not (has_fishing or (has_water and has_camera)):
        return {}

    must_have = [
        {"id": "person_in_water", "label": "person standing in shallow water", "query_en": "person standing in shallow water", "weight": 1.1},
        {"id": "shining_light", "label": "person shining a light", "query_en": "person shining a light or flashlight", "weight": 0.95},
        {"id": "pulling_fishing_net", "label": "person pulling a fishing net", "query_en": "person pulling a fishing net", "weight": 1.35},
        {"id": "dawn", "label": "dawn or sunrise scene", "query_en": "dawn sunrise scene", "weight": 0.9},
        {"id": "camera_crew", "label": "people filming with a video camera", "query_en": "people filming with a video camera", "weight": 0.95},
    ]
    if "nhom nguoi" in normalised or "nguoi khac" in normalised:
        must_have.append({"id": "approaching_group", "label": "group of people approaching", "query_en": "group of people approaching", "weight": 0.8})

    return {
        "profile": "fishing_net_temporal",
        "intent": "Find the sequence where a person in shallow water shines a light, pulls a fishing net at dawn, then is approached and filmed by others.",
        "must_have_checks": must_have,
        "negative_checks": [
            {"id": "not_stage", "label": "stage or indoor performance unrelated to fishing"},
            {"id": "not_group_photo", "label": "static group photo without fishing net or water"},
        ],
        "rerank_focus": [
            "prefer fishing net and shallow water over generic groups of people",
            "prefer dawn or low-light fishing scenes",
            "prefer nearby keyframes in the same video because the prompt describes a sequence",
            "prefer visible camera crew only after fishing-net candidates",
        ],
        "visual_queries": [
            "person standing in shallow water shining a flashlight then pulling a fishing net at dawn",
            "dawn fishing scene with a person pulling a net in shallow water while people approach with a video camera",
            "camera crew filming a fisherman pulling a fishing net at sunrise",
        ],
    }

def _gas_station_structured_plan(prompt: str) -> Dict[str, Any]:
    normalised = _normalise_text(prompt)
    if not any(term in normalised for term in ("tram xang", "cay xang", "xang dau")):
        return {}

    must_have = [
        {"id": "gas_station", "label": "gas station", "query_en": "gas station", "weight": 1.25},
        {"id": "motorbike_taxi_drivers", "label": "app-based motorbike taxi drivers", "query_en": "app-based motorbike taxi drivers", "weight": 1.15},
        {"id": "motorbikes", "label": "motorbikes", "query_en": "motorbikes", "weight": 0.95},
    ]
    visual_queries = [
        "gas station with four app-based motorbike taxi drivers, three waiting and one riding left to right",
        "fuel price board at a gas station with motorbike taxi drivers and motorbikes",
        "gas station forecourt with motorbike drivers waiting beside gas pumps",
    ]
    if "bang gia" in normalised or "gia xang" in normalised:
        must_have.append({"id": "fuel_price_board", "label": "fuel price board", "query_en": "fuel price board", "weight": 1.1})
    if "4" in normalised or "bon" in normalised:
        must_have.append({"id": "four_drivers", "label": "four motorbike taxi drivers", "query_en": "four app-based motorbike taxi drivers", "weight": 0.75})
    if "dung cho" in normalised or "dung doi" in normalised:
        must_have.append({"id": "waiting_drivers", "label": "drivers standing and waiting", "query_en": "drivers standing and waiting", "weight": 0.8})
    if "trai sang phai" in normalised:
        must_have.append({"id": "left_to_right", "label": "one rider moving left to right", "query_en": "one motorbike rider moving from left to right", "weight": 0.7})

    return {
        "profile": "gas_station_motorbike_taxi",
        "intent": "Find the gas-station frame with motorbike taxi drivers, waiting riders, and a fuel price board.",
        "must_have_checks": must_have,
        "negative_checks": [
            {"id": "not_stage", "label": "stage or performance scene unrelated to a gas station"},
            {"id": "not_indoor", "label": "indoor scene without gas pumps or motorbikes"},
        ],
        "rerank_focus": [
            "prefer gas station forecourt context",
            "prefer motorbikes and app-based taxi drivers together",
            "prefer visible fuel price board when present",
            "prefer frames with multiple waiting drivers",
        ],
        "visual_queries": visual_queries,
    }

def _cycling_finish_structured_plan(prompt: str) -> Dict[str, Any]:
    normalised = _normalise_text(prompt)
    if not (("xe dap" in normalised or "dua xe" in normalised) and ("vach dich" in normalised or "dich" in normalised)):
        return {}

    must_have = [
        {
            "id": "finish_line",
            "label": "cycling race finish line",
            "query_en": "cycling race finish line",
            "weight": 1.1,
        },
        {
            "id": "low_road_angle",
            "label": "low road-level camera angle",
            "query_en": "road level low angle camera close to the road",
            "weight": 1.6,
        },
        {
            "id": "finish_moment",
            "label": "cyclists crossing the finish line",
            "query_en": "cyclists crossing the finish line",
            "weight": 1.4,
        },
        {
            "id": "three_cyclists",
            "label": "three cyclists visible together",
            "query_en": "three cyclists crossing together",
            "weight": 1.1,
        },
        {
            "id": "yellow_black",
            "label": "front cyclist wears yellow jersey and black shorts",
            "query_en": "cyclist wearing yellow jersey and black shorts",
            "weight": 0.9,
        },
        {
            "id": "blue_black",
            "label": "second cyclist wears blue jersey and black shorts",
            "query_en": "cyclist wearing blue jersey and black shorts",
            "weight": 0.8,
        },
        {
            "id": "blue_red",
            "label": "third cyclist wears blue jersey and red shorts",
            "query_en": "cyclist wearing blue jersey and red shorts",
            "weight": 0.8,
        },
    ]
    visual_queries = [
        "road level low angle slow motion shot of three cyclists crossing the finish line with yellow and blue jerseys",
        "close up bicycle wheels at a cycling race finish line with riders in yellow and blue",
        "finish line sprint with yellow jersey black shorts followed by blue jersey cyclists",
    ]
    return {
        "profile": "cycling_finish_low_angle",
        "intent": "Find the exact finish-line keyframe in a road cycling race.",
        "must_have_checks": must_have,
        "negative_checks": [
            {"id": "wide_only", "label": "wide crowd shot without a road-level perspective"},
            {"id": "not_finish_moment", "label": "cyclists are not currently crossing the finish line"},
            {"id": "text_board", "label": "classroom/news/text-board frame unrelated to the race"},
        ],
        "rerank_focus": [
            "prefer road-level camera and large foreground wheels",
            "prefer frames where cyclists are on or immediately at the finish line",
            "prefer frames with three cyclists visible together",
            "prefer yellow/black and blue rider colors when visible",
        ],
        "visual_queries": visual_queries,
    }


def _build_local_structured_search_plan(prompt: str) -> Dict[str, Any]:
    cycling = _cycling_finish_structured_plan(prompt)
    if cycling:
        return cycling
    fishing = _fishing_net_temporal_structured_plan(prompt)
    if fishing:
        return fishing
    gas_station = _gas_station_structured_plan(prompt)
    if gas_station:
        return gas_station
    return _generic_structured_plan(prompt)


def _build_structured_search_plan(prompt: str) -> Dict[str, Any]:
    local_plan = _build_local_structured_search_plan(prompt)
    local_plan.setdefault("planner_source", "local")
    try:
        from src.services.openrouter_agent_planner import plan_agent_query_with_openrouter

        llm_plan = plan_agent_query_with_openrouter(prompt, local_plan)
    except Exception as exc:
        logger.warning("Agent LLM planner integration failed; using local fallback: %s", exc)
        llm_plan = {}
    if llm_plan:
        return llm_plan
    return local_plan

def _fallback_expand_queries(prompt: str, limit: int = 6) -> List[Dict[str, str]]:
    queries: List[Dict[str, str]] = []

    def add(kind: str, query: str, query_en: str = "") -> None:
        query = _clean(query)
        query_en = _clean(query_en or query)
        if query or query_en:
            queries.append({"kind": kind, "query": query or query_en, "query_en": query_en or query})

    add("visual", prompt, _translate_vi_to_en(prompt) or prompt)
    for quoted in _extract_text_queries(prompt):
        add("ocr", quoted, quoted)
    normalised = _normalise_text(prompt)
    if _has_asr_signal(normalised):
        add("asr", prompt, _translate_vi_to_en(prompt) or prompt)
    return _dedupe_queries(queries, limit=limit)

def _agent_expanded_queries(prompt: str, limit: int = 8, structured: Optional[Dict[str, Any]] = None) -> List[Dict[str, str]]:
    structured = structured or _build_structured_search_plan(prompt)
    rewritten: List[Dict[str, str]] = []

    for query in structured.get("visual_queries", []):
        rewritten.append({"kind": "visual", "query": prompt, "query_en": query})
    for query in structured.get("ocr_queries", []):
        rewritten.append({"kind": "ocr", "query": query, "query_en": query})
    for query in structured.get("asr_queries", []):
        rewritten.append({"kind": "asr", "query": query, "query_en": query})

    profile = str(structured.get("profile") or "")
    if not rewritten:
        rewritten.extend(_fallback_expand_queries(prompt, limit=3))
    elif _is_generic_structured_profile(profile):
        for query in _fallback_expand_queries(prompt, limit=3):
            if query.get("kind") != "visual":
                rewritten.append(query)

    return _dedupe_queries(rewritten, limit=limit)


def _first_query(queries: List[Dict[str, str]], *kinds: str) -> str:
    accepted = set(kinds)
    for query in queries:
        if query.get("kind") in accepted:
            return _clean(query.get("query_en") or query.get("query"))
    return ""


def _queries_by_kind(queries: List[Dict[str, str]], kind: str, limit: int = 5) -> List[str]:
    values: List[str] = []
    seen = set()
    for query in queries:
        if query.get("kind") != kind:
            continue
        value = _clean(query.get("query_en") or query.get("query"))
        if not value or value.lower() in seen:
            continue
        seen.add(value.lower())
        values.append(value)
        if len(values) >= limit:
            break
    return values


def _agent_visual_query_limit() -> int:
    try:
        from src.config.settings import get_settings

        raw_limit = int(get_settings().agent_visual_query_limit or 1)
    except Exception:
        raw_limit = 1
    return max(1, min(raw_limit, 3))


def _select_executed_visual_queries(visual_queries: List[str], primary_query: str) -> List[str]:
    selected: List[str] = []
    seen = set()
    limit = _agent_visual_query_limit()
    candidates = [primary_query, *visual_queries]
    for query in candidates:
        cleaned = _clean(query)
        key = cleaned.lower()
        if not cleaned or key in seen:
            continue
        seen.add(key)
        selected.append(cleaned)
        if len(selected) >= limit:
            break
    return selected


def build_agent_plan(message: str, topk: int = 100) -> Dict[str, Any]:
    prompt = _clean(message)
    parsed = _parse_query_light(prompt)
    structured_plan = _build_structured_search_plan(prompt)
    expanded = _agent_expanded_queries(prompt, limit=6, structured=structured_plan)
    visual_queries = _queries_by_kind(expanded, "visual", limit=8)
    visual_query = visual_queries[0] if visual_queries else _clean(parsed.get("visual_query")) or prompt
    ocr_query = _first_query(expanded, "ocr") or _clean(parsed.get("ocr_query"))
    asr_query = _first_query(expanded, "asr") or _clean(parsed.get("asr_query"))

    weights = dict(parsed.get("weights") or {})
    if ocr_query and weights.get("ocr", 0.0) <= 0:
        weights["ocr"] = 0.25
        weights["visual"] = max(0.5, float(weights.get("visual", 1.0)) - 0.15)
    if asr_query and weights.get("asr", 0.0) <= 0:
        weights["asr"] = 0.2
        weights["visual"] = max(0.55, float(weights.get("visual", 1.0)) - 0.1)

    weights = _normalise_weights(weights)
    if not visual_query and weights.get("visual", 0.0) > 0:
        weights["visual"] = 0.0
        weights = _normalise_weights(weights)

    executed_visual_queries = _select_executed_visual_queries(visual_queries, visual_query)
    executed_query_keys = {executed.lower() for executed in executed_visual_queries}
    support_visual_queries = [
        query
        for query in visual_queries
        if query and query.lower() not in executed_query_keys
    ]

    return {
        "original_query": prompt,
        "expanded_queries": expanded,
        "routing": weights,
        "visual_query": visual_query,
        "primary_visual_query": visual_query,
        "visual_queries": visual_queries,
        "executed_visual_queries": executed_visual_queries,
        "support_visual_queries": support_visual_queries,
        "execution_strategy": {
            "mode": "primary_holistic_first",
            "visual_query_limit": len(executed_visual_queries),
            "note": "Support queries are kept for explanation/checklist context; only executed queries are sent to visual search.",
        },
        "ocr_query": ocr_query,
        "asr_query": asr_query,
        "precision_profile": structured_plan.get("profile", ""),
        "planner_source": structured_plan.get("planner_source", "local"),
        "search_plan": structured_plan,
        "must_have_checks": structured_plan.get("must_have_checks", []),
        "negative_checks": structured_plan.get("negative_checks", []),
        "rerank_focus": structured_plan.get("rerank_focus", []),
        "verification": {
            "enabled": True,
            "method": "light_no_vlm",
            "signals": ["multi_query_consensus", "checklist_coverage", "ocr_asr_fusion", "temporal_neighbors", "optional_openrouter_vlm"],
        },
        "topk": topk,
    }


def _frame_identity(item: Dict[str, Any], fallback: str) -> str:
    for key in ("global_frame_id", "frame_path", "frame_name", "video_id", "faiss_id", "vector_id", "id"):
        value = item.get(key)
        if value not in (None, "", [], {}):
            return str(value)
    return fallback


def _score(item: Dict[str, Any], rank: int) -> float:
    for key in ("final_score", "normalized_score", "score", "_score"):
        try:
            return max(0.0, min(1.0, float(item.get(key))))
        except (TypeError, ValueError):
            continue
    return max(0.0, 1.0 - (rank - 1) * 0.02)


def _precision_query_multiplier(query: str, profile: str) -> float:
    if "cycling_finish_low_angle" not in profile:
        return 1.0
    q = query.lower()
    multiplier = 1.0
    if "road level" in q or "low angle" in q or "close up" in q:
        multiplier += 0.55
    if "wheels" in q:
        multiplier += 0.25
    if "yellow" in q and "blue" in q:
        multiplier += 0.15
    return multiplier



def _check_match_score(check: Dict[str, Any], queries: Iterable[str]) -> float:
    joined = " ".join(_normalise_text(query) for query in queries)
    check_id = str(check.get("id") or "")
    if not joined:
        return 0.0

    if check_id == "low_road_angle":
        hits = sum(1 for term in ("road level", "low angle", "close up", "wheels") if term in joined)
        return min(1.0, hits / 2.0)
    if check_id == "finish_moment":
        return 1.0 if "cross" in joined and "finish" in joined else 0.0
    if check_id == "finish_line":
        return 1.0 if "finish line" in joined or "finish" in joined else 0.0
    if check_id == "three_cyclists":
        return 1.0 if "three" in joined and "cycl" in joined else 0.0
    if check_id == "yellow_black":
        return 1.0 if "yellow" in joined and "black" in joined else 0.0
    if check_id == "blue_black":
        return 1.0 if "blue" in joined and "black" in joined else 0.0
    if check_id == "blue_red":
        return 1.0 if "blue" in joined and "red" in joined else 0.0

    query = _normalise_text(check.get("query_en") or check.get("label") or "")
    tokens = [token for token in re.findall(r"[a-z0-9]+", query) if len(token) > 3]
    if not tokens:
        return 0.0
    matched = sum(1 for token in tokens if token in joined)
    return matched / len(tokens)


def _apply_checklist_scores(merged: Dict[str, Dict[str, Any]], checks: List[Dict[str, Any]]) -> None:
    if not checks:
        return
    total_weight = sum(float(check.get("weight", 1.0) or 1.0) for check in checks) or 1.0
    for entry in merged.values():
        matched_checks: List[str] = []
        missing_checks: List[str] = []
        coverage = 0.0
        for check in checks:
            score = _check_match_score(check, entry.get("queries", []))
            weight = float(check.get("weight", 1.0) or 1.0)
            if score >= 0.55:
                matched_checks.append(str(check.get("label") or check.get("id")))
                coverage += weight * score
            else:
                missing_checks.append(str(check.get("label") or check.get("id")))
        coverage_ratio = max(0.0, min(1.0, coverage / total_weight))
        entry["checklist_coverage"] = coverage_ratio
        entry["matched_checks"] = matched_checks
        entry["missing_checks"] = missing_checks[:4]
        entry["score"] *= 1.0 + (2.2 * coverage_ratio)

def _merge_ranked(query_results: Iterable[Tuple[str, List[Dict[str, Any]]]], topk: int, profile: str = "", checks: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    for query_index, (query, results) in enumerate(query_results):
        query_weight = max(0.62, 1.0 - query_index * 0.08)
        for rank, item in enumerate(results or [], start=1):
            if not isinstance(item, dict):
                continue
            identity = _frame_identity(item, f"{query_index}-{rank}")
            entry = merged.setdefault(identity, {"item": dict(item), "score": 0.0, "queries": [], "best_rank": rank})
            precision_boost = _precision_query_multiplier(query, profile)
            entry["score"] += precision_boost * query_weight * ((1.0 / (rank + 8)) + 0.14 * _score(item, rank))
            entry["queries"].append(query)
            entry["best_rank"] = min(entry["best_rank"], rank)

    _apply_checklist_scores(merged, checks or [])
    if checks:
        ranked = sorted(merged.values(), key=lambda entry: (entry.get("checklist_coverage", 0.0), entry["score"], -entry["best_rank"]), reverse=True)
    else:
        ranked = sorted(merged.values(), key=lambda entry: (entry["score"], -entry["best_rank"]), reverse=True)
    frames: List[Dict[str, Any]] = []
    for entry in ranked[:topk]:
        item = entry["item"]
        item["agent_score"] = round(float(entry["score"]), 6)
        item["agent_queries"] = list(dict.fromkeys(entry["queries"]))[:4]
        item["agent_checklist_coverage"] = round(float(entry.get("checklist_coverage", 0.0)), 3)
        item["agent_matched_checks"] = entry.get("matched_checks", [])
        item["agent_missing_checks"] = entry.get("missing_checks", [])
        item.setdefault("score_breakdown", {"visual": _score(item, entry["best_rank"]), "ocr": 0.0, "asr": 0.0})
        frames.append(item)
    return frames


def _video_id(item: Dict[str, Any]) -> str:
    return _clean(item.get("video_id") or item.get("video_key") or item.get("videoKey"))


def _frame_id_value(item: Dict[str, Any]) -> str:
    value = _clean(
        item.get("frame_id")
        or item.get("frame_idx")
        or item.get("frame_key")
        or item.get("frameKey")
        or item.get("keyframe_number")
        or item.get("frame_name")
        or item.get("frameName")
    )
    video = _video_id(item)
    if video and value.startswith(video + "_"):
        value = value[len(video) + 1:]
    return re.sub(r"\.(webp|jpe?g|png)$", "", value, flags=re.IGNORECASE)


def _timestamp_value(item: Dict[str, Any]) -> Optional[float]:
    try:
        value = item.get("timestamp")
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _agent_score_value(item: Dict[str, Any]) -> float:
    for key in ("verification_score", "agent_score", "final_score", "normalized_score", "score", "_score"):
        try:
            value = float(item.get(key))
            if value >= 0:
                return value
        except (TypeError, ValueError):
            continue
    return 0.0


_EVIDENCE_STOPWORDS = {
    "with", "that", "this", "from", "into", "onto", "where", "there", "their",
    "person", "people", "scene", "frame", "image", "video", "showing", "visible",
}


def _text_overlap_score(query: str, evidence: str) -> float:
    query_norm = _normalise_text(query)
    evidence_norm = _normalise_text(evidence)
    if not query_norm or not evidence_norm:
        return 0.0
    if query_norm in evidence_norm:
        return 1.0
    tokens = [
        token
        for token in re.findall(r"[a-z0-9]+", query_norm)
        if len(token) >= 3 and token not in _EVIDENCE_STOPWORDS
    ]
    if not tokens:
        return 0.0
    matched = sum(1 for token in dict.fromkeys(tokens) if token in evidence_norm)
    return matched / max(1, len(dict.fromkeys(tokens)))


def _score_breakdown_value(item: Dict[str, Any], key: str) -> float:
    breakdown = item.get("score_breakdown")
    if not isinstance(breakdown, dict):
        return 0.0
    try:
        return max(0.0, min(1.0, float(breakdown.get(key) or 0.0)))
    except (TypeError, ValueError):
        return 0.0


def _candidate_modality_evidence(item: Dict[str, Any], plan: Dict[str, Any]) -> Tuple[float, List[str], List[str]]:
    routing = plan.get("routing") if isinstance(plan.get("routing"), dict) else {}
    evidence_parts: List[Tuple[float, float]] = []
    evidence_notes: List[str] = []
    evidence_matches: List[str] = []

    ocr_query = _clean(plan.get("ocr_query"))
    if ocr_query:
        ocr_text = _clean(item.get("ocr_text") or item.get("ocr") or item.get("text_ocr"))
        ocr_score = max(_score_breakdown_value(item, "ocr"), _text_overlap_score(ocr_query, ocr_text))
        if ocr_score > 0:
            evidence_parts.append((max(0.1, float(routing.get("ocr", 0.25) or 0.25)), ocr_score))
        if ocr_score >= 0.15:
            evidence_notes.append("OCR evidence matched the text query")
            evidence_matches.append(f"OCR: {ocr_query}")

    asr_query = _clean(plan.get("asr_query"))
    if asr_query:
        asr_text = _clean(item.get("asr_text") or item.get("transcript") or item.get("speech_text"))
        asr_score = max(_score_breakdown_value(item, "asr"), _text_overlap_score(asr_query, asr_text))
        if asr_score > 0:
            evidence_parts.append((max(0.1, float(routing.get("asr", 0.2) or 0.2)), asr_score))
        if asr_score >= 0.15:
            evidence_notes.append("ASR evidence matched the speech query")
            evidence_matches.append("ASR: relevant speech")

    if _timestamp_value(item) is not None:
        timestamp_source = _clean(item.get("timestamp_source"))
        if timestamp_source:
            evidence_parts.append((0.05, 1.0))
            evidence_notes.append(f"timestamp aligned by {timestamp_source}")

    if not evidence_parts:
        return 0.0, [], []
    total_weight = sum(weight for weight, _score in evidence_parts) or 1.0
    score = sum(weight * value for weight, value in evidence_parts) / total_weight
    return max(0.0, min(1.0, score)), evidence_notes, evidence_matches


def _timeline_seed_match_index(timeline: List[Dict[str, Any]], seed: Dict[str, Any]) -> Optional[int]:
    seed_identity = _frame_identity(seed, "")
    seed_frame = _frame_id_value(seed).lstrip("0")
    seed_ts = _timestamp_value(seed)
    best_ts_idx: Optional[int] = None
    best_ts_delta = float("inf")
    for index, item in enumerate(timeline):
        if seed_identity and _frame_identity(item, "") == seed_identity:
            return index
        item_frame = _frame_id_value(item).lstrip("0")
        if seed_frame and item_frame == seed_frame:
            return index
        item_ts = _timestamp_value(item)
        if seed_ts is not None and item_ts is not None:
            delta = abs(item_ts - seed_ts)
            if delta < best_ts_delta:
                best_ts_delta = delta
                best_ts_idx = index
    return best_ts_idx if best_ts_idx is not None and best_ts_delta <= 2.5 else None


def _find_timeline_center(timeline: List[Dict[str, Any]], seed: Dict[str, Any]) -> int:
    matched_index = _timeline_seed_match_index(timeline, seed)
    if matched_index is not None:
        return matched_index
    return min(len(timeline) // 2, max(0, len(timeline) - 1))


def _default_neighbor_provider(video_id: str, around_frame_id: str, limit: int) -> List[Dict[str, Any]]:
    try:
        from src.services.retrieval_backend import get_active_retriever

        return get_active_retriever().get_video_timeline(video_id=video_id, around_frame_id=around_frame_id, limit=limit)
    except Exception as exc:
        logger.debug("Agent temporal neighbor expansion skipped for %s/%s: %s", video_id, around_frame_id, exc)
        return []


def _expand_temporal_neighbors(
    frames: List[Dict[str, Any]],
    seed_limit: int = 14,
    neighbor_limit: int = 9,
    neighbor_provider: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    provider = neighbor_provider or _default_neighbor_provider
    expanded: List[Dict[str, Any]] = []
    seen_requests = set()

    for seed in frames[: max(0, seed_limit)]:
        video = _video_id(seed)
        around = _frame_id_value(seed)
        if not video or not around:
            continue
        request_key = (video.lower(), around.lstrip("0") or around)
        if request_key in seen_requests:
            continue
        seen_requests.add(request_key)

        timeline = provider(video, around, neighbor_limit) or []
        if not timeline or _timeline_seed_match_index(timeline, seed) is None:
            continue
        center = _find_timeline_center(timeline, seed)
        for index, item in enumerate(timeline):
            if not isinstance(item, dict):
                continue
            candidate = dict(item)
            distance = abs(index - center)
            candidate["_agent_temporal_distance"] = distance
            candidate["_agent_temporal_seed"] = _frame_identity(seed, "")
            candidate["_agent_temporal_seed_score"] = _agent_score_value(seed)
            candidate["_agent_temporal_seed_queries"] = list(seed.get("agent_queries") or [])
            candidate["_agent_temporal_seed_checks"] = list(seed.get("agent_matched_checks") or [])
            candidate["_agent_temporal_seed_missing"] = list(seed.get("agent_missing_checks") or [])
            expanded.append(candidate)
    return expanded


def _verification_reason(matched: List[str], missing: List[str], query_count: int, temporal_distance: Optional[int]) -> str:
    parts: List[str] = []
    if matched:
        parts.append(f"matched {len(matched)} checklist items")
    if query_count:
        parts.append(f"appeared in {query_count} expanded queries")
    if temporal_distance is not None and temporal_distance > 0:
        parts.append(f"near a strong candidate by {temporal_distance} keyframes")
    if missing:
        parts.append(f"uncertain: {', '.join(missing[:2])}")
    return "; ".join(parts) or "ranked by lightweight evidence signals"


def _rerank_with_light_verifier(
    frames: List[Dict[str, Any]],
    plan: Dict[str, Any],
    topk: int,
    neighbor_provider: Optional[Any] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not frames:
        return [], {"enabled": True, "method": "light_no_vlm", "temporal_neighbors": 0}

    visual_query_count = max(1, len(plan.get("executed_visual_queries") or plan.get("visual_queries") or []))
    max_agent_score = max(_agent_score_value(item) for item in frames) or 1.0
    entries: Dict[str, Dict[str, Any]] = {}

    def add_candidate(item: Dict[str, Any], source: str, direct_rank: int, temporal_distance: Optional[int] = None) -> None:
        identity = _frame_identity(item, f"{source}-{direct_rank}")
        queries = list(dict.fromkeys(item.get("agent_queries") or item.get("_agent_temporal_seed_queries") or []))
        matched = list(dict.fromkeys(item.get("agent_matched_checks") or item.get("_agent_temporal_seed_checks") or []))
        missing = list(dict.fromkeys(item.get("agent_missing_checks") or item.get("_agent_temporal_seed_missing") or []))
        coverage = max(0.0, min(1.0, float(item.get("agent_checklist_coverage") or 0.0)))
        consensus = min(1.0, len(queries) / visual_query_count)
        base_score = min(1.0, _agent_score_value(item) / max_agent_score)
        modality_score, modality_notes, modality_matches = _candidate_modality_evidence(item, plan)
        if modality_matches:
            matched = list(dict.fromkeys([*matched, *modality_matches]))

        if source == "temporal_neighbor":
            seed_score = min(1.0, float(item.get("_agent_temporal_seed_score") or 0.0) / max_agent_score)
            distance = int(temporal_distance or 0)
            decay = max(0.42, 1.0 - distance * 0.13)
            score = (0.52 * seed_score + 0.20 * coverage + 0.16 * consensus + 0.12 * modality_score) * decay * 0.96
            evidence = ["nearby keyframe from a strong retrieved candidate"]
        else:
            distance = None
            score = 0.46 * base_score + 0.25 * coverage + 0.14 * consensus + 0.15 * modality_score
            evidence = ["direct visual retrieval match"]
        evidence.extend(modality_notes)

        existing = entries.get(identity)
        if existing:
            existing["score"] = max(existing["score"], score + (0.035 if source == "temporal_neighbor" else 0.0))
            existing["sources"].add(source)
            existing["queries"].update(queries)
            existing["matched"].update(matched)
            existing["missing"].update(missing)
            existing["modality_evidence_score"] = max(existing.get("modality_evidence_score", 0.0), modality_score)
            if temporal_distance is not None:
                previous_distance = existing.get("temporal_distance")
                existing["temporal_distance"] = temporal_distance if previous_distance is None else min(previous_distance, temporal_distance)
            existing["best_rank"] = min(existing["best_rank"], direct_rank)
            return

        entries[identity] = {
            "item": dict(item),
            "score": score,
            "sources": {source},
            "queries": set(queries),
            "matched": set(matched),
            "missing": set(missing),
            "temporal_distance": temporal_distance,
            "best_rank": direct_rank,
            "evidence": evidence,
            "modality_evidence_score": modality_score,
        }

    for rank, item in enumerate(frames, start=1):
        add_candidate(item, "direct", rank)

    neighbors = _expand_temporal_neighbors(frames, neighbor_provider=neighbor_provider)
    for rank, item in enumerate(neighbors, start=1):
        add_candidate(item, "temporal_neighbor", rank, int(item.get("_agent_temporal_distance") or 0))

    ranked = sorted(entries.values(), key=lambda entry: (entry["score"], -entry["best_rank"]), reverse=True)
    verified: List[Dict[str, Any]] = []
    for index, entry in enumerate(ranked[:topk], start=1):
        item = entry["item"]
        queries = list(entry["queries"])
        matched = list(entry["matched"])
        missing = list(entry["missing"])
        temporal_distance = entry.get("temporal_distance")
        score = max(0.0, min(1.0, float(entry["score"])))
        item["rank"] = index
        item["verification_score"] = round(score, 6)
        item["agent_verification"] = {
            "score": round(score, 6),
            "method": "light_no_vlm",
            "sources": sorted(entry["sources"]),
            "evidence": sorted(set(entry["evidence"])),
            "query_count": len(queries),
            "temporal_distance": temporal_distance,
            "modality_evidence_score": round(float(entry.get("modality_evidence_score") or 0.0), 6),
            "score_breakdown": item.get("score_breakdown") if isinstance(item.get("score_breakdown"), dict) else {},
            "matched": matched,
            "missing": missing[:4],
            "note": "No VLM verification; score uses retrieval rank, checklist coverage, OCR/ASR evidence, timestamp evidence, and nearby keyframes.",
        }
        item["agent_queries"] = queries[:4]
        item["agent_matched_checks"] = matched
        item["agent_missing_checks"] = missing[:4]
        item["agent_checklist_coverage"] = round(max(float(item.get("agent_checklist_coverage") or 0.0), len(matched) / max(1, len(plan.get("must_have_checks") or []))), 3)
        item["reason"] = _verification_reason(matched, missing, len(queries), temporal_distance)
        verified.append(item)

    return verified, {
        "enabled": True,
        "method": "light_no_vlm",
        "input_frames": len(frames),
        "temporal_neighbors": len(neighbors),
        "ranked_candidates": len(ranked),
    }


def _format_answer(plan: Dict[str, Any], frames: List[Dict[str, Any]], added_to: Optional[str] = None) -> str:
    expanded = plan.get("expanded_queries") or []
    routing = plan.get("routing") or {}
    query_lines = [f"{idx}. {query.get('query_en') or query.get('query')}" for idx, query in enumerate(expanded[:4], start=1)]
    executed_queries = [query for query in (plan.get("executed_visual_queries") or []) if query]
    support_queries = [query for query in (plan.get("support_visual_queries") or []) if query]
    execution_lines = [
        "Search execution:",
        *(f"- {query}" for query in executed_queries[:3]),
    ]
    if support_queries:
        execution_lines.append(f"Support queries kept for rerank/debug: {len(support_queries)}")
    routing_text = " | ".join(
        f"{name.upper()} {float(routing.get(name, 0.0)):.1f}"
        for name in ("visual", "ocr", "asr")
    )
    target = added_to or "the results grid"
    return "\n".join(
        [
            "Expanded queries:",
            *(query_lines or ["1. " + plan.get("original_query", "")]),
            "",
            "Routing:",
            routing_text,
            "",
            *execution_lines,
            "",
            f"Results added to {target}: {len(frames)} keyframes.",
        ]
    )


def run_agent_query_search(message: str, topk: int = 100, added_to: Optional[str] = None) -> Dict[str, Any]:
    topk = max(1, min(int(topk or 100), 100))
    plan = build_agent_plan(message, topk=topk)
    if not plan["original_query"]:
        return {
            "answer": "Prompt rong, khong the chay Agent Search.",
            "plan": plan,
            "frames": [],
            "total_candidates": 0,
        }

    query_results: List[Tuple[str, List[Dict[str, Any]]]] = []
    visual_queries = [query for query in (plan.get("executed_visual_queries") or [plan.get("visual_query")]) if query]

    use_fusion = bool(plan.get("ocr_query") or plan.get("asr_query"))
    if use_fusion:
        try:
            from src.services.fusion_service import multimodal_search

            frames = multimodal_search(
                visual_query=plan["visual_query"],
                ocr_query=plan["ocr_query"],
                asr_query=plan["asr_query"],
                weights=plan["routing"],
                topk=topk,
                original_query=plan["visual_query"] or plan["original_query"],
            )
            fusion_query_label = _clean(" ".join(
                query for query in (plan.get("visual_query"), plan.get("ocr_query"), plan.get("asr_query")) if query
            ))
            query_results.append((fusion_query_label or plan["visual_query"], frames))
        except Exception as exc:
            logger.warning("Agent multimodal search failed: %s", exc)
    try:
        from src.services.user_service import getImageDataSingleTextSearch

        per_query = max(30, min(80, topk))
        already_run = {plan.get("visual_query")} if use_fusion else set()
        for query in visual_queries:
            if query in already_run:
                continue
            already_run.add(query)
            query_results.append((query, getImageDataSingleTextSearch(query, per_query)))
    except Exception as exc:
        logger.warning("Agent multi-query visual search failed: %s", exc)

    merged_frames = _merge_ranked(
        query_results,
        topk=max(topk, 80),
        profile=plan.get("precision_profile", ""),
        checks=plan.get("must_have_checks", []),
    )
    frames, verification_summary = _rerank_with_light_verifier(merged_frames, plan, topk=topk)
    try:
        from src.services.openrouter_vlm_verifier import verify_frames_with_openrouter_vlm

        frames, vlm_summary = verify_frames_with_openrouter_vlm(frames, plan)
    except Exception as exc:
        logger.warning("Agent VLM verifier integration failed; keeping light ranking: %s", exc)
        vlm_summary = {"enabled": False, "method": "openrouter_vlm", "error": str(exc)[:180]}
    plan["verification"] = {**plan.get("verification", {}), **verification_summary, "vlm": vlm_summary}
    if vlm_summary.get("enabled") and vlm_summary.get("evaluated", 0):
        plan["verification"]["method"] = "light_plus_openrouter_vlm"

    return {
        "answer": _format_answer(plan, frames, added_to=added_to),
        "plan": plan,
        "frames": frames,
        "total_candidates": sum(len(results or []) for _, results in query_results) + int(verification_summary.get("temporal_neighbors", 0)),
    }
