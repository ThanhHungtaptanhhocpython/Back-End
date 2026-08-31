"""NLP Processing Utility.

Provides query parsing and dynamic weight calculation for Multimodal Fusion.
"""

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv

try:
    from deep_translator import GoogleTranslator
except Exception:
    GoogleTranslator = None

load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env", override=True)

logger = logging.getLogger(__name__)

# Structured outcome codes shared with the API layer and the frontend.
TRANSLATION_OK = "ok"
TRANSLATION_INVALID_INPUT = "invalid_input"
TRANSLATION_PROVIDER_UNAVAILABLE = "provider_unavailable"


@dataclass
class TranslationResult:
    """Outcome of a single translation attempt.

    ``text`` is always safe to show the user: the real translation on success,
    the trimmed original text on a provider failure, and ``""`` only when the
    input was blank. ``translated`` is True only when a provider actually
    produced a different string. ``status`` is one of the ``TRANSLATION_*``
    codes above; ``error_code`` mirrors it on failure and is ``None`` on
    success.
    """

    text: str
    translated: bool
    provider: str  # "google" | "openrouter" | "cache" | "identity" | "none"
    status: str
    error_code: Optional[str] = None
    detail: Optional[str] = None


def _normalize_lang(value: Optional[str]) -> str:
    return (value or "").strip().lower()


class Translation():
    # Only ever holds successful, provider-produced translations that differ
    # from their input. Identity results, failures and exceptions are never
    # stored, so a transient provider outage cannot "freeze" a query to its
    # original text until the backend restarts.
    _cache: Dict[tuple[str, str, str], str] = {}

    def __init__(self, from_lang='vi', to_lang='en'):
        self.__from_lang = _normalize_lang(from_lang) or "auto"
        self.__to_lang = _normalize_lang(to_lang) or "en"
        self.translator = None
        self.last_provider = "none"
        self.last_translated = False
        self.last_status = TRANSLATION_OK
        self.last_error_code: Optional[str] = None
        self.last_detail: Optional[str] = None
        self.last_result: Optional[TranslationResult] = None
        self._init_translator()

    def _init_translator(self):
        try:
            if GoogleTranslator:
                self.translator = GoogleTranslator(source=self.__from_lang, target=self.__to_lang)
            else:
                self.translator = None
        except Exception as exc:
            logger.warning(
                "Failed to initialize translation provider (provider=google, error=%s)",
                type(exc).__name__,
            )
            self.translator = None

    def preprocessing(self, text):
        return text.strip() if text else ""

    @staticmethod
    def _is_real_translation(candidate: str, source: str) -> bool:
        """True when ``candidate`` is a non-empty string that differs from the input."""
        cleaned = (candidate or "").strip()
        return bool(cleaned) and cleaned.casefold() != source.strip().casefold()

    def _translate_with_google(self, text: str) -> str:
        if self.translator is None:
            self._init_translator()
        if self.translator is None:
            return ""
        try:
            return (self.translator.translate(text) or "").strip()
        except Exception as exc:
            # Never log the user's text or provider payloads -- provider + type only.
            logger.warning(
                "Translation provider call failed (provider=google, error=%s)",
                type(exc).__name__,
            )
            return ""

    _TRANSLATE_SYSTEM_PROMPT = (
        "You are a translation engine for video retrieval. Return only a complete "
        "translation, with no notes, quotes, markdown, or explanations. Preserve "
        "event order, subjects, actions, time references, and spatial relations. "
        "Translate Vietnamese 'dung duoi nuoc' as 'standing in the water' unless the "
        "source explicitly says the person is submerged or diving."
    )

    def _translate_with_gateway(self, text: str) -> tuple[str, str]:
        """Multi-provider Text chain. Returns (translation, provider_id).

        ``("", "")`` when the gateway is disabled, has no usable Text provider,
        or every provider failed.
        """
        from src.config.settings import get_settings

        settings = get_settings()
        if not settings.ai_gateway_enabled:
            return "", ""
        try:
            from src.services.ai import gateway as ai_gateway
        except Exception:
            return "", ""
        if not ai_gateway.text_available(settings):
            return "", ""
        messages = [
            {"role": "system", "content": self._TRANSLATE_SYSTEM_PROMPT},
            {"role": "user", "content": f"Translate from {self.__from_lang} to {self.__to_lang}:\n{text}"},
        ]
        try:
            result, _attempts = ai_gateway.text_completion(
                messages,
                settings=settings,
                max_tokens=settings.openrouter_translate_max_tokens,
            )
        except Exception as exc:
            logger.warning(
                "Translation gateway exhausted (provider chain, error=%s)",
                type(exc).__name__,
            )
            return "", ""
        return (result.text or "").strip(), result.provider

    def _translate_with_openrouter(self, text: str) -> str:
        from src.config.settings import get_settings

        settings = get_settings()
        api_key = settings.openrouter_api_key
        if not api_key:
            return ""
        try:
            from langchain_openai import ChatOpenAI

            model = settings.openrouter_translate_model or settings.openrouter_model
            llm = ChatOpenAI(
                model=model,
                temperature=0,
                api_key=api_key,
                max_tokens=settings.openrouter_translate_max_tokens,
                base_url=settings.openrouter_base_url,
                default_headers={
                    "HTTP-Referer": settings.openrouter_site_url,
                    "X-Title": settings.openrouter_app_name,
                },
            )
            response = llm.invoke([
                (
                    "system",
                    "You are a translation engine for video retrieval. Return only a complete translation, with no notes, quotes, markdown, or explanations. Preserve event order, subjects, actions, time references, and spatial relations. Translate Vietnamese 'dung duoi nuoc' as 'standing in the water' unless the source explicitly says the person is submerged or diving.",
                ),
                (
                    "human",
                    f"Translate from {self.__from_lang} to {self.__to_lang}:\n{text}",
                ),
            ])
            return str(getattr(response, "content", "") or "").strip()
        except Exception as exc:
            logger.warning(
                "Translation fallback failed (provider=openrouter, error=%s)",
                type(exc).__name__,
            )
            return ""

    def translate_detailed(self, text) -> TranslationResult:
        """Translate ``text`` and return a structured, honest outcome.

        ``from_lang`` / ``to_lang`` supplied by the caller are the source of
        truth. Translation is only skipped when the two languages are actually
        the same or the input is blank -- never because the text happens to be
        Latin script or unaccented Vietnamese.
        """
        cleaned = self.preprocessing(text)
        if not cleaned:
            return TranslationResult(
                text="",
                translated=False,
                provider="none",
                status=TRANSLATION_INVALID_INPUT,
                error_code=TRANSLATION_INVALID_INPUT,
                detail="Input text is empty after trimming.",
            )

        # The only legitimate identity: source and target really are the same.
        if self.__from_lang == self.__to_lang:
            return TranslationResult(
                text=cleaned,
                translated=False,
                provider="identity",
                status=TRANSLATION_OK,
            )

        cache_key = (self.__from_lang, self.__to_lang, cleaned)
        cached = self._cache.get(cache_key)
        if cached is not None:
            # The cache only ever holds successful, different-from-input results.
            return TranslationResult(
                text=cached,
                translated=True,
                provider="cache",
                status=TRANSLATION_OK,
            )

        # 1) Primary provider.
        google_text = self._translate_with_google(cleaned)
        if self._is_real_translation(google_text, cleaned):
            self._cache[cache_key] = google_text
            return TranslationResult(google_text, True, "google", TRANSLATION_OK)

        # 2a) Multi-provider Text chain (translation + Agent planner), when the
        #     AI gateway is enabled. Reports the provider that actually answered.
        gateway_text, gateway_provider = "", ""
        try:
            gateway_text, gateway_provider = self._translate_with_gateway(cleaned)
        except Exception as exc:
            logger.warning(
                "Translation gateway error (error=%s)", type(exc).__name__
            )
        if self._is_real_translation(gateway_text, cleaned):
            self._cache[cache_key] = gateway_text
            return TranslationResult(
                gateway_text, True, gateway_provider or "gateway", TRANSLATION_OK
            )

        # 2b) Single-provider OpenRouter fallback -- only if configured.
        try:
            fallback = (self._translate_with_openrouter(cleaned) or "").strip()
        except Exception as exc:
            logger.warning(
                "Translation fallback error (provider=openrouter, error=%s)",
                type(exc).__name__,
            )
            fallback = ""
        if self._is_real_translation(fallback, cleaned):
            self._cache[cache_key] = fallback
            return TranslationResult(fallback, True, "openrouter", TRANSLATION_OK)

        # 3) Nothing produced a usable translation. Report a structured failure:
        # never cache it, never claim success, keep the original text for the UI.
        logger.warning(
            "Translation unavailable (providers tried: google=%s, gateway=%s, openrouter=%s)",
            "returned" if google_text else "empty/failed",
            "returned" if gateway_text else "empty/failed/not-configured",
            "returned" if fallback else "empty/failed/not-configured",
        )
        return TranslationResult(
            text=cleaned,
            translated=False,
            provider="none",
            status=TRANSLATION_PROVIDER_UNAVAILABLE,
            error_code=TRANSLATION_PROVIDER_UNAVAILABLE,
            detail="No translation provider returned a usable translation.",
        )

    def __call__(self, text):
        """Backward-compatible entry point: returns a plain string.

        The string is the live translation on success, or the trimmed original
        text on failure (``""`` for blank input). Structured details are on
        ``last_result`` / ``last_status`` / ``last_error_code``.
        """
        result = self.translate_detailed(text)
        self.last_result = result
        self.last_provider = result.provider
        self.last_translated = result.translated
        self.last_status = result.status
        self.last_error_code = result.error_code
        self.last_detail = result.detail
        return result.text

class QueryPlanner:
    """Parses natural language queries to route them to appropriate modalities."""

    ASR_KEYWORDS = [
        "nghe ti\u1ebfng",
        "n\u00f3i r\u1eb1ng",
        "\u00e2m thanh",
        "c\u00f3 ti\u1ebfng",
        "n\u00f3i",
        "nghe",
    ]
    @classmethod
    def parse_query(cls, query: str) -> Dict[str, Any]:
        """Parse a query and generate dynamic fusion weights.
        
        Rules:
        - Text in double quotes ("...") is extracted as an OCR query.
        - The presence of ASR_KEYWORDS triggers an ASR query (using the rest of the text).
        - Weights are dynamically adjusted based on which modalities are active.
        """
        visual_query = query
        ocr_query = ""
        asr_query = ""
        
        weights = {
            "visual": 1.0,
            "ocr": 0.0,
            "asr": 0.0
        }

        # 1. Extract OCR Text (Rule: Text enclosed in double quotes)
        quotes_pattern = r'"([^"]*)"'
        matches = re.findall(quotes_pattern, query)
        
        if matches:
            # Combine all quoted phrases into a single OCR query
            ocr_query = " ".join(matches).strip()
            # Remove quoted text from visual query so Faiss doesn't try to "look" for text
            visual_query = re.sub(quotes_pattern, '', visual_query).strip()
            
            # Boost OCR weight since user explicitly requested text
            weights["ocr"] = 0.5
            weights["visual"] = 0.5

        # 2. Extract ASR Intent (Rule: Presence of auditory keywords)
        lower_query = query.lower()
        has_audio_intent = any(kw in lower_query for kw in cls.ASR_KEYWORDS)
        
        if has_audio_intent:
            asr_query = visual_query  # Whatever remains after OCR extraction
            weights["asr"] = 0.3
            
            # Rebalance weights if OCR is also active
            if weights["ocr"] > 0:
                weights["visual"] = 0.4
                weights["ocr"] = 0.3
            else:
                weights["visual"] = 0.7

        # Edge case: visual query became empty (e.g. user just searched `"Sale 50%"`)
        if not visual_query:
            weights["visual"] = 0.0
            if weights["ocr"] > 0 and weights["asr"] == 0:
                weights["ocr"] = 1.0
            elif weights["asr"] > 0 and weights["ocr"] == 0:
                weights["asr"] = 1.0
            elif weights["ocr"] > 0 and weights["asr"] > 0:
                weights["ocr"] = 0.5
                weights["asr"] = 0.5

        return {
            "original_query": query,
            "visual_query": visual_query,
            "ocr_query": ocr_query,
            "asr_query": asr_query,
            "weights": weights
        }

    @classmethod
    def generate_vqa_question(cls, query: str) -> str:
        """Convert a Vietnamese query into an English VQA Yes/No question."""
        # Remove quotes to avoid confusing the translation or VQA model
        clean_query = query.replace('"', '').strip()
        
        # Translate to English
        translator = Translation()
        eng_query = translator(clean_query)
        
        # If translation failed or returned empty, fallback
        if not eng_query:
            eng_query = clean_query
            
        # Format as Yes/No question
        # Assuming the translated query is a noun phrase like "man riding a bicycle"
        question = f"Is there a {eng_query}?"
        return question
