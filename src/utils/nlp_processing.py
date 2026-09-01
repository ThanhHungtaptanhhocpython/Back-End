"""NLP Processing Utility.

Provides query parsing and dynamic weight calculation for Multimodal Fusion.
"""

import logging
import re
from pathlib import Path
from typing import Dict, Any

from dotenv import load_dotenv

try:
    from deep_translator import GoogleTranslator
except Exception:
    GoogleTranslator = None

load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env", override=True)

logger = logging.getLogger(__name__)

class Translation():
    _cache: Dict[tuple[str, str, str], str] = {}
    _SUPPORTED_LANGUAGES = {"en", "vi"}

    def __init__(self, from_lang='vi', to_lang='en'):
        self.__to_lang = str(to_lang or "").strip().lower()
        self.__from_lang = str(from_lang or "").strip().lower()
        if self.__from_lang not in self._SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported source language: {self.__from_lang}")
        if self.__to_lang not in self._SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported target language: {self.__to_lang}")
        self.translator = None
        self.last_provider = "none"
        self.last_translated = False
        self._init_translator()

    def _init_translator(self):
        try:
            if GoogleTranslator:
                self.translator = GoogleTranslator(source=self.__from_lang, target=self.__to_lang)
            else:
                self.translator = None
        except Exception as e:
            logger.warning(f"Failed to initialize GoogleTranslator: {e}")
            self.translator = None

    def preprocessing(self, text):
        return text.strip() if text else ""

    @staticmethod
    def _clean_provider_output(value: Any) -> str:
        """Remove common LLM wrappers without changing translation content."""
        text = str(value or "").strip()
        if text.startswith("```") and text.endswith("```"):
            text = re.sub(r"^```(?:text|markdown)?\s*", "", text, flags=re.IGNORECASE)
            text = re.sub(r"\s*```$", "", text)
        text = re.sub(
            r"^(?:translation|translated\s+text|english|vietnamese|tiếng\s+anh|tiếng\s+việt)\s*:\s*",
            "",
            text,
            count=1,
            flags=re.IGNORECASE,
        ).strip()
        if len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'"}:
            text = text[1:-1].strip()
        return text

    @staticmethod
    def _same_text(left: str, right: str) -> bool:
        normalize = lambda value: re.sub(r"\s+", " ", value or "").strip().casefold()
        return normalize(left) == normalize(right)

    def _usable_translation(self, source: str, candidate: str) -> bool:
        if not candidate or self._same_text(source, candidate):
            return False
        rejected_prefixes = (
            "here is the translation",
            "the translation is",
            "bản dịch là",
            "tôi không thể dịch",
            "i cannot translate",
        )
        return not candidate.casefold().startswith(rejected_prefixes)

    def _prefer_contextual_translation(self, text: str) -> bool:
        """Google often misreads Vietnamese without diacritics as unrelated words."""
        if self.__from_lang != "vi":
            return False
        has_letters = bool(re.search(r"[A-Za-z]", text))
        has_vietnamese_diacritics = bool(re.search(r"[à-ỹÀ-ỸđĐ]", text))
        return has_letters and not has_vietnamese_diacritics

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
            return self._clean_provider_output(getattr(response, "content", ""))
        except Exception as e:
            logger.warning(f"OpenRouter translation fallback failed: {e}")
            return ""

    def __call__(self, text):
        cleaned = self.preprocessing(text)
        self.last_provider = "none"
        self.last_translated = False
        if not cleaned:
            return ""

        if self.__from_lang == self.__to_lang:
            self.last_provider = "identity"
            return cleaned

        cache_key = (self.__from_lang, self.__to_lang, cleaned)
        if cache_key in self._cache:
            self.last_provider = "cache"
            cached = self._cache[cache_key]
            self.last_translated = cached != cleaned
            return cached

        result = ""
        tried_openrouter = False
        if self._prefer_contextual_translation(cleaned):
            tried_openrouter = True
            fallback = self._translate_with_openrouter(cleaned)
            if self._usable_translation(cleaned, fallback):
                self.last_provider = "openrouter"
                self.last_translated = True
                self._cache[cache_key] = fallback
                return fallback

        if self.translator is None:
            self._init_translator()
        if self.translator is not None:
            try:
                result = self._clean_provider_output(self.translator.translate(cleaned))
                if self._usable_translation(cleaned, result):
                    self.last_provider = "google"
                    self.last_translated = True
                    self._cache[cache_key] = result
                    return result
            except Exception as e:
                logger.warning(f"Translation call failed: {e}")

        if not tried_openrouter:
            fallback = self._translate_with_openrouter(cleaned)
            if self._usable_translation(cleaned, fallback):
                self.last_provider = "openrouter"
                self.last_translated = True
                self._cache[cache_key] = fallback
                return fallback

        self.last_provider = "identity"
        self._cache[cache_key] = cleaned
        return cleaned

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
