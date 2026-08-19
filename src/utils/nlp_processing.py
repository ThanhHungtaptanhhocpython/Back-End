"""NLP Processing Utility.

Provides query parsing and dynamic weight calculation for Multimodal Fusion.
"""

import re
from typing import Dict, Any

try:
    from deep_translator import GoogleTranslator
except Exception:
    GoogleTranslator = None

class Translation():
    def __init__(self, from_lang='vi', to_lang='en'):
        # The class Translation is a wrapper for deep-translator
        self.__to_lang = to_lang
        self.__from_lang = from_lang
        try:
            if GoogleTranslator:
                self.translator = GoogleTranslator(source=self.__from_lang, target=self.__to_lang)
            else:
                self.translator = None
        except Exception:
            self.translator = None

    def preprocessing(self, text):
        return text.lower()

    def __call__(self, text):
        text = self.preprocessing(text)
        if self.translator is None:
            return text
        try:
            return self.translator.translate(text)
        except Exception:
            return text

class QueryPlanner:
    """Parses natural language queries to route them to appropriate modalities."""

    ASR_KEYWORDS = ["nghe tiếng", "nói rằng", "âm thanh", "có tiếng", "nói", "nghe"]

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
