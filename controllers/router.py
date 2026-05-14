"""
Router — Language detection and legal relevance classification.

Uses fast heuristic classification instead of an LLM call to eliminate
the 2-5 second latency that the LLM-based router added per query.
"""

import re
from controllers.base_agent import BaseAgent


# ── Keyword sets for fast classification ─────────────────────────────────────

_AR_GREETINGS = {
    "مرحبا", "أهلا", "السلام عليكم", "السلام عليكم ورحمة الله",
    "اهلا", "هاي", "هلا", "صباح الخير", "مساء الخير", "مساء الورد",
}
_EN_GREETINGS = {
    "hello", "hi", "hey", "hi there", "hello there",
    "good morning", "good evening", "good afternoon", "greetings",
}

_LEGAL_KEYWORDS_AR = [
    "قانون", "مادة", "المادة", "مدني", "عقد", "عقود", "التزام", "التزامات",
    "ملكية", "حيازة", "ميراث", "إرث", "وصية", "تركة", "تعويض", "مسؤولية",
    "إثبات", "بينة", "شهادة", "دعوى", "محكمة", "قضاء", "حكم", "أحكام",
    "جريمة", "عقوبة", "حق", "حقوق", "واجب", "شرط", "بطلان", "فسخ",
    "إيجاب", "قبول", "تعاقد", "ضمان", "كفالة", "رهن", "دين", "دائن",
    "مدين", "أجل", "تقادم", "نفاذ", "وكالة", "وكيل", "شفعة", "ارتفاق",
    "أهلية", "شخصية", "ولاية", "وصاية", "قاصر", "زواج", "طلاق", "نفقة",
    "حضانة", "نسب", "عقار", "إيجار", "مؤجر", "مستأجر", "بيع", "هبة",
    "تجاري", "شركة", "إفلاس", "مشرع", "تشريع", "لائحة", "نظام",
]

_LEGAL_KEYWORDS_EN = [
    "law", "legal", "article", "civil", "code", "contract", "obligation",
    "property", "inheritance", "evidence", "court", "judgment", "liability",
    "compensation", "rights", "duty", "penalty", "crime", "void", "breach",
    "guarantee", "mortgage", "debt", "creditor", "debtor", "statute",
    "regulation", "legislation", "egyptian", "ownership", "possession",
    "marriage", "divorce", "custody", "testament", "will", "lease",
]

_NON_LEGAL_KEYWORDS_AR = [
    "طبخ", "وصفة", "طعام", "أكل", "رياضة", "كرة", "لعبة", "فيلم",
    "أغنية", "موسيقى", "طقس", "جو", "برمجة", "كود", "قصيدة", "شعر",
    "سفر", "رحلة", "فندق",
]
_NON_LEGAL_KEYWORDS_EN = [
    "cook", "recipe", "food", "sport", "game", "movie", "song", "music",
    "weather", "code", "program", "poem", "poetry", "travel", "hotel",
]


class RouterAgent(BaseAgent):
    """
    Fast heuristic-based router — classifies language and legal relevance
    in <1ms using keyword matching instead of an LLM call.
    """

    def execute(self, query: str) -> dict:
        """
        Classify a user query for language and legal relevance.

        Returns:
            Dict with keys: language, is_legal, reason, confidence
        """
        language = self._detect_language(query)
        is_greeting = self._is_greeting(query, language)

        if is_greeting:
            return {
                "language": language,
                "is_legal": False,
                "reason": "Greeting detected",
                "confidence": 0.95,
            }

        is_legal, confidence, reason = self._classify_legal(query, language)

        return {
            "language": language,
            "is_legal": is_legal,
            "reason": reason,
            "confidence": confidence,
        }

    # ── Internal helpers ─────────────────────────────────────────────────

    @staticmethod
    def _detect_language(query: str) -> str:
        """Detect language based on Unicode character ranges."""
        arabic_chars = sum(1 for c in query if "\u0600" <= c <= "\u06FF")
        latin_chars = sum(1 for c in query if "A" <= c <= "z")
        return "ar" if arabic_chars >= latin_chars else "en"

    @staticmethod
    def _is_greeting(query: str, language: str) -> bool:
        q = query.strip()
        if language == "ar":
            return any(g in q for g in _AR_GREETINGS)
        return q.lower() in _EN_GREETINGS or any(
            g in q.lower() for g in _EN_GREETINGS
        )

    @staticmethod
    def _classify_legal(query: str, language: str) -> tuple:
        """Return (is_legal, confidence, reason)."""
        q_lower = query.lower()

        if language == "ar":
            legal_hits = sum(1 for kw in _LEGAL_KEYWORDS_AR if kw in query)
            non_legal_hits = sum(1 for kw in _NON_LEGAL_KEYWORDS_AR if kw in query)
        else:
            legal_hits = sum(1 for kw in _LEGAL_KEYWORDS_EN if kw in q_lower)
            non_legal_hits = sum(1 for kw in _NON_LEGAL_KEYWORDS_EN if kw in q_lower)

        # Article number reference is a strong legal signal
        has_article_ref = bool(
            re.search(r"(?:مادة|مــادة|المادة)\s*[\(（]?\s*\d+", query)
            or re.search(r"[Aa]rt(?:icle)?\.?\s*\d+", query)
        )
        if has_article_ref:
            legal_hits += 3

        if legal_hits > 0 and non_legal_hits == 0:
            confidence = min(0.95, 0.7 + legal_hits * 0.05)
            return True, confidence, f"Legal keywords matched ({legal_hits})"
        elif non_legal_hits > 0 and legal_hits == 0:
            confidence = min(0.95, 0.7 + non_legal_hits * 0.05)
            return False, confidence, f"Non-legal keywords matched ({non_legal_hits})"
        elif legal_hits > non_legal_hits:
            return True, 0.65, "Mixed signals, leaning legal"
        elif non_legal_hits > legal_hits:
            return False, 0.65, "Mixed signals, leaning non-legal"
        else:
            # No keywords matched — default to legal (benefit of the doubt)
            return True, 0.5, "No strong signals, defaulting to legal"
