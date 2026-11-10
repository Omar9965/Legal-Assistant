"""
Router — Language detection and legal relevance classification.

Uses fast heuristic classification instead of an LLM call to eliminate
the 2-5 second latency that the LLM-based router added per query.
Greeting detection matches whole phrases / short token-only messages —
never substrings (so "hi" inside "this" no longer misclassifies queries).
"""

import re
from utils.constants import (
    LEGAL_KEYWORDS_AR,
    LEGAL_KEYWORDS_EN,
    NON_LEGAL_KEYWORDS_AR,
    NON_LEGAL_KEYWORDS_EN,
    ARTICLE_PATTERNS,
    greeting_phrases,
    greeting_tokens,
)


class RouterAgent:
    """
    Fast heuristic-based router — classifies language and legal relevance
    in <1ms using keyword matching instead of an LLM call.
    """

    def execute(self, query: str) -> dict:
        """
        Classify a user query for language and legal relevance.

        Returns:
            Dict with keys: language, is_legal, is_greeting, reason, confidence
        """
        language = self._detect_language(query)
        is_greeting = self._is_greeting(query, language)

        if is_greeting:
            return {
                "language": language,
                "is_legal": False,
                "is_greeting": True,
                "reason": "Greeting detected",
                "confidence": 0.95,
            }

        is_legal, confidence, reason = self._classify_legal(query, language)

        return {
            "language": language,
            "is_legal": is_legal,
            "is_greeting": False,
            "reason": reason,
            "confidence": confidence,
        }

    # ── Internal helpers ─────────────────────────────────────────────────

    @staticmethod
    def _detect_language(query: str) -> str:
        """Detect language based on Unicode character ranges."""
        arabic_chars = sum(1 for c in query if "\u0600" <= c <= "\u06FF")
        latin_chars = sum(1 for c in query if c.isascii() and c.isalpha())
        return "ar" if arabic_chars >= latin_chars else "en"

    @staticmethod
    def _is_greeting(query: str, language: str) -> bool:
        """Greeting only if the message IS a greeting: a whole-phrase match,
        or a short message made entirely of greeting words. This avoids
        substring false positives like "hi" inside "this"."""
        letters_only = re.sub(r"[^\u0621-\u064Aa-zA-Z\s]", " ", query)
        normalized = re.sub(r"\s+", " ", letters_only).strip().lower()
        if not normalized:
            return False

        if normalized in {p.lower() for p in greeting_phrases(language)}:
            return True

        words = normalized.split()
        vocab = greeting_tokens(language)
        return len(words) <= 3 and all(w in vocab for w in words)

    @staticmethod
    def _classify_legal(query: str, language: str) -> tuple:
        """Return (is_legal, confidence, reason)."""
        q_lower = query.lower()

        if language == "ar":
            legal_hits = sum(1 for kw in LEGAL_KEYWORDS_AR if kw in query)
            non_legal_hits = sum(1 for kw in NON_LEGAL_KEYWORDS_AR if kw in query)
        else:
            legal_hits = sum(
                1 for kw in LEGAL_KEYWORDS_EN if kw in q_lower.split()
            )
            non_legal_hits = sum(
                1 for kw in NON_LEGAL_KEYWORDS_EN if kw in q_lower.split()
            )

        # Article number reference is a strong legal signal
        has_article_ref = any(re.search(pat, query) for pat in ARTICLE_PATTERNS)
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
