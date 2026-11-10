"""
Shared Constants — Centralized definitions used across the Legal AI codebase.

Single source of truth for greeting sets, legal categories, and other
shared keyword dictionaries to avoid duplication.
"""

# ── Greeting Sets ─────────────────────────────────────────────────────────────
# Phrases are matched as *whole phrases only* (never as substrings), and the
# token set below is used for short messages composed purely of greeting words.

AR_GREETINGS: set[str] = {
    "مرحبا", "أهلا", "السلام عليكم", "السلام عليكم ورحمة الله",
    "اهلا", "هاي", "هلا", "صباح الخير", "مساء الخير", "مساء الورد",
}

EN_GREETINGS: set[str] = {
    "hello", "hi", "hey", "hi there", "hello there",
    "good morning", "good evening", "good afternoon", "greetings",
}

_GREETING_PHRASES: dict[str, set[str]] = {"ar": AR_GREETINGS, "en": EN_GREETINGS}


def greeting_phrases(language: str) -> set[str]:
    """Return full greeting phrases for a language ("ar" or "en")."""
    return _GREETING_PHRASES.get(language, EN_GREETINGS)


def greeting_tokens(language: str) -> set[str]:
    """Return the individual words that make up greeting phrases."""
    tokens: set[str] = set()
    for phrase in greeting_phrases(language):
        tokens.update(phrase.lower().split())
    return tokens

# ── Legal Category Keywords ──────────────────────────────────────────────────

LEGAL_CATEGORIES: dict[str, list[str]] = {
    "contracts":   ["عقد", "عقود", "تعاقد", "إيجاب", "قبول", "اتفاقية"],
    "obligations": ["التزام", "التزامات", "مسؤولية", "تعويض", "دين"],
    "property":    ["ملكية", "حيازة", "عقار", "ارتفاق", "أرض", "مبنى"],
    "inheritance": ["ميراث", "إرث", "وصية", "تركة", "وارث"],
    "evidence":    ["إثبات", "بينة", "شهادة", "دليل", "برهان"],
    "persons":     ["أهلية", "شخصية", "ولاية", "وصاية", "قاصر"],
    "family":      ["زواج", "طلاق", "نفقة", "حضانة", "نسب"],
    "procedural":  ["دعوى", "محكمة", "قضاء", "تنفيذ", "حجز"],
}

# ── Router Keyword Sets ──────────────────────────────────────────────────────

LEGAL_KEYWORDS_AR: list[str] = [
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

LEGAL_KEYWORDS_EN: list[str] = [
    "law", "legal", "article", "civil", "code", "contract", "obligation",
    "property", "inheritance", "evidence", "court", "judgment", "liability",
    "compensation", "rights", "duty", "penalty", "crime", "void", "breach",
    "guarantee", "mortgage", "debt", "creditor", "debtor", "statute",
    "regulation", "legislation", "egyptian", "ownership", "possession",
    "marriage", "divorce", "custody", "testament", "will", "lease",
]

NON_LEGAL_KEYWORDS_AR: list[str] = [
    "طبخ", "وصفة", "طعام", "أكل", "رياضة", "كرة", "لعبة", "فيلم",
    "أغنية", "موسيقى", "طقس", "جو", "برمجة", "كود", "قصيدة", "شعر",
    "سفر", "رحلة", "فندق",
]

NON_LEGAL_KEYWORDS_EN: list[str] = [
    "cook", "recipe", "food", "sport", "game", "movie", "song", "music",
    "weather", "code", "program", "poem", "poetry", "travel", "hotel",
]

# ── Article Reference Patterns ───────────────────────────────────────────────
# Shared regex building blocks for detecting/extracting article references
# ("مادة 131", "مادة (5)", "Article 20", "Art. 20").

ARTICLE_MARKERS = r"مادة|مــادة|المادة"
ARTICLE_NUM = r"(?:\d+|[٠-٩]+)"
ARTICLE_PATTERNS: list[str] = [
    rf"(?:{ARTICLE_MARKERS})\s*[\(（]?\s*(\d+)\s*[\)）]?",
    r"[Aa]rt(?:icle)?\.?\s*(\d+)",
]
