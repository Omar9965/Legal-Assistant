"""
Shared Constants — Centralized definitions used across the Legal AI codebase.

Single source of truth for greeting sets, legal categories, and other
shared keyword dictionaries to avoid duplication.
"""

# ── Greeting Sets ─────────────────────────────────────────────────────────────

AR_GREETINGS: set[str] = {
    "مرحبا", "أهلا", "السلام عليكم", "السلام عليكم ورحمة الله",
    "اهلا", "هاي", "هلا", "صباح الخير", "مساء الخير", "مساء الورد",
}

EN_GREETINGS: set[str] = {
    "hello", "hi", "hey", "hi there", "hello there",
    "good morning", "good evening", "good afternoon", "greetings",
}

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
