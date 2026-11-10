"""Tests for query expansion fallback logic."""

from models.query_expansion import fallback_expansions


def test_fallback_expansions_arabic_contract():
    expansions = fallback_expansions("أحكام عقد الإيجار", "ar")
    assert expansions
    # Should include synonym substitutions for عقد
    all_text = " ".join(expansions)
    assert any(syn in all_text for syn in ["تعاقد", "اتفاقية", "ميثاق"])


def test_fallback_expansions_english_contract():
    expansions = fallback_expansions("the contract is void", "en")
    assert expansions
    all_text = " ".join(expansions)
    assert "agreement" in all_text


def test_fallback_expansions_article_query():
    expansions = fallback_expansions("المادة 131", "ar")
    assert expansions
    # Article queries get special expansion patterns
    all_text = " ".join(expansions)
    assert any(marker in all_text for marker in ["تفاصيل", "شرح", "أحكام"])