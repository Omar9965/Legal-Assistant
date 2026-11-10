"""Tests for the heuristic RouterAgent (language, greeting, legal classification)."""

from controllers.router import RouterAgent

agent = RouterAgent()


def test_english_greeting():
    r = agent.execute("Hello")
    assert r["is_greeting"] is True
    assert r["is_legal"] is False
    assert r["language"] == "en"


def test_english_greeting_phrase():
    r = agent.execute("good morning")
    assert r["is_greeting"] is True


def test_substring_hi_is_not_greeting():
    # Regression: "hi" inside "this" must NOT be treated as a greeting.
    r = agent.execute("What is this article about?")
    assert r["is_greeting"] is False
    assert r["is_legal"] is True


def test_legal_question_not_greeting():
    r = agent.execute("What does article 131 of the Egyptian civil code say?")
    assert r["is_greeting"] is False
    assert r["is_legal"] is True


def test_arabic_greeting():
    r = agent.execute("مرحبا")
    assert r["is_greeting"] is True
    assert r["language"] == "ar"


def test_arabic_legal_question():
    r = agent.execute("ما هي أحكام المادة 131 من القانون المدني المصري؟")
    assert r["language"] == "ar"
    assert r["is_legal"] is True
    assert r["is_greeting"] is False


def test_non_legal_english():
    r = agent.execute("give me a pizza recipe")
    assert r["is_legal"] is False


def test_language_detection_prefers_arabic():
    r = agent.execute("ما هي المادة 131 about")
    assert r["language"] == "ar"


def test_empty_query_defaults_to_legal():
    r = agent.execute("")
    assert r["is_greeting"] is False
    assert r["is_legal"] is True
