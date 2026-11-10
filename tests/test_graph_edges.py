"""Tests for LangGraph conditional edge routing."""

from controllers.graph_edges import (
    cache_hit_check,
    is_legal_check,
    retrieval_quality_check,
)


def test_is_legal_check_legal():
    assert is_legal_check({"is_legal": True}) == "legal"


def test_is_legal_check_not_legal():
    assert is_legal_check({"is_legal": False}) == "not_legal"


def test_cache_hit_check_hit():
    assert cache_hit_check({"cache_hit": True}) == "cache_hit"


def test_cache_hit_check_miss():
    assert cache_hit_check({"cache_hit": False}) == "cache_miss"


def test_retrieval_quality_retry_on_empty():
    assert retrieval_quality_check({"last_retrieval_quality": "empty", "retrieval_attempts": 1}) == "retry"


def test_retrieval_quality_retry_on_insufficient():
    assert retrieval_quality_check({"last_retrieval_quality": "insufficient", "retrieval_attempts": 1}) == "retry"


def test_retrieval_quality_proceed_after_max_attempts():
    assert retrieval_quality_check({"last_retrieval_quality": "empty", "retrieval_attempts": 2}) == "proceed"


def test_retrieval_quality_proceed_on_success():
    assert retrieval_quality_check({"last_retrieval_quality": "success", "retrieval_attempts": 1}) == "proceed"
