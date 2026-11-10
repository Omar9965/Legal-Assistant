"""Tests for document processing helpers (article extraction, hash skip logic)."""

from models.document_processor import (
    _compute_file_hash,
    extract_article_number,
    is_unchanged,
)


def test_extract_article_number_arabic():
    assert extract_article_number("تنص المادة 131 على ...") == "131"


def test_extract_article_number_arabic_parenthesized():
    assert extract_article_number("مادة (٥)") == "5" or extract_article_number("مادة (5)") == "5"


def test_extract_article_number_english():
    assert extract_article_number("Article 20 states that ...") == "20"


def test_extract_article_number_none():
    assert extract_article_number("ما هي أحكام البيع؟") is None


def test_compute_file_hash_stable(tmp_path):
    p = tmp_path / "sample.pdf"
    p.write_bytes(b"%PDF-1.4 fake content")
    assert _compute_file_hash(str(p)) == _compute_file_hash(str(p))


def test_is_unchanged_false_when_not_processed(tmp_path):
    p = tmp_path / "doc.pdf"
    p.write_bytes(b"pdf-bytes")
    assert is_unchanged(str(p), str(tmp_path)) is False


def test_is_unchanged_true_when_hash_matches(tmp_path):
    p = tmp_path / "doc.pdf"
    p.write_bytes(b"pdf-bytes")
    (tmp_path / "doc.md").write_text("extracted text", encoding="utf-8")
    (tmp_path / "doc.md.hash").write_text(_compute_file_hash(str(p)), encoding="utf-8")
    assert is_unchanged(str(p), str(tmp_path)) is True


def test_is_unchanged_false_when_hash_differs(tmp_path):
    p = tmp_path / "doc.pdf"
    p.write_bytes(b"pdf-bytes")
    (tmp_path / "doc.md").write_text("extracted text", encoding="utf-8")
    (tmp_path / "doc.md.hash").write_text("deadbeef", encoding="utf-8")
    assert is_unchanged(str(p), str(tmp_path)) is False
