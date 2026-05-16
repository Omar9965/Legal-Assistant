"""
Document Processor — Converts Arabic legal PDFs into structured chunks.

Fix: Uses sort=False (not sort=True) to preserve RTL Arabic text order.
sort=True spatially reorders text blocks, which reverses/scrambles Arabic.
"""

import os
import re
import hashlib
import unicodedata
from typing import Optional
import fitz
from langchain_core.documents import Document
from utils.constants import LEGAL_CATEGORIES




def _compute_file_hash(file_path: str) -> str:
    h = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# ── PDF Extraction ───────────────────────────────────────────────────────────

def process_pdf_with_pymupdf(
    pdf_path: str,
    save_text: bool = True,
    processed_dir: str = ".",
) -> str:
    """
    Extract text from an Arabic PDF, one page at a time.

    KEY FIX: sort=False preserves RTL reading order.
    sort=True reorders spans by x/y position which reverses Arabic words.

    Page numbers are embedded as sentinels so downstream code can track
    which page each article came from.
    """
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    md_path   = os.path.join(processed_dir, f"{base_name}.md")
    hash_path = os.path.join(processed_dir, f"{base_name}.md.hash")

    current_hash = _compute_file_hash(pdf_path)

    if os.path.exists(md_path) and os.path.exists(hash_path):
        with open(hash_path, "r", encoding="utf-8") as hf:
            if hf.read().strip() == current_hash:
                print(f"[Cache hit] '{base_name}' — skipping extraction.")
                with open(md_path, "r", encoding="utf-8") as mf:
                    return mf.read()

    print(f"[Extracting] '{base_name}'...")
    doc = fitz.open(pdf_path)
    pages_text: list[str] = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text", sort=False)
        if text.strip():
            # Embed a page sentinel so we never lose page provenance
            sentinel = f"<!-- page:{page_num + 1} -->"
            pages_text.append(f"{sentinel}\n{text}")

    doc.close()

    # Join with a clear page-break marker (distinct from paragraph breaks)
    full_text = "\n\n<!-- page_break -->\n\n".join(pages_text)

    # Normalize Arabic Presentation Forms → standard Arabic Unicode
    full_text = unicodedata.normalize("NFKC", full_text)

    # Clean up excessive whitespace while preserving structure
    full_text = _clean_text(full_text)

    if save_text:
        os.makedirs(processed_dir, exist_ok=True)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(full_text)
        with open(hash_path, "w", encoding="utf-8") as hf:
            hf.write(current_hash)
        print(f"[Saved] → {md_path}")

    return full_text


def _clean_text(text: str) -> str:
    """
    Clean extracted text while preserving page sentinels:
    - Remove lines that are only whitespace
    - Collapse 3+ newlines into 2
    - Strip trailing spaces per line
    - Re-join lines split mid-word by PDF layout, but never across
      page-break sentinels or known block starters
    """
    lines  = text.split("\n")
    cleaned: list[str] = []
    buffer  = ""

    # Patterns that always start a new block — never merge into previous line
    _NEW_BLOCK = re.compile(
        r"^(?:"
        r"مادة|مــادة|المادة"
        r"|الفصل|الباب|الكتاب|القسم"
        r"|أولاً|ثانياً|ثالثاً|رابعاً|خامساً"
        r"|سادساً|سابعاً|ثامناً|تاسعاً|عاشراً"
        r"|\d+[\.\-\)）]"          # numbered items  e.g. "1." "2-" "3)"
        r"|<!--"                   # sentinel lines
        r")"
    )

    # Characters that signal the previous line ended a complete thought
    _SENTENCE_END = set(".،؛:؟!\u200f\"»")

    for line in lines:
        stripped = line.strip()

        # Always flush buffer and pass sentinel lines through untouched
        if stripped.startswith("<!--"):
            if buffer:
                cleaned.append(buffer)
                buffer = ""
            cleaned.append(stripped)
            continue

        if not stripped:
            if buffer:
                cleaned.append(buffer)
                buffer = ""
            cleaned.append("")
            continue

        if buffer:
            last_char        = buffer[-1]
            ends_sentence    = last_char in _SENTENCE_END
            starts_new_block = bool(_NEW_BLOCK.match(stripped))

            if not ends_sentence and not starts_new_block:
                buffer = buffer + " " + stripped
            else:
                cleaned.append(buffer)
                buffer = stripped
        else:
            buffer = stripped

    if buffer:
        cleaned.append(buffer)

    result = "\n".join(cleaned)
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result.strip()


# ── Helpers ──────────────────────────────────────────────────────────────────

def extract_article_number(text: str) -> Optional[str]:
    patterns = [
        r"(?:مادة|مــادة|المادة)\s*[\(（]?\s*(\d+)\s*[\)）]?",
        r"[Aa]rt(?:icle)?\.?\s*(\d+)",
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            return m.group(1)
    return None


def _detect_category(text: str) -> str:
    """
    Assign a legal category using word-boundary aware matching so that
    a keyword embedded inside a longer word does not cause a false positive.
    Category order defines priority when multiple categories match.
    """
    for category, keywords in LEGAL_CATEGORIES.items():
        for kw in keywords:
            if re.search(rf"(?<!\w){re.escape(kw)}(?!\w)", text):
                return category
    return "general"


def _extract_page_number(text: str) -> Optional[int]:
    """Pull the page number from the nearest preceding page sentinel."""
    m = re.search(r"<!--\s*page:(\d+)\s*-->", text)
    return int(m.group(1)) if m else None


def _strip_sentinels(text: str) -> str:
    """Remove page sentinel comments from chunk content."""
    text = re.sub(r"<!--\s*page(?:_break|:\d+)\s*-->\n?", "", text)
    return text.strip()


def _fallback_chunk(text: str, chunk_size: int, overlap: int) -> list[str]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) + 2 > chunk_size and current:
            chunks.append(current.strip())
            current = (current[-overlap:] + "\n\n" + para) if overlap > 0 else para
        else:
            current = (current + "\n\n" + para) if current else para
    if current.strip():
        chunks.append(current.strip())
    if not chunks and text.strip():
        for i in range(0, len(text), chunk_size - overlap):
            chunks.append(text[i : i + chunk_size])
    return chunks


# ── Chunking ─────────────────────────────────────────────────────────────────

def chunk_legal_text(
    text: str,
    source: str = "law-131-1948.pdf",
    chunk_size: int = 1500,
    overlap: int = 200,
) -> list[Document]:
    """
    Split legal text into article-level chunks with rich metadata.

    Metadata per chunk:
        source          — original filename
        article_number  — normalized article number (str) or None
        category        — detected legal category
        language        — "ar"
        page_number     — page the article starts on (int) or None
        chunk_index     — article index within the document (int)
        sub_chunk_index — split index when an article exceeds chunk_size (int)
    """
    documents: list[Document] = []
    article_pattern = r"(?=(?:مادة|مــادة|المادة)\s*[\(（]?\s*(?:\d+|[٠-٩]+)\s*[\)）]?)"
    parts = re.split(article_pattern, text)

    def _make_doc(content: str, chunk_idx: int, sub_idx: int) -> Document:
        page_num    = _extract_page_number(content)
        clean       = _strip_sentinels(content)
        article_num = extract_article_number(clean)
        category    = _detect_category(clean)
        return Document(
            page_content=clean,
            metadata={
                "source":          source,
                "article_number":  article_num,
                "category":        category,
                "language":        "ar",
                "page_number":     page_num,
                "chunk_index":     chunk_idx,
                "sub_chunk_index": sub_idx,
            },
        )

    if len(parts) > 1:
        for i, part in enumerate(parts):
            part = part.strip()
            if not part or len(part) < 20:
                continue
            if len(part) > chunk_size * 2:
                for j, sc in enumerate(_fallback_chunk(part, chunk_size, overlap)):
                    documents.append(_make_doc(sc, i, j))
            else:
                documents.append(_make_doc(part, i, 0))
    else:
        # Fallback for non-article-structured text
        for i, chunk in enumerate(_fallback_chunk(text, chunk_size, overlap)):
            documents.append(_make_doc(chunk, i, 0))

    return documents


# ── High-level entry point ───────────────────────────────────────────────────

def process_pdf(pdf_path: str, processed_dir: str = ".") -> list[Document]:
    """Full pipeline: extract → clean → chunk → return Documents."""
    print(f"[Processing] {pdf_path}")
    text   = process_pdf_with_pymupdf(pdf_path, processed_dir=processed_dir)
    source = os.path.basename(pdf_path)
    docs   = chunk_legal_text(text, source=source)
    print(f"[Done] {len(docs)} chunks from '{source}'")
    return docs