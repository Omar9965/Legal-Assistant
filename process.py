"""
Ingestion Script — Process the base Egyptian Civil Code PDF and populate ChromaDB.

Usage: python process.py

Skip logic per PDF: if the vector store is empty there is nothing to preserve,
so all PDFs are ingested (ignoring hash). Otherwise a PDF is skipped only when
its extracted text + MD5 hash on disk match the current file (is_unchanged).

Safety flow per PDF: extract FIRST (fail fast, keep existing data), and only
delete the old chunks once extraction has succeeded — a failed extraction can
no longer silently erase knowledge.
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.document_processor import process_pdf, is_unchanged
from models.vector_store import (
    add_documents, delete_documents_by_source, get_collection_count
)
from models import cache
from utils.config import PDF_DIR, LEGAL_AR_COLLECTION, PROCESSED_DIR

BATCH_SIZE = 500


def main():
    print("=" * 60)
    print("Legal AI — Document Ingestion Pipeline")
    print("=" * 60)

    if not os.path.exists(PDF_DIR):
        print(f"Directory {PDF_DIR} does not exist.")
        return

    pdf_files = [f for f in os.listdir(PDF_DIR) if f.endswith(".pdf")]

    if not pdf_files:
        print(f"No PDF files found in {PDF_DIR}")
        return

    print(f"\nFound {len(pdf_files)} PDF file(s) in {PDF_DIR}:")

    # Empty index = no data dir effectively → ingest everything, hash is moot
    force_all = get_collection_count(LEGAL_AR_COLLECTION) == 0
    if force_all:
        print("[Ingestion] No existing index found — ingesting all PDFs (ignoring hash).")

    changed_pdfs = []
    for f in pdf_files:
        pdf_path = os.path.join(PDF_DIR, f)
        changed = force_all or not is_unchanged(pdf_path, PROCESSED_DIR)
        if changed:
            changed_pdfs.append(f)
        status = "changed" if changed else "unchanged"
        print(f"   • {f} [{status}]")

    if not changed_pdfs:
        count = get_collection_count(LEGAL_AR_COLLECTION)
        print("\n✓ All PDFs unchanged since last processing. Skipping.")
        print(f"\nCollection '{LEGAL_AR_COLLECTION}' now has {count} documents.")
        print("\n" + "=" * 60)
        print(" Nothing to do. Run the app with: python app.py")
        print("=" * 60)
        return

    print(f"\n{len(changed_pdfs)} PDF(s) to process...")

    added_any = False
    for pdf_file in changed_pdfs:
        pdf_path = os.path.join(PDF_DIR, pdf_file)
        start_time = time.time()

        print(f"\n[Extracting] '{pdf_file}'...")
        try:
            documents = process_pdf(pdf_path, processed_dir=PROCESSED_DIR)
        except Exception as e:
            print(f"Error extracting {pdf_file}: {e}")
            print("  KEPT existing chunks — no data was removed.")
            continue

        if not documents:
            print(f"Warning: extraction returned no chunks for '{pdf_file}'.")
            print("  KEPT existing chunks — no data was removed.")
            continue

        # Extraction succeeded → now it is safe to replace old chunks
        print(f"  Removing old chunks for '{pdf_file}'...")
        delete_documents_by_source(pdf_file, LEGAL_AR_COLLECTION)

        print(f"  Adding {len(documents)} chunks to ChromaDB...")
        for i in range(0, len(documents), BATCH_SIZE):
            batch = documents[i:i + BATCH_SIZE]
            add_documents(batch, LEGAL_AR_COLLECTION)
            print(f"    Added batch {i // BATCH_SIZE + 1} ({len(batch)} docs)")

        added_any = True
        print(f"✓ {pdf_file}: {len(documents)} chunks in {time.time() - start_time:.1f}s")

    count = get_collection_count(LEGAL_AR_COLLECTION)
    print(f"\nCollection '{LEGAL_AR_COLLECTION}' now has {count} documents.")

    if added_any:
        cache.invalidate()
        print("[Ingestion] Response cache invalidated (corpus changed).")

    print("\n" + "=" * 60)
    print(" Processing complete! You can now run the app")
    print("=" * 60)


if __name__ == "__main__":
    main()
