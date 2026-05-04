"""
Ingestion Script — Process the base Egyptian Civil Code PDF and populate ChromaDB.

Usage: python process.py
"""

import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.document_processor import process_pdf, _compute_file_hash
from models.vector_store import add_documents, clear_collection, get_collection_count
from utils.config import PDF_DIR, LEGAL_AR_COLLECTION, PROCESSED_DIR

BATCH_SIZE = 500
MAX_WORKERS = min(4, (os.cpu_count() or 1))


def _process_single_pdf(pdf_file: str) -> tuple[str, list, float, str | None]:
    """Worker function to process a single PDF. Returns (filename, documents, elapsed, error)."""
    import os as _os
    import sys as _sys
    _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))

    from models.document_processor import process_pdf as _process_pdf
    from utils.config import PROCESSED_DIR as _PROCESSED_DIR, PDF_DIR as _PDF_DIR

    pdf_path = _os.path.join(_PDF_DIR, pdf_file)
    start_time = time.time()
    try:
        documents = _process_pdf(pdf_path, processed_dir=_PROCESSED_DIR)
        elapsed = time.time() - start_time
        return pdf_file, documents, elapsed, None
    except Exception as e:
        elapsed = time.time() - start_time
        return pdf_file, [], elapsed, str(e)


def _get_stored_hash(pdf_file: str) -> str | None:
    """Read stored hash for a PDF if it exists."""
    base_name = os.path.splitext(pdf_file)[0]
    hash_path = os.path.join(PROCESSED_DIR, f"{base_name}.md.hash")
    if os.path.exists(hash_path):
        with open(hash_path, "r") as f:
            return f.read().strip()
    return None


def _has_pdf_changed(pdf_file: str) -> bool:
    """Check if PDF has changed since last processing."""
    pdf_path = os.path.join(PDF_DIR, pdf_file)
    current_hash = _compute_file_hash(pdf_path)
    stored_hash = _get_stored_hash(pdf_file)
    return current_hash != stored_hash


def main():
    print("=" * 60)
    print("Legal AI — Document Ingestion Pipeline")
    print("=" * 60)

    pdf_files = [f for f in os.listdir(PDF_DIR) if f.endswith(".pdf")]

    if not pdf_files:
        print(f"No PDF files found in {PDF_DIR}")
        return

    print(f"\nFound {len(pdf_files)} PDF file(s) in {PDF_DIR}:")
    for f in pdf_files:
        changed = _has_pdf_changed(f)
        status = "changed" if changed else "unchanged"
        print(f"   • {f} [{status}]")

    changed_pdfs = [f for f in pdf_files if _has_pdf_changed(f)]

    if not changed_pdfs:
        print("\n✓ All PDFs unchanged since last processing. Skipping.")
        count = get_collection_count(LEGAL_AR_COLLECTION)
        print(f"  Collection '{LEGAL_AR_COLLECTION}' has {count} documents.")
        print("\n" + "=" * 60)
        print(" Nothing to do. Run the app with: python app.py")
        print("=" * 60)
        return

    print(f"\n{len(changed_pdfs)} PDF(s) have changed. Processing...")

    if changed_pdfs:
        print(f"\nClearing '{LEGAL_AR_COLLECTION}' collection...")
        clear_collection(LEGAL_AR_COLLECTION)

    all_documents = []

    if len(changed_pdfs) == 1:
        pdf_file = changed_pdfs[0]
        print(f"\nProcessing: {pdf_file}")
        start_time = time.time()
        try:
            documents = process_pdf(os.path.join(PDF_DIR, pdf_file), processed_dir=PROCESSED_DIR)
            all_documents.extend(documents)
            elapsed = time.time() - start_time
            print(f"{len(documents)} chunks extracted in {elapsed:.1f}s")
        except Exception as e:
            print(f"Error processing {pdf_file}: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"\nExtracting PDFs in parallel ({MAX_WORKERS} workers)...")
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(_process_single_pdf, pdf): pdf for pdf in changed_pdfs}
            for future in as_completed(futures):
                pdf_file, documents, elapsed, error = future.result()
                if error:
                    print(f"Error processing {pdf_file}: {error}")
                else:
                    all_documents.extend(documents)
                    print(f"✓ {pdf_file}: {len(documents)} chunks in {elapsed:.1f}s")

    if not all_documents:
        print("\nNo documents were extracted. Check the PDF files.")
        return

    print(f"\nAdding {len(all_documents)} chunks to ChromaDB...")
    start_time = time.time()

    for i in range(0, len(all_documents), BATCH_SIZE):
        batch = all_documents[i:i + BATCH_SIZE]
        add_documents(batch, LEGAL_AR_COLLECTION)
        print(f"  Added batch {i // BATCH_SIZE + 1} ({len(batch)} docs)")

    elapsed = time.time() - start_time
    print(f"Done in {elapsed:.1f}s")

    count = get_collection_count(LEGAL_AR_COLLECTION)
    print(f"\nCollection '{LEGAL_AR_COLLECTION}' now has {count} documents.")

    print("\n" + "=" * 60)
    print(" Processing complete! You can now run the app")
    print("=" * 60)


if __name__ == "__main__":
    main()
