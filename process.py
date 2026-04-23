"""
Ingestion Script — Process the base Egyptian Civil Code PDF and populate ChromaDB.

Usage: python ingest.py
"""

import os
import sys
import time



# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.document_processor import process_pdf
from models.vector_store import add_documents, clear_collection, get_collection_count
from utils.config import PDF_DIR, LEGAL_AR_COLLECTION, PROCESSED_DIR


def main():
    print("=" * 60)
    print("Legal AI — Document Ingestion Pipeline")
    print("=" * 60)

    # Find PDF files
    pdf_files = [f for f in os.listdir(PDF_DIR) if f.endswith(".pdf")]
    
    if not pdf_files:
        print(f"No PDF files found in {PDF_DIR}")
        return

    print(f"\nFound {len(pdf_files)} PDF file(s) in {PDF_DIR}:")
    for f in pdf_files:
        print(f"   • {f}")

    # Clear existing collection for clean re-ingestion
    print(f"\nClearing existing '{LEGAL_AR_COLLECTION}' collection...")
    clear_collection(LEGAL_AR_COLLECTION)

    # Process each PDF
    all_documents = []
    for pdf_file in pdf_files:
        pdf_path = os.path.join(PDF_DIR, pdf_file)
        print(f"\nProcessing: {pdf_file}")
        start_time = time.time()

        try:
            documents = process_pdf(pdf_path, processed_dir=PROCESSED_DIR)
            all_documents.extend(documents)
            elapsed = time.time() - start_time
            print(f"{len(documents)} chunks extracted in {elapsed:.1f}s")
        except Exception as e:
            print(f"Error processing {pdf_file}: {e}")
            import traceback
            traceback.print_exc()

    if not all_documents:
        print("\nNo documents were extracted. Check the PDF files.")
        return

    # Add to ChromaDB
    print(f"\nAdding {len(all_documents)} chunks to ChromaDB...")
    start_time = time.time()
    add_documents(all_documents, LEGAL_AR_COLLECTION)
    elapsed = time.time() - start_time
    print(f"Done in {elapsed:.1f}s")

    # Verify
    count = get_collection_count(LEGAL_AR_COLLECTION)
    print(f"\nCollection '{LEGAL_AR_COLLECTION}' now has {count} documents.")

    print("\n" + "=" * 60)
    print(" Processing complete! You can now run the app")
    print("=" * 60)


if __name__ == "__main__":
    main()
