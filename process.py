"""
Ingestion Script — Process the base Egyptian Civil Code PDF and populate ChromaDB.

Usage: python process.py
"""

import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.document_processor import process_pdf, _compute_file_hash
from models.vector_store import add_documents, clear_collection, get_collection_count
from utils.config import PDF_DIR, LEGAL_AR_COLLECTION, PROCESSED_DIR
from langchain_core.documents import Document

BATCH_SIZE = 500
MAX_WORKERS = min(4, (os.cpu_count() or 1))


def _process_single_pdf_worker(pdf_file: str) -> Tuple[str, List[Document], float, Optional[str]]:
    """Worker function to process a single PDF. Must be at module level for ProcessPoolExecutor pickling."""
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


class FileChangeDetector:
    """Responsible for detecting if files have changed since last ingestion. (SRP)"""
    def __init__(self, pdf_dir: str, processed_dir: str):
        self.pdf_dir = pdf_dir
        self.processed_dir = processed_dir

    def _get_stored_hash(self, pdf_file: str) -> Optional[str]:
        base_name = os.path.splitext(pdf_file)[0]
        hash_path = os.path.join(self.processed_dir, f"{base_name}.md.hash")
        if os.path.exists(hash_path):
            with open(hash_path, "r") as f:
                return f.read().strip()
        return None

    def has_changed(self, pdf_file: str) -> bool:
        pdf_path = os.path.join(self.pdf_dir, pdf_file)
        current_hash = _compute_file_hash(pdf_path)
        stored_hash = self._get_stored_hash(pdf_file)
        return current_hash != stored_hash


class PDFExtractor:
    """Responsible for orchestrating the extraction of text from PDFs. (SRP)"""
    def __init__(self, pdf_dir: str, processed_dir: str, max_workers: int):
        self.pdf_dir = pdf_dir
        self.processed_dir = processed_dir
        self.max_workers = max_workers

    def extract_all(self, pdf_files: List[str]) -> List[Document]:
        all_documents = []
        if len(pdf_files) == 1:
            all_documents.extend(self._extract_single(pdf_files[0]))
        else:
            all_documents.extend(self._extract_parallel(pdf_files))
        return all_documents

    def _extract_single(self, pdf_file: str) -> List[Document]:
        print(f"\nProcessing: {pdf_file}")
        start_time = time.time()
        try:
            pdf_path = os.path.join(self.pdf_dir, pdf_file)
            documents = process_pdf(pdf_path, processed_dir=self.processed_dir)
            elapsed = time.time() - start_time
            print(f"{len(documents)} chunks extracted in {elapsed:.1f}s")
            return documents
        except Exception as e:
            print(f"Error processing {pdf_file}: {e}")
            import traceback
            traceback.print_exc()
            return []

    def _extract_parallel(self, pdf_files: List[str]) -> List[Document]:
        print(f"\nExtracting PDFs in parallel ({self.max_workers} workers)...")
        all_documents = []
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(_process_single_pdf_worker, pdf): pdf for pdf in pdf_files}
            for future in as_completed(futures):
                pdf_file, documents, elapsed, error = future.result()
                if error:
                    print(f"Error processing {pdf_file}: {error}")
                else:
                    all_documents.extend(documents)
                    print(f"✓ {pdf_file}: {len(documents)} chunks in {elapsed:.1f}s")
        return all_documents


class VectorStoreIngester:
    """Responsible for persisting documents into the vector store. (SRP)"""
    def __init__(self, collection_name: str, batch_size: int):
        self.collection_name = collection_name
        self.batch_size = batch_size

    def clear(self) -> None:
        print(f"\nClearing '{self.collection_name}' collection...")
        clear_collection(self.collection_name)

    def ingest(self, documents: List[Document]) -> None:
        if not documents:
            print("\nNo documents to ingest.")
            return

        print(f"\nAdding {len(documents)} chunks to ChromaDB...")
        start_time = time.time()

        for i in range(0, len(documents), self.batch_size):
            batch = documents[i:i + self.batch_size]
            add_documents(batch, self.collection_name)
            print(f"  Added batch {i // self.batch_size + 1} ({len(batch)} docs)")

        elapsed = time.time() - start_time
        print(f"Done in {elapsed:.1f}s")

    def print_stats(self) -> None:
        count = get_collection_count(self.collection_name)
        print(f"\nCollection '{self.collection_name}' now has {count} documents.")


class IngestionPipeline:
    """Facade pattern that orchestrates the entire document ingestion workflow."""
    def __init__(
        self, 
        pdf_dir: str, 
        detector: FileChangeDetector, 
        extractor: PDFExtractor, 
        ingester: VectorStoreIngester
    ):
        self.pdf_dir = pdf_dir
        self.detector = detector
        self.extractor = extractor
        self.ingester = ingester

    def run(self) -> None:
        print("=" * 60)
        print("Legal AI — Document Ingestion Pipeline")
        print("=" * 60)

        pdf_files = [f for f in os.listdir(self.pdf_dir) if f.endswith(".pdf")]

        if not pdf_files:
            print(f"No PDF files found in {self.pdf_dir}")
            return

        print(f"\nFound {len(pdf_files)} PDF file(s) in {self.pdf_dir}:")
        changed_pdfs = []
        for f in pdf_files:
            changed = self.detector.has_changed(f)
            if changed:
                changed_pdfs.append(f)
            status = "changed" if changed else "unchanged"
            print(f"   • {f} [{status}]")

        if not changed_pdfs:
            print("\n✓ All PDFs unchanged since last processing. Skipping.")
            self.ingester.print_stats()
            print("\n" + "=" * 60)
            print(" Nothing to do. Run the app with: python app.py")
            print("=" * 60)
            return

        print(f"\n{len(changed_pdfs)} PDF(s) have changed. Processing...")

        # IMPORTANT: If any PDF changed, we clear the collection and must reprocess ALL
        # to prevent data loss.
        self.ingester.clear()
        
        all_documents = self.extractor.extract_all(pdf_files)
        
        if not all_documents:
            print("\nNo documents were extracted. Check the PDF files.")
            return

        self.ingester.ingest(all_documents)
        self.ingester.print_stats()

        print("\n" + "=" * 60)
        print(" Processing complete! You can now run the app")
        print("=" * 60)


def main():
    # Dependency Injection
    detector = FileChangeDetector(pdf_dir=PDF_DIR, processed_dir=PROCESSED_DIR)
    extractor = PDFExtractor(pdf_dir=PDF_DIR, processed_dir=PROCESSED_DIR, max_workers=MAX_WORKERS)
    ingester = VectorStoreIngester(collection_name=LEGAL_AR_COLLECTION, batch_size=BATCH_SIZE)
    
    pipeline = IngestionPipeline(
        pdf_dir=PDF_DIR,
        detector=detector,
        extractor=extractor,
        ingester=ingester
    )
    
    pipeline.run()


if __name__ == "__main__":
    main()
