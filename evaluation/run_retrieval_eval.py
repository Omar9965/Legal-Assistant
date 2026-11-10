"""
Run the project's existing retriever against the golden set.

IMPORTANT:
1. Edit only retrieve_documents() below to call your actual ResearcherAgent /
   vector_store retrieval path. Do not create a second retrieval implementation.
2. The function must return ranked LangChain Documents, Chroma records, or dicts.
3. Run from the repository root so your project imports and .env/config work.

Example:
    python scripts/evaluation/run_retrieval_eval.py \
        --golden-set "Pasted text(8).txt" --top-k 10
"""

from __future__ import annotations

import argparse
import importlib
import json
import statistics
import time
from pathlib import Path
from typing import Any


def retrieve_documents(
    query: str,
    top_k: int,
    history=None,
    language: str = "ar",
):
    from controllers.researcher import ResearcherAgent
    from utils.config import TOP_K

    if top_k > TOP_K:
        raise ValueError(
            f"Requested top_k={top_k}, but configured TOP_K={TOP_K}"
        )

    researcher = ResearcherAgent()

    result = researcher.execute(
        query=query,
        language=language,
    )

    return result.get("retrieved_docs", [])[:top_k]

def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no}: {exc}") from exc
    return rows


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    # Normalize Arabic-Indic and Persian digits.
    table = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
    try:
        return int(str(value).translate(table).strip())
    except (TypeError, ValueError):
        return None


def article_number_from_result(item: Any) -> int | None:
    """Extract an article number from a LangChain Document, dict, or Chroma row."""
    metadata = {}
    if isinstance(item, dict):
        metadata = item.get("metadata") or item
    else:
        metadata = getattr(item, "metadata", {}) or {}

    for key in (
        "article_number", "article_id", "article", "article_no",
        "law_article", "number",
    ):
        number = _as_int(metadata.get(key))
        if number is not None:
            return number
    return None


def serialize_result(item: Any) -> dict[str, Any]:
    if isinstance(item, dict):
        metadata = item.get("metadata") or {}
        content = (
            item.get("page_content")
            or item.get("document")
            or item.get("text")
            or ""
        )
        result_id = item.get("id") or metadata.get("id")
    else:
        metadata = getattr(item, "metadata", {}) or {}
        content = getattr(item, "page_content", "") or str(item)
        result_id = getattr(item, "id", None)

    return {
        "id": result_id,
        "article_number": article_number_from_result(item),
        "metadata": metadata,
        "content_preview": str(content)[:500],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden-set", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("retrieval_results.jsonl"))
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--include-types",
        nargs="*",
        default=None,
        help="Optional query types to include, e.g. conceptual scenario followup",
    )
    parser.add_argument(
        "--include-unlabeled",
        action="store_true",
        help="Also run greeting/OOD records with no relevant_articles label",
    )
    args = parser.parse_args()

    if args.top_k < 1:
        parser.error("--top-k must be >= 1")

    rows = load_jsonl(args.golden_set)
    selected = []
    for row in rows:
        qtype = row.get("type", "unknown")
        if args.include_types and qtype not in args.include_types:
            continue
        if not row.get("relevant_articles") and not args.include_unlabeled:
            continue
        selected.append(row)

    if not selected:
        raise SystemExit("No records selected. Check the dataset and filters.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    latencies = []
    failures = 0

    with args.output.open("w", encoding="utf-8") as out:
        for index, row in enumerate(selected, start=1):
            started = time.perf_counter()
            error = None
            documents = []
            try:
                documents = retrieve_documents(
                    query=row["query"],
                    top_k=args.top_k,
                    history=row.get("history"),
                    language=row.get("language", "ar"),
                ) or []

                if not isinstance(documents, (list, tuple)):
                    documents = list(documents)
            except Exception as exc:
                failures += 1
                error = f"{type(exc).__name__}: {exc}"

            latency_ms = (time.perf_counter() - started) * 1000
            latencies.append(latency_ms)

            # Deduplicate article IDs while preserving the retriever's ranking.
            retrieved_articles = []
            seen = set()
            serialized = []
            for doc in documents:
                item = serialize_result(doc)
                serialized.append(item)
                article = item["article_number"]
                if article is not None and article not in seen:
                    seen.add(article)
                    retrieved_articles.append(article)

            result = {
                "id": row.get("id"),
                "query": row.get("query"),
                "language": row.get("language", "unknown"),
                "type": row.get("type", "unknown"),
                "history": row.get("history", []),
                "relevant_articles": row.get("relevant_articles", []),
                "retrieved_articles": retrieved_articles,
                "retrieved_documents": serialized,
                "latency_ms": round(latency_ms, 3),
                "error": error,
            }
            out.write(json.dumps(result, ensure_ascii=False) + "\n")

            status = "ERROR" if error else f"{len(retrieved_articles)} articles"
            print(f"[{index:03}/{len(selected):03}] {row.get('id')} "
                  f"{latency_ms:.1f} ms — {status}")

    successful = [x for x in latencies if x >= 0]
    print("\nRetrieval run complete")
    print(f"Queries: {len(selected)} | failures: {failures}")
    if successful:
        print(f"Mean latency: {statistics.mean(successful):.1f} ms")
        print(f"Median latency: {statistics.median(successful):.1f} ms")
    print(f"Raw results: {args.output}")


if __name__ == "__main__":
    main()
