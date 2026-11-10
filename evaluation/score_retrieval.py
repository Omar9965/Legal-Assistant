"""
Calculate deterministic article-level retrieval metrics.

Input: retrieval_results.jsonl produced by run_retrieval_eval.py
Outputs:
  - retrieval_metrics.json
  - retrieval_per_query.csv

Metrics are calculated only for records with relevant_articles labels.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


K_VALUES = (1, 3, 5, 10)


def query_metrics(relevant: list[int], retrieved: list[int]) -> dict[str, float]:
    relevant_set = set(relevant)
    ranked = retrieved

    result: dict[str, float] = {}
    for k in K_VALUES:
        top_k = ranked[:k]
        hits = relevant_set.intersection(top_k)
        result[f"recall@{k}"] = len(hits) / len(relevant_set) if relevant_set else 0.0
        result[f"precision@{k}"] = len(hits) / k
        result[f"hit_rate@{k}"] = float(bool(hits))

    reciprocal_rank = 0.0
    for rank, article in enumerate(ranked, start=1):
        if article in relevant_set:
            reciprocal_rank = 1.0 / rank
            break
    result["mrr"] = reciprocal_rank
    return result


def mean_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    metric_names = list(rows[0]["metrics"]) if rows else []
    return {
        name: round(statistics.mean(r["metrics"][name] for r in rows), 4)
        for name in metric_names
    }


def grouped_metrics(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(key, "unknown"))].append(row)
    return {
        group: {
            "query_count": len(items),
            "metrics": mean_metrics(items),
        }
        for group, items in sorted(groups.items())
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("retrieval_results.jsonl"))
    parser.add_argument("--json-output", type=Path, default=Path("retrieval_metrics.json"))
    parser.add_argument("--csv-output", type=Path, default=Path("retrieval_per_query.csv"))
    args = parser.parse_args()

    rows = []
    with args.input.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            relevant = row.get("relevant_articles") or []
            if not relevant:
                continue
            metrics = query_metrics(relevant, row.get("retrieved_articles") or [])
            row["metrics"] = metrics
            rows.append(row)

    if not rows:
        raise SystemExit("No labeled records found in the results file.")

    latency = [r["latency_ms"] for r in rows if r.get("latency_ms") is not None]
    ordered_latency = sorted(latency)

    def percentile(values: list[float], p: float) -> float | None:
        if not values:
            return None
        index = round((len(values) - 1) * p)
        return round(values[index], 2)

    errors = sum(bool(r.get("error")) for r in rows)
    output = {
        "dataset": {
            "evaluated_queries": len(rows),
            "failed_queries": errors,
            "k_values": list(K_VALUES),
            "relevance_unit": "article_number",
        },
        "overall": mean_metrics(rows),
        "by_type": grouped_metrics(rows, "type"),
        "by_language": grouped_metrics(rows, "language"),
        "latency_ms": {
            "mean": round(statistics.mean(latency), 2) if latency else None,
            "median": round(statistics.median(latency), 2) if latency else None,
            "p95": percentile(ordered_latency, 0.95),
        },
        "notes": [
            "Macro-average: each query contributes equally.",
            "Article IDs are deduplicated in ranked retrieval order.",
            "Unlabeled greeting/out-of-domain records are excluded.",
            "Article lookup queries should also be inspected separately from semantic queries.",
        ],
    }

    args.json_output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    columns = [
        "id", "type", "language", "latency_ms", "error",
        "relevant_articles", "retrieved_articles",
        *[f"recall@{k}" for k in K_VALUES],
        *[f"precision@{k}" for k in K_VALUES],
        *[f"hit_rate@{k}" for k in K_VALUES],
        "mrr",
    ]
    with args.csv_output.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            flat = dict(row)
            flat["relevant_articles"] = json.dumps(row["relevant_articles"], ensure_ascii=False)
            flat["retrieved_articles"] = json.dumps(row["retrieved_articles"], ensure_ascii=False)
            flat.update(row["metrics"])
            writer.writerow(flat)

    print(json.dumps(output["overall"], indent=2))
    print(f"\nMetrics JSON: {args.json_output}")
    print(f"Per-query CSV: {args.csv_output}")


if __name__ == "__main__":
    main()
