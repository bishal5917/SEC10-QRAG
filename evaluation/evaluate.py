"""
RAG Evaluation Runner
=====================
Runs ground-truth QnA pairs from qna_data.csv against the live API
and computes all evaluation metrics across retrieval, generation,
and end-to-end categories.

Usage:
    # Full evaluation
    python evaluation/evaluate.py --csv /path/to/qna_data.csv

    # Quick smoke-test
    python evaluation/evaluate.py --csv /path/to/qna_data.csv --limit 10

    # Only table questions
    python evaluation/evaluate.py --csv /path/to/qna_data.csv --chunk-type Table

    # Save full results
    python evaluation/evaluate.py --csv /path/to/qna_data.csv --output results.json

    python evaluation/evaluate.py --csv /data/csvs/qna_data.csv --limit 5
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Optional

import httpx

sys.path.insert(0, str(Path(__file__).parent.parent))
from evaluation.metrics import compute_all
from src.logger import get_logger

log = get_logger("evaluate")

AVAILABLE_TICKERS = {"AAPL", "AMZN", "INTC", "NVDA"}

# Metric groups for display
RETRIEVAL_METRICS   = ["context_precision", "context_recall", "context_f1", "mrr", "ndcg"]
GENERATION_METRICS  = ["bleu", "rouge_1", "rouge_2", "rouge_l", "meteor", "bertscore"]
END_TO_END_METRICS  = ["faithfulness", "hallucination_rate", "factual_consistency",
                        "answer_relevance", "exact_number_match"]
ALL_METRICS = RETRIEVAL_METRICS + GENERATION_METRICS + END_TO_END_METRICS


def _is_answerable(source_docs: str) -> bool:
    return any(t in source_docs for t in AVAILABLE_TICKERS)


def _query_api(base_url: str, question: str, timeout: float = 180.0) -> Optional[dict]:
    try:
        resp = httpx.post(
            f"{base_url}/query",
            json={"question": question, "top_k": 8},
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        log.error(f"API call failed: {e}")
        return None


def _bar(score: float, width: int = 20) -> str:
    """ASCII progress bar for a 0-1 score."""
    filled = int(score * width)
    return f"[{'█' * filled}{'░' * (width - filled)}] {score:.3f}"


def _avg(results: list, metric: str) -> float:
    vals = [r["metrics"][metric] for r in results if metric in r["metrics"]]
    return round(sum(vals) / len(vals), 4) if vals else 0.0


def _print_report(results: list, report: dict) -> None:
    avgs = report["summary"]
    n = avgs["total_questions"]

    print("\n" + "═" * 65)
    print("  RAG EVALUATION REPORT")
    print("═" * 65)
    print(f"  Questions evaluated : {n}")
    print(f"  Avg latency         : {avgs['avg_latency_s']}s per question")

    # ── Retrieval ──────────────────────────────────────────────
    print("\n  ┌─ RETRIEVAL METRICS " + "─" * 44)
    print("  │  (Did the retriever fetch the right chunks?)")
    print("  │")
    labels = {
        "context_precision": "Context Precision  — of retrieved chunks, how many are relevant?",
        "context_recall":    "Context Recall     — of all needed info, how much was retrieved?",
        "context_f1":        "Context F1         — harmonic mean of precision & recall",
        "mrr":               "MRR                — first relevant chunk at rank? (1.0=rank1)",
        "ndcg":              "nDCG               — ranking quality (relevant chunks ranked high?)",
    }
    for m in RETRIEVAL_METRICS:
        score = avgs[m]
        print(f"  │  {_bar(score)}  {labels[m]}")

    # ── Generation ─────────────────────────────────────────────
    print("\n  ├─ GENERATION METRICS " + "─" * 43)
    print("  │  (Is the generated text close to the reference answer?)")
    print("  │")
    labels = {
        "bleu":      "BLEU               — n-gram precision vs reference (brevity-penalized)",
        "rouge_1":   "ROUGE-1            — unigram overlap with reference",
        "rouge_2":   "ROUGE-2            — bigram overlap with reference",
        "rouge_l":   "ROUGE-L            — longest common subsequence F1",
        "meteor":    "METEOR             — recall + synonym matching + word order",
        "bertscore": "BERTScore          — semantic similarity (meaning, not just words)",
    }
    for m in GENERATION_METRICS:
        score = avgs[m]
        print(f"  │  {_bar(score)}  {labels[m]}")

    # ── End-to-end ─────────────────────────────────────────────
    print("\n  ├─ END-TO-END METRICS " + "─" * 43)
    print("  │  (Is the answer correct, grounded, and relevant?)")
    print("  │")
    labels = {
        "faithfulness":        "Faithfulness       — answer grounded in context (no hallucination)",
        "hallucination_rate":  "Hallucination Rate — fraction of answer NOT in context (lower=better)",
        "factual_consistency": "Factual Consistency— faithful + correct numbers combined",
        "answer_relevance":    "Answer Relevance   — answer addresses the question",
        "exact_number_match":  "Exact Number Match — correct financial figures cited ★ key metric",
    }
    for m in END_TO_END_METRICS:
        score = avgs[m]
        # Hallucination rate: lower is better — invert bar
        if m == "hallucination_rate":
            bar = f"[{'█' * int(score*20)}{'░' * (20-int(score*20))}] {score:.3f} ↓ lower=better"
        else:
            bar = _bar(score)
        print(f"  │  {bar}  {labels[m]}")

    # ── By chunk type ──────────────────────────────────────────
    print("\n  ├─ BREAKDOWN BY CHUNK TYPE " + "─" * 38)
    print(f"  │  {'Chunk Type':<10} {'Prec':>6} {'Recall':>7} {'ROUGE-L':>8} {'ExactNum':>9} {'Faith':>7} {'Halluc':>7}")
    print("  │  " + "-" * 58)
    for ct, ms in report["by_chunk_type"].items():
        print(f"  │  {ct:<10} {ms['context_precision']:>6.3f} {ms['context_recall']:>7.3f} "
              f"{ms['rouge_l']:>8.3f} {ms['exact_number_match']:>9.3f} "
              f"{ms['faithfulness']:>7.3f} {ms['hallucination_rate']:>7.3f}")

    # ── By question type ───────────────────────────────────────
    print("\n  ├─ BREAKDOWN BY QUESTION TYPE " + "─" * 35)
    print(f"  │  {'Question Type':<25} {'MRR':>5} {'nDCG':>6} {'BLEU':>6} {'ROUGE-L':>8} {'ExactNum':>9}")
    print("  │  " + "-" * 61)
    for qt, ms in report["by_question_type"].items():
        short = qt.replace("Single-Doc ", "").replace(" RAG", "")
        print(f"  │  {short:<25} {ms['mrr']:>5.3f} {ms['ndcg']:>6.3f} "
              f"{ms['bleu']:>6.3f} {ms['rouge_l']:>8.3f} {ms['exact_number_match']:>9.3f}")

    # ── Weak spots ─────────────────────────────────────────────
    print("\n  └─ WEAK SPOTS (metrics scoring below 0.5) " + "─" * 22)
    weak = [(m, avgs[m]) for m in ALL_METRICS if m != "hallucination_rate" and avgs[m] < 0.5]
    if weak:
        for m, v in sorted(weak, key=lambda x: x[1]):
            print(f"     ⚠  {m:<25} {v:.3f}")
    else:
        print("     ✓  All metrics above 0.5")

    print("═" * 65 + "\n")


def run_evaluation(
    csv_path: str,
    base_url: str = "http://localhost:8000",
    limit: Optional[int] = None,
    chunk_type_filter: Optional[str] = None,
    output_path: Optional[str] = None,
):
    with open(csv_path, encoding="utf-8-sig") as f:
        all_rows = list(csv.DictReader(f))

    rows = [r for r in all_rows if _is_answerable(r["Source Docs"])]
    if chunk_type_filter:
        rows = [r for r in rows if r["Source Chunk Type"].lower() == chunk_type_filter.lower()]
    if limit:
        rows = rows[:limit]

    log.info("=" * 60)
    log.info(f"Evaluating {len(rows)} questions | filter={chunk_type_filter or 'all'}")
    log.info("=" * 60)

    results = []
    totals = {m: 0.0 for m in ALL_METRICS}

    for i, row in enumerate(rows, 1):
        question  = row["Question"]
        reference = row["Answer"]
        q_type    = row["Question Type"]
        chunk_type = row["Source Chunk Type"]

        log.info(f"[{i}/{len(rows)}] {chunk_type} | {question[:65]}...")

        t0 = time.perf_counter()
        api_result = _query_api(base_url, question)
        elapsed = time.perf_counter() - t0

        if api_result is None:
            log.warning("  Skipped — API returned no result")
            continue

        generated  = api_result["answer"]
        chunk_dicts = [{"text": c["text_preview"]} for c in api_result.get("chunks", [])]
        metrics = compute_all(question, generated, reference, chunk_dicts)

        log.info(
            f"  recall={metrics['context_recall']:.2f} "
            f"faith={metrics['faithfulness']:.2f} "
            f"rouge_l={metrics['rouge_l']:.2f} "
            f"exact_num={metrics['exact_number_match']:.2f} "
            f"halluc={metrics['hallucination_rate']:.2f} "
            f"({elapsed:.1f}s)"
        )

        results.append({
            "question":          question,
            "question_type":     q_type,
            "chunk_type":        chunk_type,
            "source_docs":       row["Source Docs"],
            "generated_answer":  generated,
            "reference_answer":  reference,
            "metrics":           metrics,
            "latency_s":         round(elapsed, 2),
        })
        for m in ALL_METRICS:
            totals[m] += metrics.get(m, 0.0)

    if not results:
        log.error("No results collected.")
        return

    n = len(results)

    def avg_by(key: str) -> dict:
        groups: dict = {}
        for r in results:
            groups.setdefault(r[key], []).append(r["metrics"])
        return {
            g: {m: round(sum(x.get(m, 0) for x in ms) / len(ms), 4) for m in ALL_METRICS}
            for g, ms in groups.items()
        }

    summary = {
        "total_questions": n,
        "avg_latency_s": round(sum(r["latency_s"] for r in results) / n, 2),
        **{m: round(totals[m] / n, 4) for m in ALL_METRICS},
    }

    report = {
        "summary":          summary,
        "by_question_type": avg_by("question_type"),
        "by_chunk_type":    avg_by("chunk_type"),
        "per_question":     results,
    }

    _print_report(results, report)

    if output_path:
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        log.info(f"Full results saved to: {output_path}")

    return report


def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG system against ground truth QnA")
    parser.add_argument("--csv",        required=True,                        help="Path to qna_data.csv")
    parser.add_argument("--url",        default="http://localhost:8000",       help="RAG API base URL")
    parser.add_argument("--limit",      type=int,                              help="Max questions (quick test)")
    parser.add_argument("--chunk-type", choices=["Table", "Text"],             help="Filter by chunk type")
    parser.add_argument("--output",                                            help="Save results to JSON file")
    args = parser.parse_args()

    try:
        httpx.get(f"{args.url}/health", timeout=5.0).raise_for_status()
    except Exception:
        print(f"ERROR: Cannot reach API at {args.url}. Is the stack running?")
        print("  Run: ./scripts/start.sh")
        sys.exit(1)

    run_evaluation(
        csv_path=args.csv,
        base_url=args.url,
        limit=args.limit,
        chunk_type_filter=args.chunk_type,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
