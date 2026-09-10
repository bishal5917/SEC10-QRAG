"""
Evaluation Runner Module.

Orchestrates end-to-end RAG evaluation:
    1. Load ground-truth Q&A pairs from the CSV dataset.
    2. Run each question through the RAG pipeline to get a generated answer.
    3. Compute automatic metrics (semantic sim, ROUGE, sources, numbers).
    4. Optionally run the LLM-as-judge for correctness/completeness scores.
    5. Aggregate results and produce a summary report + detailed CSV.

The runner reuses a single RAGWorkflow instance (models loaded once) and
processes all questions sequentially.
"""

import time
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.table import Table

from app.core.config import settings
from app.core.logging import get_logger
from app.evaluation import metrics
from app.evaluation import retrieval_metrics
from app.evaluation.llm_judge import LLMJudge
from app.graph.workflow import RAGWorkflow

logger = get_logger(__name__)
console = Console()


# Expected CSV columns (after whitespace stripping)
COL_QUESTION = "Question"
COL_SOURCE_DOCS = "Source Docs"
COL_QUESTION_TYPE = "Question Type"
COL_CHUNK_TYPE = "Source Chunk Type"
COL_ANSWER = "Answer"


def load_qna_dataset(csv_path: Path) -> pd.DataFrame:
    """
    Load and clean the ground-truth Q&A dataset.

    The source CSV has heavily space-padded column names and values,
    so this strips whitespace from both headers and string cells.

    Args:
        csv_path: Path to the Q&A CSV file.

    Returns:
        Cleaned DataFrame with normalized column names.
    """
    df = pd.read_csv(csv_path, skipinitialspace=True)

    # Strip whitespace from column names
    df.columns = [c.strip() for c in df.columns]

    # Strip whitespace from string cells
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip()

    logger.info(f"Loaded {len(df)} Q&A pairs from {csv_path.name}")
    return df


class EvaluationRunner:
    """
    Runs the full evaluation loop over a Q&A dataset.

    Usage:
        runner = EvaluationRunner(use_llm_judge=True)
        results_df = runner.run(csv_path, limit=10)
        runner.print_summary(results_df)
    """

    def __init__(self, use_llm_judge: bool = True, include_traditional: bool = True):
        """
        Initialize the evaluation runner.

        Loads the RAG workflow (all models) once and, if enabled,
        the LLM judge.

        Args:
            use_llm_judge: Whether to run LLM-as-judge scoring (uses
                          extra API calls; disable to save quota).
            include_traditional: Whether to compute traditional NLP metrics
                                 (BLEU, METEOR, BERTScore). BERTScore loads
                                 a BERT model on first use.
        """
        console.print("[bold blue]Initializing evaluation harness...[/bold blue]")

        # Load the full RAG pipeline (embedders, vector store, reranker, generator)
        self.workflow = RAGWorkflow()

        # Reuse the workflow's text embedder for semantic similarity metric
        self.embedder = self.workflow.text_embeddings

        self.use_llm_judge = use_llm_judge
        self.include_traditional = include_traditional
        self.judge = LLMJudge() if use_llm_judge else None

        console.print("[bold green]✓ Evaluation harness ready![/bold green]\n")

    def run(self, csv_path: Path, limit: int = None) -> pd.DataFrame:
        """
        Run evaluation over the dataset.

        Args:
            csv_path: Path to the Q&A CSV.
            limit: Optional max number of questions to evaluate
                   (useful for quick tests). None = all questions.

        Returns:
            DataFrame with one row per question containing the question,
            generated answer, reference answer, and all metric scores.
        """
        df = load_qna_dataset(csv_path)

        if limit:
            df = df.head(limit)
            console.print(f"[dim]Evaluating first {limit} questions[/dim]")

        results = []
        total = len(df)

        for idx, row in df.iterrows():
            question = row[COL_QUESTION]
            reference = row[COL_ANSWER]
            source_docs = row.get(COL_SOURCE_DOCS, "")

            console.print(f"\n[cyan]Q{idx + 1}/{total}:[/cyan] {question[:80]}...")

            # ─── Generate answer + capture retrieved docs (1 API call) ────
            try:
                generated, retrieval_result = self.workflow.query_with_trace(
                    question=question
                )
            except Exception as e:
                logger.error(f"Generation failed for Q{idx + 1}: {e}")
                generated = f"[GENERATION ERROR: {e}]"
                retrieval_result = None

            # ─── Compute automatic metrics (local, no API) ───────────────
            row_result = {
                "question": question,
                "question_type": row.get(COL_QUESTION_TYPE, ""),
                "chunk_type": row.get(COL_CHUNK_TYPE, ""),
                "generated_answer": generated,
                "reference_answer": reference,
            }

            auto_metrics = metrics.compute_all_metrics(
                generated,
                reference,
                self.embedder,
                include_traditional=self.include_traditional,
            )
            row_result.update(auto_metrics)

            # ─── Retrieval metrics (Hit Rate, MRR, Negative Rejection) ────
            # Combine text + table docs in ranked order for document-level scoring.
            retrieved_docs = []
            if retrieval_result is not None:
                retrieved_docs = (
                    retrieval_result.text_documents
                    + retrieval_result.table_documents
                )
            retrieval_scores = retrieval_metrics.compute_retrieval_metrics(
                retrieved_docs=retrieved_docs,
                source_docs=source_docs,
                generated=generated,
                reference=reference,
            )
            row_result.update(retrieval_scores)

            # ─── LLM-as-judge (optional, 1 API call) ─────────────────────
            if self.use_llm_judge:
                # Wait between the generation call and the judge call so the
                # two API requests don't breach the per-minute rate limit.
                self._rate_limit_pause(
                    settings.eval_delay_before_judge,
                    reason="before judge call",
                )
                judge_scores = self.judge.evaluate(question, generated, reference)
                row_result.update(judge_scores)

            results.append(row_result)

            # Print quick per-question scores
            self._print_row_scores(row_result)

            # ─── Pause before the next question (skip after the last one) ─
            is_last = idx >= total - 1
            if not is_last:
                # With the judge on there are 2 calls/question, so we need the
                # full spacing. Without it there's only 1 call/question, so a
                # shorter pause is enough to stay under the per-minute limit.
                delay = settings.eval_delay_between_rows
                if not self.use_llm_judge:
                    delay = delay / 2
                self._rate_limit_pause(delay, reason="before next question")

        return pd.DataFrame(results)

    def _rate_limit_pause(self, seconds: float, reason: str = "") -> None:
        """
        Sleep for the given duration to respect API rate limits.

        No-op if seconds <= 0 or if the LLM judge is disabled (in which
        case only generation calls occur and spacing is less critical).

        Args:
            seconds: How long to wait.
            reason: Short description shown in the log line.
        """
        if seconds <= 0:
            return

        console.print(f"[dim]  ⏳ waiting {seconds:.0f}s ({reason})...[/dim]")
        time.sleep(seconds)

    def _print_row_scores(self, row: dict) -> None:
        """
        Print ALL implemented metrics for a single question, grouped by category.

        Every metric the system computes is shown so you can see the full
        per-question picture during an evaluation run.
        """
        def g(key):
            """Fetch a metric value, showing '—' when it's missing/None."""
            val = row.get(key, None)
            return "—" if val is None else val

        # (label, key) for every metric, grouped by category
        groups = [
            ("Answer Quality", [
                ("semantic_similarity", "semantic_similarity"),
                ("bleu", "bleu"),
                ("meteor", "meteor"),
                ("rougeL_f1", "rougeL_f1"),
                ("bertscore_f1", "bertscore_f1"),
            ]),
            ("RAG-Specific", [
                ("source_precision", "source_precision"),
                ("source_recall", "source_recall"),
                ("source_f1", "source_f1"),
                ("number_recall", "number_recall"),
            ]),
            ("Retrieval", [
                ("hit_rate", "hit_rate"),
                ("mrr", "mrr"),
                ("negative_rejection", "negative_rejection"),
            ]),
            ("LLM Judge", [
                ("judge_correctness", "judge_correctness"),
                ("judge_completeness", "judge_completeness"),
                ("judge_faithfulness", "judge_faithfulness"),
                ("judge_overall", "judge_overall"),
            ]),
        ]

        console.print("  [bold]Metrics:[/bold]")
        for group_name, items in groups:
            parts = [f"{label}={g(key)}" for label, key in items]
            console.print(f"    [cyan]{group_name}:[/cyan] [dim]" + "  ".join(parts) + "[/dim]")

    def print_summary(self, results_df: pd.DataFrame) -> None:
        """
        Print an aggregate summary table of all metrics.

        Args:
            results_df: DataFrame returned by run().
        """
        console.print("\n")
        table = Table(title="📊 Evaluation Summary (averages across all questions)")
        table.add_column("Metric", style="cyan")
        table.add_column("Average Score", style="green", justify="right")

        # Numeric metrics to average
        metric_cols = [
            # ─ Answer quality (text overlap + semantic) ─
            ("Semantic Similarity", "semantic_similarity"),
            ("BLEU", "bleu"),
            ("METEOR", "meteor"),
            ("ROUGE-L F1", "rougeL_f1"),
            ("BERTScore F1", "bertscore_f1"),
            # ─ RAG-specific (sources + numbers) ─
            ("Source Precision", "source_precision"),
            ("Source Recall", "source_recall"),
            ("Source F1", "source_f1"),
            ("Number Recall", "number_recall"),
            # ─ Retrieval quality (document-level) ─
            ("Hit Rate", "hit_rate"),
            ("MRR", "mrr"),
            # ─ Negative rejection (only 'no info' questions) ─
            ("Negative Rejection (R-Rate)", "negative_rejection"),
        ]

        if self.use_llm_judge:
            metric_cols.extend([
                ("Judge: Correctness (1-5)", "judge_correctness"),
                ("Judge: Completeness (1-5)", "judge_completeness"),
                ("Judge: Faithfulness (1-5)", "judge_faithfulness"),
                ("Judge: Overall (1-5)", "judge_overall"),
            ])

        for label, col in metric_cols:
            if col in results_df.columns:
                # Average ignoring None/NaN
                avg = results_df[col].dropna().mean()
                if pd.notna(avg):
                    table.add_row(label, f"{avg:.4f}")
                else:
                    table.add_row(label, "—")

        console.print(table)

    def save_results(self, results_df: pd.DataFrame, output_path: Path) -> None:
        """
        Save detailed per-question results to a CSV file.

        Args:
            results_df: DataFrame returned by run().
            output_path: Destination CSV path.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(output_path, index=False)
        console.print(f"\n[green]✓ Detailed results saved to:[/green] {output_path}")
