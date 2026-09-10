"""
Evaluation Script.

Runs the RAG system against a ground-truth Q&A dataset and reports
quality metrics (semantic similarity, BLEU, METEOR, ROUGE-L, BERTScore,
source accuracy, number recall, and optional LLM-as-judge scores).

All behavior is controlled via `app/core/config.py` — no command-line
arguments. Edit the `eval_*` settings there to change the dataset,
question limit, whether to use the LLM judge, traditional metrics, output
path, and rate-limit delays.

Usage:
    python3 evaluate.py

Relevant config settings (in app/core/config.py):
    eval_dataset             - path to the Q&A CSV
    eval_limit               - max questions to evaluate (None = all)
    eval_use_llm_judge       - enable/disable LLM-as-judge
    eval_include_traditional - enable/disable BLEU/METEOR/ROUGE-L/BERTScore
    eval_output_path         - where to save detailed results
    eval_delay_before_judge  - seconds between generation and judge calls
    eval_delay_between_rows  - seconds between consecutive questions
"""

import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

from app.core.config import settings
from app.core.logging import get_logger, setup_logging
from app.evaluation.runner import EvaluationRunner

console = Console()
logger = get_logger(__name__)


def _resolve_path(path_str: str) -> Path:
    """Resolve a config path relative to the project root if not absolute."""
    path = Path(path_str)
    if not path.is_absolute():
        path = settings.project_root / path
    return path


def main():
    setup_logging()  # level read from config (settings.log_level)

    # Validate API key
    if not settings.gemini_api_key:
        console.print(
            "[bold red]Error:[/bold red] GEMINI_API_KEY not set. "
            "Add it to your .env file."
        )
        sys.exit(1)

    # Resolve dataset path from config
    dataset_path = _resolve_path(settings.eval_dataset)
    if not dataset_path.exists():
        console.print(f"[bold red]Error:[/bold red] Dataset not found: {dataset_path}")
        sys.exit(1)

    console.print(Panel.fit(
        "[bold]📊 RAG Evaluation[/bold]\n"
        f"Dataset: {dataset_path.name}\n"
        f"Limit: {settings.eval_limit or 'all questions'}\n"
        f"LLM Judge: {'enabled' if settings.eval_use_llm_judge else 'disabled'}\n"
        f"Traditional metrics: {'enabled' if settings.eval_include_traditional else 'disabled'}",
        border_style="blue",
    ))

    runner = None
    try:
        runner = EvaluationRunner(
            use_llm_judge=settings.eval_use_llm_judge,
            include_traditional=settings.eval_include_traditional,
        )
        results_df = runner.run(dataset_path, limit=settings.eval_limit)

        # Print summary and save detailed results
        runner.print_summary(results_df)
        runner.save_results(results_df, _resolve_path(settings.eval_output_path))

    except KeyboardInterrupt:
        console.print("\n[dim]Evaluation interrupted.[/dim]")
    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}")
        logger.exception("Evaluation failed")
        sys.exit(1)
    finally:
        # Close the embedded Qdrant client cleanly to avoid shutdown-time noise
        if runner is not None:
            runner.workflow.close()


if __name__ == "__main__":
    main()
