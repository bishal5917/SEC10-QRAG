"""
Query CLI Script.

Interactive and single-shot query interface for the multimodal RAG system.
Uses the LangGraph workflow to process queries through:
    Retrieve → Grade → Generate

Usage:
    # Interactive mode (type questions, get answers)
    python3 query.py

    # Single question
    python3 query.py -q "Are there any notable changes in Apple's liquidity position or cash flows as reported in these 10-Qs?"

    python3 query.py -q "Examine how Intel's effective tax rate in the most recent 10-Q compares with the tax-related discussions in the notes section?"

"""

import argparse
import sys

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from app.core.config import settings
from app.core.logging import setup_logging
from app.graph.workflow import RAGWorkflow

console = Console()


def interactive_mode(workflow: RAGWorkflow) -> None:
    """
    Run the RAG system in interactive question-answer mode.

    Provides a REPL-style interface where users can ask multiple
    questions in sequence. Supports special commands:
        - 'quit'/'exit': End the session
        - 'stats': Show vector store statistics
        - 'verbose': Toggle detailed retrieval logging

    Args:
        workflow: Initialized RAGWorkflow instance.
    """
    console.print(Panel.fit(
        "[bold]🔍 Multimodal RAG — Query Interface[/bold]\n\n"
        "Ask questions about the ingested earnings reports.\n"
        "Supports text, table, and image-based answers.\n\n"
        "[dim]Commands: 'quit' to exit | 'stats' for store info[/dim]",
        border_style="green",
    ))

    while True:
        console.print()
        try:
            question = console.input("[bold cyan]❓ Question:[/bold cyan] ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye![/dim]")
            break

        if not question:
            continue

        # Handle special commands
        if question.lower() in ("quit", "exit", "q"):
            console.print("[dim]Goodbye![/dim]")
            break

        if question.lower() == "stats":
            stats = workflow.vector_store.get_stats()
            console.print(f"[dim]Vector store: {stats}[/dim]")
            continue

        # Execute the RAG workflow
        console.print("[dim]  Searching and generating...[/dim]")

        answer = workflow.query(question=question)

        console.print(Panel(
            Markdown(answer),
            title="[bold green]📝 Answer[/bold green]",
            border_style="green",
            padding=(1, 2),
        ))


def single_query(
    workflow: RAGWorkflow,
    question: str,
    ticker: str = None,
    no_images: bool = False,
    grade: bool = False,
) -> None:
    """
    Execute a single query and display the answer.

    Args:
        workflow: Initialized RAGWorkflow instance.
        question: The question to answer.
        ticker: Optional company ticker to filter results.
        no_images: If True, skip image context for faster response.
        grade: If True, enable LLM-based document grading.
    """
    # Build metadata filter
    metadata_filter = None
    if ticker:
        metadata_filter = {"ticker": ticker.upper()}

    console.print(f"[dim]  Query: {question}[/dim]")
    if metadata_filter:
        console.print(f"[dim]  Filter: {metadata_filter}[/dim]")

    answer = workflow.query(
        question=question,
        metadata_filter=metadata_filter,
        include_images=not no_images,
        skip_grading=not grade,
    )

    console.print(Panel(
        Markdown(answer),
        title="[bold green]📝 Answer[/bold green]",
        border_style="green",
        padding=(1, 2),
    ))


def main():
    """CLI entry point for querying."""
    parser = argparse.ArgumentParser(
        description="Query the multimodal RAG system"
    )
    parser.add_argument(
        "-q", "--question",
        type=str,
        help="Single question to answer (omit for interactive mode)",
    )
    parser.add_argument(
        "--ticker",
        type=str,
        help="Filter results by company ticker (e.g., AAPL, NVDA)",
    )
    parser.add_argument(
        "--no-images",
        action="store_true",
        help="Skip image context for faster response",
    )
    parser.add_argument(
        "--grade",
        action="store_true",
        help="Enable LLM-based document grading (uses extra API calls, off by default)",
    )
    args = parser.parse_args()

    # Configure logging from config (settings.log_level)
    setup_logging()

    # Validate API key
    if not settings.gemini_api_key:
        console.print(
            "[bold red]Error:[/bold red] GEMINI_API_KEY not set.\n"
            "Get a free key at https://aistudio.google.com/apikey\n"
            "Then add it to your .env file: GEMINI_API_KEY=your_key_here"
        )
        sys.exit(1)

    # Initialize the workflow (loads all models)
    try:
        workflow = RAGWorkflow()
    except Exception as e:
        console.print(f"\n[bold red]Initialization Error:[/bold red] {e}")
        sys.exit(1)

    # Check if vector store has data
    stats = workflow.vector_store.get_stats()
    if sum(stats.values()) == 0:
        console.print(
            "[bold yellow]Warning:[/bold yellow] Vector store is empty. "
            "Run [bold]python ingest.py[/bold] first to process PDFs."
        )
        sys.exit(1)

    # Execute query
    try:
        if args.question:
            single_query(
                workflow,
                question=args.question,
                ticker=args.ticker,
                no_images=args.no_images,
                grade=args.grade,
            )
        else:
            interactive_mode(workflow)

    except KeyboardInterrupt:
        console.print("\n[dim]Interrupted.[/dim]")
    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}")
        sys.exit(1)
    finally:
        # Close the embedded Qdrant client cleanly to avoid shutdown-time noise
        workflow.close()


if __name__ == "__main__":
    main()
