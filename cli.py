"""
Interactive CLI for the SEC 10-Q RAG system.
Run inside the container:  python cli.py
Run from host (needs deps): python cli.py --url http://localhost:8000
"""
import argparse
import json
import sys
import httpx

GREEN  = "\033[92m"
CYAN   = "\033[96m"
YELLOW = "\033[93m"
BOLD   = "\033[1m"
RESET  = "\033[0m"


def query(base_url: str, question: str, source_filter: list[str] | None = None) -> dict:
    payload = {"question": question, "top_k": 8}
    if source_filter:
        payload["source_filter"] = source_filter
    resp = httpx.post(f"{base_url}/query", json=payload, timeout=180.0)
    resp.raise_for_status()
    return resp.json()


def list_sources(base_url: str) -> list[str]:
    resp = httpx.get(f"{base_url}/sources", timeout=30.0)
    resp.raise_for_status()
    return resp.json()["sources"]


def print_result(result: dict) -> None:
    print(f"\n{BOLD}{GREEN}Answer:{RESET}")
    print(result["answer"])
    print(f"\n{BOLD}{CYAN}Sources used:{RESET}")
    for s in result["sources"]:
        print(f"  • {s}")
    print(f"\n{BOLD}{YELLOW}Retrieved chunks:{RESET}")
    for c in result["chunks"]:
        print(f"  [{c['chunk_type'].upper()}] {c['source']} p{c['page']} "
              f"(score={c['score']}) — {c['text_preview'][:80]}...")
    print()


def main():
    parser = argparse.ArgumentParser(description="SEC 10-Q RAG CLI")
    parser.add_argument("--url", default="http://localhost:8000", help="RAG API base URL")
    parser.add_argument("--sources", nargs="*", help="Filter to specific PDF filenames")
    parser.add_argument("--question", "-q", help="Ask a single question and exit")
    parser.add_argument("--list-sources", action="store_true", help="List indexed PDFs and exit")
    args = parser.parse_args()

    base_url = args.url.rstrip("/")

    # Health check
    try:
        httpx.get(f"{base_url}/health", timeout=5.0).raise_for_status()
    except Exception:
        print(f"ERROR: Cannot reach API at {base_url}. Is the stack running?")
        print("  Run: ./scripts/setup.sh   (first time)")
        print("  Run: docker compose -f docker/docker-compose.yml up -d")
        sys.exit(1)

    # --list-sources
    if args.list_sources:
        sources = list_sources(base_url)
        print(f"\nIndexed PDFs ({len(sources)}):")
        for s in sources:
            print(f"  • {s}")
        return

    # --question (single shot, non-interactive)
    if args.question:
        result = query(base_url, args.question, args.sources)
        print_result(result)
        return

    # Interactive loop
    sources = list_sources(base_url)
    print(f"\n{BOLD}SEC 10-Q RAG — Interactive Mode{RESET}")
    print(f"Indexed documents: {len(sources)}")
    for s in sources:
        print(f"  • {s}")
    print("\nCommands:")
    print("  Type your question and press Enter")
    print("  /sources          — list indexed PDFs")
    print("  /filter <file>    — restrict next query to a specific PDF")
    print("  /filter clear     — remove filter")
    print("  /quit             — exit\n")

    active_filter: list[str] | None = None

    while True:
        try:
            if active_filter:
                prompt = f"{YELLOW}[filter: {', '.join(active_filter)}]{RESET}\n> "
            else:
                prompt = "> "
            user_input = input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue

        if user_input == "/quit":
            print("Bye!")
            break

        if user_input == "/sources":
            sources = list_sources(base_url)
            print(f"\nIndexed PDFs ({len(sources)}):")
            for s in sources:
                print(f"  • {s}")
            print()
            continue

        if user_input.startswith("/filter"):
            parts = user_input.split(maxsplit=1)
            if len(parts) == 1 or parts[1] == "clear":
                active_filter = None
                print("Filter cleared.\n")
            else:
                active_filter = [p.strip() for p in parts[1].split(",")]
                print(f"Filter set to: {active_filter}\n")
            continue

        try:
            result = query(base_url, user_input, active_filter)
            print_result(result)
        except httpx.HTTPStatusError as e:
            print(f"API error: {e.response.status_code} — {e.response.text}\n")
        except Exception as e:
            print(f"Error: {e}\n")


if __name__ == "__main__":
    main()
