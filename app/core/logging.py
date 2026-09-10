"""
Logging Configuration Module.

Sets up structured logging for the application with rich formatting
for terminal output during development.
"""

import logging

from rich.logging import RichHandler


def setup_logging(level=None) -> None:
    """
    Configure application-wide logging with rich formatting.

    The level is read from config (settings.log_level) unless one is passed
    explicitly. Accepts either a string ("INFO", "DEBUG", ...) or a logging
    constant (logging.INFO).

    Args:
        level: Optional override. If None, uses settings.log_level from config.
    """
    # Default to the config-driven level (config controls verbosity, not CLI)
    if level is None:
        from app.core.config import settings
        level = settings.log_level

    # Accept string levels ("INFO") or numeric constants (logging.INFO)
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    # Configure the root logger.
    # force=True removes any handlers a library may have installed during import,
    # so our configuration always takes effect regardless of import order.
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        force=True,
        handlers=[
            RichHandler(
                rich_tracebacks=True,
                tracebacks_show_locals=True,
                show_path=False,
            )
        ],
    )

    # Suppress noisy third-party loggers
    logging.getLogger("qdrant_client").setLevel(logging.WARNING)
    logging.getLogger("fastembed").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    # Google GenAI SDK emits an AFC recommendation on every generate_content
    # call — informational noise, so keep it at ERROR.
    logging.getLogger("google_genai").setLevel(logging.ERROR)
    logging.getLogger("google.genai").setLevel(logging.ERROR)

    # LangChain's built-in debug tracing: prints every LangChain-wrapped step
    # (retrieval, chains, LLM) to the console with full inputs/outputs.
    # Import path is langchain_core.globals on langchain-core 1.x.
    from app.core.config import settings as _settings
    from langchain_core.globals import set_debug
    set_debug(_settings.langchain_debug)


def get_logger(name: str) -> logging.Logger:
    """
    Get a named logger for a module.

    Usage:
        from app.core.logging import get_logger
        logger = get_logger(__name__)
        logger.info("Processing started")

    Args:
        name: Typically __name__ of the calling module.

    Returns:
        Configured logger instance.
    """
    return logging.getLogger(name)
