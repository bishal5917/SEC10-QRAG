"""
Observability Module — Langfuse integration (self-hosted v3, SDK v4 API).

Provides:
    - get_langfuse_handler(): a LangChain CallbackHandler when enabled, else None
    - get_callbacks():        [handler] or [] for LangGraph .invoke(config=...)
    - observe():              a decorator to trace direct (non-LangChain) calls

How it fits the pipeline:
    The pipeline is LangChain/LangGraph-native, so passing the handler into the
    graph invocation captures the full trace tree (retrieve → rerank → prompt →
    LLM → response) with timing and token/cost automatically.

Safe by default:
    If enable_langfuse is False, keys are missing, or langfuse isn't installed,
    everything degrades to no-ops and the app runs normally. Observability is
    purely additive.

SDK v4 specifics:
    - Initialize a global Langfuse client once (with keys/host).
    - CallbackHandler() takes NO keys — it uses that global client.
"""

from functools import lru_cache

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def _init_client():
    """
    Initialize the global Langfuse client once (idempotent, cached).

    Returns:
        True if a client was initialized, False otherwise.
    """
    if not settings.enable_langfuse:
        return False

    if not (settings.langfuse_public_key and settings.langfuse_secret_key):
        logger.warning(
            "enable_langfuse=True but LANGFUSE_PUBLIC_KEY/SECRET_KEY are missing; "
            "skipping Langfuse tracing."
        )
        return False

    try:
        from langfuse import Langfuse

        # Initializing the client registers it globally; CallbackHandler and the
        # observe decorator both use this global instance.
        Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
        )
        logger.info(f"Langfuse tracing enabled → {settings.langfuse_host}")
        return True

    except Exception as e:
        logger.warning(f"Could not initialize Langfuse client, tracing disabled: {e}")
        return False


@lru_cache(maxsize=1)
def get_langfuse_handler():
    """
    Build a Langfuse LangChain CallbackHandler, or return None if disabled.

    Returns:
        A langfuse.langchain.CallbackHandler when enabled + configured, else None.
    """
    if not _init_client():
        return None

    try:
        from langfuse.langchain import CallbackHandler

        return CallbackHandler()  # uses the global client initialized above
    except Exception as e:
        logger.warning(f"Could not create Langfuse CallbackHandler: {e}")
        return None


def get_callbacks() -> list:
    """
    Return a callbacks list for LangChain/LangGraph invoke(config={...}).

    Returns:
        [handler] if Langfuse is active, else [] (safe to always spread).
    """
    handler = get_langfuse_handler()
    return [handler] if handler is not None else []


def observe(name: str = None):
    """
    Decorator to trace a plain (non-LangChain) function in Langfuse.

    Used for the direct-Gemini calls (query optimizer, LLM judge) that bypass
    LangChain and so aren't captured by the callback handler. When Langfuse is
    disabled or unavailable, this is a transparent no-op.

    Args:
        name: Optional span name shown in Langfuse.
    """
    def decorator(func):
        if not settings.enable_langfuse:
            return func
        try:
            _init_client()
            from langfuse import observe as _lf_observe
            return _lf_observe(name=name)(func) if name else _lf_observe(func)
        except Exception:
            return func

    return decorator
