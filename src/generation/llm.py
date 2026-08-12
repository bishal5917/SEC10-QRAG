import time
import httpx
from typing import List, Dict, Any

from src.config import OLLAMA_BASE_URL, LLM_MODEL, VISION_MODEL
from src.logger import get_logger

log = get_logger("llm")

_TEXT_PROMPT = """\
You are a helpful assistant. Answer the question using ONLY the information provided in the context below.

Rules:
- For TABLE context: tables are formatted as markdown. Column headers are in the
  first row. Bold text above a table (e.g. **Q3 2022  |  Q3 2021**) are floating
  period headers — use them to identify which quarter each column belongs to.
  Always match row labels to their correct column before quoting any figure.
- For FIGURE context: data comes from a chart spatially reconstructed as text.
  Words on the same line belong to the same chart column. Read section headers
  and period labels to correctly attribute each value.
- Quote exact figures from the context. Do not round or estimate.
- Cite source document(s) at the end: SOURCE(S): <filename>, ...
- If the context does not contain enough information, say so clearly; Otherwise answer the question concisely, with no fluff.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

_VISION_PROMPT = """\
You are a helpful assistant. The image shows a chart or diagram from a document \
({source}, page {page}).

Describe what the chart shows, then answer this question using exact figures \
from the chart:

QUESTION: {question}

Rules:
- Quote exact values visible in the chart (axis labels, bar values, legends).
- Identify the time periods shown if present.
- Do not estimate or round figures.
- End with: SOURCE(S): {source}, page {page}
"""


def _ollama_post(payload: dict, timeout: float = 180.0) -> str:
    t0 = time.perf_counter()
    try:
        resp = httpx.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json=payload,
            timeout=timeout,
        )
        resp.raise_for_status()
    except httpx.TimeoutException:
        log.error("Ollama request timed out")
        raise
    except httpx.HTTPStatusError as e:
        log.error(f"Ollama HTTP error: {e.response.status_code} — {e.response.text[:200]}")
        raise
    elapsed = (time.perf_counter() - t0) * 1000
    answer = resp.json()["response"].strip()
    log.info(f"Ollama response in {elapsed:.1f}ms | answer_chars={len(answer)}")
    return answer


def _format_text_context(chunks: List[Dict[str, Any]]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, 1):
        header = f"[{i}] SOURCE: {chunk['source']} | Page {chunk['page']} | Type: {chunk['chunk_type'].upper()}"
        parts.append(f"{header}\n{chunk['text']}")
    return "\n\n---\n\n".join(parts)


def generate_answer(question: str, chunks: List[Dict[str, Any]]) -> str:
    # Split chunks into figure (has image) vs text/table
    figure_chunks = [c for c in chunks if c.get("image_b64")]
    text_chunks   = [c for c in chunks if not c.get("image_b64")]

    answers = []

    # ── Text + table chunks → llama3:instruct ────────────────────────────────
    if text_chunks:
        context = _format_text_context(text_chunks)
        prompt = _TEXT_PROMPT.format(context=context, question=question)
        log.info(f"Text/table call | model={LLM_MODEL} | chunks={len(text_chunks)} | prompt_chars={len(prompt)}")
        answers.append(_ollama_post({"model": LLM_MODEL, "prompt": prompt, "stream": False}))

    # ── Figure chunks → llava (one call per figure) ───────────────────────────
    for chunk in figure_chunks:
        prompt = _VISION_PROMPT.format(
            source=chunk["source"],
            page=chunk["page"],
            question=question,
        )
        log.info(f"Vision call | model={VISION_MODEL} | source={chunk['source']} p{chunk['page']}")
        answers.append(_ollama_post({
            "model": VISION_MODEL,
            "prompt": prompt,
            "images": [chunk["image_b64"]],
            "stream": False,
        }))

    return "\n\n---\n\n".join(answers) if answers else "No relevant context found."
