"""
RAG-Specific Evaluation Metrics.

Task-specific metrics tailored to Retrieval-Augmented Generation over
financial documents, plus the top-level orchestrator that combines these
with the traditional NLP metrics (BLEU/METEOR/ROUGE-L/BERTScore).

Metrics defined here:
    1. Source Accuracy      — did the answer cite the correct PDF sources?
    2. Number Match         — do the financial figures match the reference?
    3. Semantic Similarity  — embedding cosine similarity (meaning match)

These are complementary to the traditional overlap metrics:
    - Source accuracy verifies citation correctness (critical for RAG).
    - Number match verifies exact financial values (critical for finance QA).
    - Semantic similarity catches correct answers phrased differently.

Traditional overlap metrics (BLEU, METEOR, ROUGE-L, BERTScore) live in
traditional_metrics.py and are pulled in by compute_all_metrics().
"""

import re

import numpy as np

from app.core.logging import get_logger
from app.evaluation.traditional_metrics import compute_traditional_metrics

logger = get_logger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Source Accuracy — did the answer cite the correct PDF files?
# ══════════════════════════════════════════════════════════════════════════════

# Matches PDF filenames like "2023 Q3 AAPL.pdf" in answer text
_SOURCE_PATTERN = re.compile(r"\d{4}\s+Q\d\s+[A-Z]{2,5}\.pdf", re.IGNORECASE)


def extract_sources(text: str) -> set[str]:
    """
    Extract PDF source filenames referenced in an answer.

    Finds all occurrences of the "{year} Q{n} {TICKER}.pdf" pattern
    used in both generated and reference answers.

    Args:
        text: Answer text to scan for source citations.

    Returns:
        Set of normalized source filenames (lowercased for comparison).
    """
    matches = _SOURCE_PATTERN.findall(text or "")
    return {m.strip().lower() for m in matches}


def source_accuracy(generated: str, reference: str) -> dict:
    """
    Compare cited sources between generated and reference answers.

    Computes precision, recall, and F1 over the set of PDF sources.

    Args:
        generated: The generated answer text.
        reference: The ground-truth answer text.

    Returns:
        Dict with precision, recall, f1, and the source sets.
    """
    gen_sources = extract_sources(generated)
    ref_sources = extract_sources(reference)

    if not ref_sources:
        # No sources in reference — can't score meaningfully
        return {
            "source_precision": None,
            "source_recall": None,
            "source_f1": None,
            "generated_sources": sorted(gen_sources),
            "reference_sources": sorted(ref_sources),
        }

    true_positives = len(gen_sources & ref_sources)

    precision = true_positives / len(gen_sources) if gen_sources else 0.0
    recall = true_positives / len(ref_sources) if ref_sources else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "source_precision": round(precision, 4),
        "source_recall": round(recall, 4),
        "source_f1": round(f1, 4),
        "generated_sources": sorted(gen_sources),
        "reference_sources": sorted(ref_sources),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Number Match — do the financial figures match the reference?
# ══════════════════════════════════════════════════════════════════════════════

# Matches financial figures like "$82,959", "82,959 million", "$81.8 billion"
_NUMBER_PATTERN = re.compile(r"\$?\s?(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)")


def extract_numbers(text: str) -> set[str]:
    """
    Extract numeric financial figures from answer text.

    Normalizes numbers by removing commas and dollar signs so that
    "$82,959" and "82959" are treated as equal.

    Args:
        text: Answer text to scan for numbers.

    Returns:
        Set of normalized numeric strings.
    """
    matches = _NUMBER_PATTERN.findall(text or "")
    normalized = set()
    for m in matches:
        cleaned = m.replace(",", "").strip()
        # Ignore very small numbers (likely list indices, years handled separately)
        try:
            val = float(cleaned)
            # Keep meaningful financial figures (>= 100 to skip percentages/indices)
            if val >= 100:
                normalized.add(cleaned)
        except ValueError:
            continue
    return normalized


def number_match(generated: str, reference: str) -> dict:
    """
    Compare numeric figures between generated and reference answers.

    Computes what fraction of reference numbers appear in the generated
    answer (recall) — the key concern for financial accuracy.

    Args:
        generated: The generated answer text.
        reference: The ground-truth answer text.

    Returns:
        Dict with number recall and the matched/missing number sets.
    """
    gen_numbers = extract_numbers(generated)
    ref_numbers = extract_numbers(reference)

    if not ref_numbers:
        return {
            "number_recall": None,
            "matched_numbers": [],
            "missing_numbers": [],
        }

    matched = gen_numbers & ref_numbers
    missing = ref_numbers - gen_numbers

    recall = len(matched) / len(ref_numbers)

    return {
        "number_recall": round(recall, 4),
        "matched_numbers": sorted(matched),
        "missing_numbers": sorted(missing),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Semantic Similarity — embedding cosine similarity (meaning match)
# ══════════════════════════════════════════════════════════════════════════════

def semantic_similarity(
    generated: str,
    reference: str,
    embedder,
) -> float:
    """
    Compute cosine similarity between answer embeddings.

    Uses the same BGE embedding model as the RAG system to measure
    how semantically close the generated answer is to the reference,
    regardless of exact wording.

    Args:
        generated: The generated answer text.
        reference: The ground-truth answer text.
        embedder: A text embedder with an embed_documents method
                  (e.g., BGETextEmbeddings). Embeddings are normalized,
                  so dot product equals cosine similarity.

    Returns:
        Cosine similarity score in [0, 1] (rounded to 4 decimals).
    """
    embeddings = embedder.embed_documents([generated or "", reference or ""])
    gen_vec = np.array(embeddings[0])
    ref_vec = np.array(embeddings[1])

    # Embeddings are already L2-normalized, so dot product = cosine similarity
    cosine = float(np.dot(gen_vec, ref_vec))

    # Clamp to [0, 1] for readability (cosine can be slightly negative)
    cosine = max(0.0, min(1.0, cosine))

    return round(cosine, 4)


# ══════════════════════════════════════════════════════════════════════════════
# Orchestrator — combines RAG-specific + traditional metrics
# ══════════════════════════════════════════════════════════════════════════════

def compute_all_metrics(
    generated: str,
    reference: str,
    embedder,
    include_traditional: bool = True,
) -> dict:
    """
    Compute the full suite of automatic metrics for one Q&A pair.

    Combines:
        - RAG-specific metrics (source accuracy, number match, semantic similarity)
        - Traditional NLP metrics (BLEU, METEOR, ROUGE-L, BERTScore)

    Args:
        generated: The generated answer text.
        reference: The ground-truth answer text.
        embedder: Text embedder for semantic similarity.
        include_traditional: Whether to include traditional NLP metrics
                             (BLEU, METEOR, ROUGE-L, BERTScore). BERTScore
                             loads a BERT model, so disable for faster runs.

    Returns:
        Combined dict of all metric results.
    """
    results = {}

    # ─── RAG-specific metrics ─────────────────────────────────────────────
    results.update(source_accuracy(generated, reference))
    results.update(number_match(generated, reference))
    results["semantic_similarity"] = semantic_similarity(
        generated, reference, embedder
    )

    # ─── Traditional NLP metrics (BLEU, METEOR, ROUGE-L, BERTScore) ───────
    if include_traditional:
        results.update(compute_traditional_metrics(generated, reference))

    return results
