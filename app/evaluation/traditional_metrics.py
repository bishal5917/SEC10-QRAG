"""
Traditional NLP Evaluation Metrics.

Classic reference-based text-generation metrics that compare a generated
answer against a gold-standard reference by measuring word/token overlap
or embedding similarity. All metrics run locally (no API calls).

Metrics implemented:
    1. BLEU      — n-gram precision overlap (from machine translation)
    2. METEOR    — overlap with stemming + synonym matching (smarter than BLEU)
    3. ROUGE-L   — longest common subsequence overlap (from summarization)
    4. BERTScore — token-level semantic similarity using BERT embeddings

Characteristics for financial QA:
    - BLEU: strict n-gram precision. Tends to score low on long/varied answers
            but provides a useful lower-bound baseline.
    - METEOR: accounts for stems and synonyms, more forgiving than BLEU.
    - ROUGE-L: rewards preserving the reference's word order at the sentence level.
    - BERTScore: embedding-based, captures meaning even with different wording;
            the most robust of the four for varied phrasing.
"""

import warnings

from rouge_score import rouge_scorer

from app.core.logging import get_logger

logger = get_logger(__name__)


# ─── NLTK setup (used by BLEU + METEOR) ───────────────────────────────────────

def _ensure_nltk_data() -> None:
    """
    Download required NLTK data packages if not already present.

    METEOR needs 'wordnet' (synonyms) and 'punkt' (tokenization).
    Downloads are cached locally by NLTK after the first run.
    """
    import nltk

    required = [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4"),
    ]

    for resource_path, package in required:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            logger.info(f"Downloading NLTK package: {package}")
            nltk.download(package, quiet=True)


# ─── BLEU ─────────────────────────────────────────────────────────────────────

def bleu_score(generated: str, reference: str) -> dict:
    """
    Compute BLEU score between generated and reference answers.

    BLEU measures n-gram precision: how many n-grams in the generated
    text appear in the reference. Uses smoothing to handle short texts
    and cases with no higher-order n-gram matches.

    Args:
        generated: The generated answer text.
        reference: The ground-truth reference answer.

    Returns:
        Dict with the BLEU score (0-1).
    """
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

    _ensure_nltk_data()

    # Tokenize into words (simple whitespace + lowercasing)
    gen_tokens = (generated or "").lower().split()
    ref_tokens = (reference or "").lower().split()

    if not gen_tokens or not ref_tokens:
        return {"bleu": 0.0}

    # Smoothing avoids zero scores when higher-order n-grams don't match
    smoothing = SmoothingFunction().method1

    score = sentence_bleu(
        [ref_tokens],  # list of reference token lists
        gen_tokens,
        smoothing_function=smoothing,
    )

    return {"bleu": round(float(score), 4)}


# ─── METEOR ───────────────────────────────────────────────────────────────────

def meteor_score_metric(generated: str, reference: str) -> dict:
    """
    Compute METEOR score between generated and reference answers.

    METEOR improves on BLEU by matching word stems and synonyms
    (via WordNet), making it more tolerant of valid rephrasing.

    Args:
        generated: The generated answer text.
        reference: The ground-truth reference answer.

    Returns:
        Dict with the METEOR score (0-1).
    """
    from nltk.translate.meteor_score import meteor_score

    _ensure_nltk_data()

    gen_tokens = (generated or "").lower().split()
    ref_tokens = (reference or "").lower().split()

    if not gen_tokens or not ref_tokens:
        return {"meteor": 0.0}

    score = meteor_score([ref_tokens], gen_tokens)

    return {"meteor": round(float(score), 4)}


# ─── ROUGE-L ──────────────────────────────────────────────────────────────────

# Reuse a single scorer instance (stemmer initialization is not free)
_rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


def rouge_l(generated: str, reference: str) -> dict:
    """
    Compute ROUGE-L score between generated and reference answers.

    ROUGE-L measures the longest common subsequence, capturing
    fluency and content overlap at the sentence level.

    Args:
        generated: The generated answer text.
        reference: The ground-truth reference answer.

    Returns:
        Dict with ROUGE-L precision, recall, and F1.
    """
    scores = _rouge.score(reference or "", generated or "")
    rl = scores["rougeL"]
    return {
        "rougeL_precision": round(rl.precision, 4),
        "rougeL_recall": round(rl.recall, 4),
        "rougeL_f1": round(rl.fmeasure, 4),
    }


# ─── BERTScore ────────────────────────────────────────────────────────────────

def bert_score_metric(generated: str, reference: str) -> dict:
    """
    Compute BERTScore between generated and reference answers.

    BERTScore uses contextual BERT embeddings to measure token-level
    semantic similarity, rather than exact string overlap. It's robust
    to paraphrasing and word choice differences.

    Args:
        generated: The generated answer text.
        reference: The ground-truth reference answer.

    Returns:
        Dict with BERTScore precision, recall, and F1 (each 0-1).
    """
    if not generated or not reference:
        return {
            "bertscore_precision": 0.0,
            "bertscore_recall": 0.0,
            "bertscore_f1": 0.0,
        }

    try:
        from bert_score import score as bert_score_fn

        # Suppress verbose transformer warnings during scoring
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            precision, recall, f1 = bert_score_fn(
                [generated],
                [reference],
                lang="en",
                verbose=False,
                rescale_with_baseline=False,
            )

        return {
            "bertscore_precision": round(float(precision[0]), 4),
            "bertscore_recall": round(float(recall[0]), 4),
            "bertscore_f1": round(float(f1[0]), 4),
        }

    except Exception as e:
        logger.warning(f"BERTScore failed: {e}")
        return {
            "bertscore_precision": None,
            "bertscore_recall": None,
            "bertscore_f1": None,
        }


# ─── Orchestrator ─────────────────────────────────────────────────────────────

def compute_traditional_metrics(
    generated: str,
    reference: str,
    include_bertscore: bool = True,
) -> dict:
    """
    Compute all traditional NLP metrics for one Q&A pair.

    Args:
        generated: The generated answer text.
        reference: The ground-truth reference answer.
        include_bertscore: Whether to compute BERTScore. BERTScore loads a
                           BERT model on first use, so disable for faster runs.

    Returns:
        Combined dict with BLEU, METEOR, ROUGE-L (and BERTScore) results.
    """
    results = {}
    results.update(bleu_score(generated, reference))
    results.update(meteor_score_metric(generated, reference))
    results.update(rouge_l(generated, reference))

    if include_bertscore:
        results.update(bert_score_metric(generated, reference))

    return results
