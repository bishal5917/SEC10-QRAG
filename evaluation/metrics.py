"""
RAG Evaluation Metrics
======================

Covers all metrics from the standard RAG evaluation framework:

RETRIEVAL LEVEL
  - Context Precision   : of retrieved chunks, how many are actually relevant?
  - Context Recall      : of all relevant info, how much did we retrieve?
  - Context F1          : harmonic mean of precision and recall
  - MRR                 : Mean Reciprocal Rank — how high is the first relevant chunk?
  - nDCG                : normalized Discounted Cumulative Gain — ranks relevant chunks higher

GENERATION LEVEL
  - BLEU                : n-gram precision vs reference (brevity-penalized)
  - ROUGE-1/2/L         : n-gram and subsequence recall vs reference
  - METEOR              : ROUGE + synonym matching + word order
  - BERTScore           : semantic similarity via token embeddings (needs bert-score pkg)

END-TO-END
  - Faithfulness        : fraction of answer sentences grounded in retrieved context
  - Hallucination Rate  : 1 - faithfulness (fraction of sentences NOT grounded)
  - Factual Consistency : exact financial figures in answer that match reference
  - Answer Relevance    : answer addresses the question keywords
  - Exact Number Match  : critical for financial QA — correct figures cited
"""

import re
import math
from collections import Counter
from typing import List, Dict, Any, Optional


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _tokenize(text: str) -> List[str]:
    return re.findall(r'\b\w+\b', text.lower())


def _ngrams(tokens: List[str], n: int) -> Counter:
    return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))


def _extract_numbers(text: str) -> List[str]:
    """Extract financial figures: $82,959  117,154  34.2%  $11.7B etc."""
    return re.findall(r'\$?[\d,]+\.?\d*\s*(?:billion|million|[BbMm%])?', text)


_STOPWORDS = {
    'the','a','an','is','are','was','were','in','of','to','and','or','for',
    'with','that','this','it','its','be','has','have','had','from','by','as',
    'at','on','not','but','which','their','they','we','our','been','will',
    'would','could','should','may','might','do','did','does','if','so','than',
}


# ─────────────────────────────────────────────────────────────
# RETRIEVAL METRICS
# ─────────────────────────────────────────────────────────────

def context_precision(
    retrieved_chunks: List[Dict[str, Any]],
    reference_answer: str,
) -> float:
    """
    Of the retrieved chunks, what fraction are actually relevant?
    A chunk is relevant if it shares meaningful token overlap with the reference answer.
    Range: 0.0 – 1.0  (higher = less noise in retrieval)
    """
    if not retrieved_chunks:
        return 0.0
    ref_tokens = set(_tokenize(reference_answer)) - _STOPWORDS
    relevant = 0
    for chunk in retrieved_chunks:
        chunk_tokens = set(_tokenize(chunk["text"])) - _STOPWORDS
        if not chunk_tokens:
            continue
        overlap = len(ref_tokens & chunk_tokens) / len(ref_tokens | chunk_tokens)
        if overlap >= 0.05:  # Jaccard threshold
            relevant += 1
    return round(relevant / len(retrieved_chunks), 4)


def context_recall(
    reference_answer: str,
    retrieved_chunks: List[Dict[str, Any]],
) -> float:
    """
    What fraction of key terms in the reference answer appear in retrieved context?
    Range: 0.0 – 1.0  (higher = retriever found the right chunks)
    """
    ref_tokens = set(_tokenize(reference_answer)) - _STOPWORDS
    if not ref_tokens:
        return 0.0
    context_tokens = set(_tokenize(" ".join(c["text"] for c in retrieved_chunks)))
    return round(len(ref_tokens & context_tokens) / len(ref_tokens), 4)


def context_f1(
    retrieved_chunks: List[Dict[str, Any]],
    reference_answer: str,
) -> float:
    """
    Harmonic mean of context precision and recall.
    Range: 0.0 – 1.0
    """
    p = context_precision(retrieved_chunks, reference_answer)
    r = context_recall(reference_answer, retrieved_chunks)
    if p + r == 0:
        return 0.0
    return round(2 * p * r / (p + r), 4)


def mrr(
    retrieved_chunks: List[Dict[str, Any]],
    reference_answer: str,
) -> float:
    """
    Mean Reciprocal Rank — rewards systems that place the most relevant
    chunk at the top of the ranked list.
    Score = 1/rank of first relevant chunk. If none relevant, score = 0.
    Range: 0.0 – 1.0  (1.0 = most relevant chunk is rank 1)
    """
    ref_tokens = set(_tokenize(reference_answer)) - _STOPWORDS
    for rank, chunk in enumerate(retrieved_chunks, 1):
        chunk_tokens = set(_tokenize(chunk["text"])) - _STOPWORDS
        if not chunk_tokens:
            continue
        overlap = len(ref_tokens & chunk_tokens) / max(len(ref_tokens), 1)
        if overlap >= 0.1:
            return round(1.0 / rank, 4)
    return 0.0


def ndcg(
    retrieved_chunks: List[Dict[str, Any]],
    reference_answer: str,
) -> float:
    """
    Normalized Discounted Cumulative Gain — measures ranking quality.
    Relevant chunks ranked higher contribute more to the score.
    Range: 0.0 – 1.0  (1.0 = perfect ranking)
    """
    ref_tokens = set(_tokenize(reference_answer)) - _STOPWORDS

    def relevance(chunk: Dict[str, Any]) -> float:
        chunk_tokens = set(_tokenize(chunk["text"])) - _STOPWORDS
        if not chunk_tokens or not ref_tokens:
            return 0.0
        return len(ref_tokens & chunk_tokens) / len(ref_tokens | chunk_tokens)

    scores = [relevance(c) for c in retrieved_chunks]

    # DCG
    dcg = sum(s / math.log2(i + 2) for i, s in enumerate(scores))

    # Ideal DCG (sorted descending)
    ideal = sorted(scores, reverse=True)
    idcg = sum(s / math.log2(i + 2) for i, s in enumerate(ideal))

    if idcg == 0:
        return 0.0
    return round(dcg / idcg, 4)


# ─────────────────────────────────────────────────────────────
# GENERATION METRICS
# ─────────────────────────────────────────────────────────────

def bleu(generated_answer: str, reference_answer: str, max_n: int = 4) -> float:
    """
    BLEU score: n-gram precision with brevity penalty.
    Measures how much of the generated text appears in the reference.
    Range: 0.0 – 1.0  (higher = more n-gram overlap with reference)
    """
    gen_tokens = _tokenize(generated_answer)
    ref_tokens = _tokenize(reference_answer)

    if not gen_tokens:
        return 0.0

    # Brevity penalty
    bp = 1.0 if len(gen_tokens) >= len(ref_tokens) else math.exp(1 - len(ref_tokens) / len(gen_tokens))

    precisions = []
    for n in range(1, max_n + 1):
        gen_ng = _ngrams(gen_tokens, n)
        ref_ng = _ngrams(ref_tokens, n)
        if not gen_ng:
            precisions.append(0.0)
            continue
        clipped = sum(min(count, ref_ng[gram]) for gram, count in gen_ng.items())
        precisions.append(clipped / sum(gen_ng.values()))

    if any(p == 0 for p in precisions):
        return 0.0

    log_avg = sum(math.log(p) for p in precisions) / max_n
    return round(bp * math.exp(log_avg), 4)


def rouge(generated_answer: str, reference_answer: str) -> Dict[str, float]:
    """
    ROUGE-1, ROUGE-2, ROUGE-L scores.
    ROUGE-1/2: unigram/bigram recall vs reference.
    ROUGE-L:   longest common subsequence F1.
    Range: 0.0 – 1.0 each
    """
    from rouge_score import rouge_scorer as rs
    scorer = rs.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference_answer, generated_answer)
    return {
        "rouge_1": round(scores['rouge1'].fmeasure, 4),
        "rouge_2": round(scores['rouge2'].fmeasure, 4),
        "rouge_l": round(scores['rougeL'].fmeasure, 4),
    }


def meteor(generated_answer: str, reference_answer: str) -> float:
    """
    METEOR: harmonic mean of unigram precision and recall with
    a fragmentation penalty for word order.
    Better than BLEU for short answers — handles partial matches.
    Range: 0.0 – 1.0

    Note: full METEOR uses WordNet synonyms (needs NLTK).
    This is a lightweight approximation using token overlap + stemming.
    """
    gen_tokens = _tokenize(generated_answer)
    ref_tokens = _tokenize(reference_answer)

    if not gen_tokens or not ref_tokens:
        return 0.0

    # Simple stem: strip common suffixes
    def stem(w: str) -> str:
        for suffix in ('ing', 'tion', 'ed', 'ly', 'er', 'est', 's'):
            if w.endswith(suffix) and len(w) - len(suffix) > 2:
                return w[:-len(suffix)]
        return w

    gen_stemmed = [stem(t) for t in gen_tokens]
    ref_stemmed = [stem(t) for t in ref_tokens]

    ref_counter = Counter(ref_stemmed)
    matches = 0
    for t in gen_stemmed:
        if ref_counter.get(t, 0) > 0:
            matches += 1
            ref_counter[t] -= 1

    if matches == 0:
        return 0.0

    precision = matches / len(gen_stemmed)
    recall = matches / len(ref_stemmed)
    f_mean = (10 * precision * recall) / (9 * precision + recall)

    # Fragmentation penalty (simplified)
    penalty = 0.5 * (matches / max(len(gen_stemmed), 1)) ** 3
    score = f_mean * (1 - penalty)
    return round(score, 4)


def bertscore(generated_answer: str, reference_answer: str) -> float:
    """
    BERTScore: semantic similarity using contextual token embeddings.
    Captures meaning even when exact words differ.
    Range: 0.0 – 1.0

    Requires: pip install bert-score
    Falls back to ROUGE-L if bert-score is not installed.
    """
    try:
        from bert_score import score as bert_score_fn
        P, R, F1 = bert_score_fn(
            [generated_answer], [reference_answer],
            lang="en", verbose=False,
        )
        return round(F1[0].item(), 4)
    except ImportError:
        # Graceful fallback — log and use ROUGE-L as proxy
        rouge_scores = rouge(generated_answer, reference_answer)
        return rouge_scores["rouge_l"]


# ─────────────────────────────────────────────────────────────
# END-TO-END METRICS
# ─────────────────────────────────────────────────────────────

def faithfulness(
    generated_answer: str,
    retrieved_chunks: List[Dict[str, Any]],
) -> float:
    """
    What fraction of answer sentences are grounded in the retrieved context?
    Range: 0.0 – 1.0  (1.0 = fully grounded, no hallucination)
    """
    context_tokens = set(_tokenize(" ".join(c["text"] for c in retrieved_chunks)))
    sentences = [s.strip() for s in re.split(r'[.!?\n]', generated_answer) if len(s.strip()) > 20]
    if not sentences:
        return 1.0
    grounded = 0
    for sent in sentences:
        sent_tokens = set(_tokenize(sent)) - _STOPWORDS
        if not sent_tokens:
            continue
        if len(sent_tokens & context_tokens) / len(sent_tokens) >= 0.5:
            grounded += 1
    return round(grounded / len(sentences), 4)


def hallucination_rate(
    generated_answer: str,
    retrieved_chunks: List[Dict[str, Any]],
) -> float:
    """
    Fraction of answer sentences NOT grounded in retrieved context.
    = 1 - faithfulness.
    Range: 0.0 – 1.0  (0.0 = no hallucination)
    """
    return round(1.0 - faithfulness(generated_answer, retrieved_chunks), 4)


def factual_consistency(
    generated_answer: str,
    reference_answer: str,
    retrieved_chunks: List[Dict[str, Any]],
) -> float:
    """
    Combined score: answer is both faithful to context AND matches reference.
    Harmonic mean of faithfulness and exact_number_match.
    Range: 0.0 – 1.0
    """
    f = faithfulness(generated_answer, retrieved_chunks)
    n = exact_number_match(generated_answer, reference_answer)
    if f + n == 0:
        return 0.0
    return round(2 * f * n / (f + n), 4)


def answer_relevance(question: str, generated_answer: str) -> float:
    """
    Does the answer address the question?
    Range: 0.0 – 1.0
    """
    q_tokens = set(_tokenize(question)) - _STOPWORDS - {'what','how','when','where','why','which','who'}
    if not q_tokens:
        return 0.0
    a_tokens = set(_tokenize(generated_answer))
    return round(len(q_tokens & a_tokens) / len(q_tokens), 4)


def exact_number_match(generated_answer: str, reference_answer: str) -> float:
    """
    For financial QA: what fraction of exact figures in the reference
    appear in the generated answer?
    Range: 0.0 – 1.0  (most critical metric for this system)
    """
    ref_numbers = _extract_numbers(reference_answer)
    if not ref_numbers:
        return 1.0
    gen_numbers = _extract_numbers(generated_answer)
    matched = sum(1 for n in ref_numbers if any(n in g or g in n for g in gen_numbers))
    return round(matched / len(ref_numbers), 4)


# ─────────────────────────────────────────────────────────────
# Master compute function
# ─────────────────────────────────────────────────────────────

def compute_all(
    question: str,
    generated_answer: str,
    reference_answer: str,
    retrieved_chunks: List[Dict[str, Any]],
) -> Dict[str, float]:
    """
    Compute all metrics for a single QnA pair.
    Returns a flat dict of metric_name -> score.
    """
    rouge_scores = rouge(generated_answer, reference_answer)
    return {
        # Retrieval
        "context_precision":   context_precision(retrieved_chunks, reference_answer),
        "context_recall":      context_recall(reference_answer, retrieved_chunks),
        "context_f1":          context_f1(retrieved_chunks, reference_answer),
        "mrr":                 mrr(retrieved_chunks, reference_answer),
        "ndcg":                ndcg(retrieved_chunks, reference_answer),
        # Generation
        "bleu":                bleu(generated_answer, reference_answer),
        "rouge_1":             rouge_scores["rouge_1"],
        "rouge_2":             rouge_scores["rouge_2"],
        "rouge_l":             rouge_scores["rouge_l"],
        "meteor":              meteor(generated_answer, reference_answer),
        "bertscore":           bertscore(generated_answer, reference_answer),
        # End-to-end
        "faithfulness":        faithfulness(generated_answer, retrieved_chunks),
        "hallucination_rate":  hallucination_rate(generated_answer, retrieved_chunks),
        "factual_consistency": factual_consistency(generated_answer, reference_answer, retrieved_chunks),
        "answer_relevance":    answer_relevance(question, generated_answer),
        "exact_number_match":  exact_number_match(generated_answer, reference_answer),
    }
