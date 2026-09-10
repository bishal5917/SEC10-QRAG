"""
LLM-as-Judge Evaluation Module.

Uses Gemini to score generated answers against reference answers on
dimensions that automatic metrics (ROUGE, embeddings) cannot fully
capture — factual correctness, completeness, and faithfulness.

Why LLM-as-judge?
    Automatic metrics have blind spots:
        - ROUGE penalizes correct answers with different wording.
        - Embedding similarity can be high even when a key number is wrong.
    An LLM judge reads both answers and evaluates them the way a human
    grader would — checking if the facts, numbers, and conclusions align.

The judge returns structured scores (1-5) with a short justification,
which are parsed into numeric values for aggregation.
"""

import json
import re
import time

from google import genai
from google.genai import types

from app.core.config import settings
from app.core.logging import get_logger
from app.core.observability import observe

logger = get_logger(__name__)


# ─── Judge Prompt ─────────────────────────────────────────────────────────────

JUDGE_PROMPT = """You are an expert evaluator grading an AI-generated answer against a reference (ground-truth) answer for a financial question.

Evaluate the generated answer on three dimensions, each scored 1 to 5:

1. CORRECTNESS (1-5): Are the facts and numbers in the generated answer accurate compared to the reference? (5 = all facts/numbers match, 1 = mostly wrong)
2. COMPLETENESS (1-5): Does the generated answer cover all the key points in the reference? (5 = covers everything, 1 = misses most)
3. FAITHFULNESS (1-5): Does the generated answer avoid adding fabricated or unsupported claims? (5 = fully grounded, 1 = much hallucination)

QUESTION:
{question}

REFERENCE ANSWER (ground truth):
{reference}

GENERATED ANSWER (to evaluate):
{generated}

Respond with ONLY a valid JSON object in this exact format (no markdown, no extra text):
{{"correctness": <int>, "completeness": <int>, "faithfulness": <int>, "justification": "<one sentence explanation>"}}"""


class LLMJudge:
    """
    Gemini-based judge that scores answer quality on multiple dimensions.

    Usage:
        judge = LLMJudge()
        scores = judge.evaluate(question, generated_answer, reference_answer)
        # scores → {"correctness": 5, "completeness": 4, "faithfulness": 5, ...}
    """

    def __init__(self, rate_limit_delay: float = 2.0):
        """
        Initialize the Gemini judge client.

        Args:
            rate_limit_delay: Seconds to wait between judge calls to
                              respect the free-tier rate limit.
        """
        if not settings.gemini_api_key:
            raise ValueError("GEMINI_API_KEY required for LLM-as-judge evaluation")

        self.client = genai.Client(api_key=settings.gemini_api_key)
        self.model = settings.gemini_model
        self.rate_limit_delay = rate_limit_delay

    @observe(name="llm_judge")
    def evaluate(
        self,
        question: str,
        generated: str,
        reference: str,
    ) -> dict:
        """
        Score a generated answer against the reference using Gemini.

        Args:
            question: The original question.
            generated: The generated answer to evaluate.
            reference: The ground-truth reference answer.

        Returns:
            Dict with correctness, completeness, faithfulness (1-5 ints),
            an overall average, and a justification string.
            Returns None scores if the judge call fails.
        """
        prompt = JUDGE_PROMPT.format(
            question=question,
            reference=reference,
            generated=generated,
        )

        try:
            # Respect rate limits on free tier
            time.sleep(self.rate_limit_delay)

            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.0,  # Deterministic grading
                    # Headroom for the model's reasoning tokens + the JSON output.
                    # 256 was too small for 3.x models and truncated the JSON.
                    max_output_tokens=1024,
                ),
            )

            return self._parse_response(response.text)

        except Exception as e:
            logger.warning(f"LLM judge failed: {e}")
            return {
                "judge_correctness": None,
                "judge_completeness": None,
                "judge_faithfulness": None,
                "judge_overall": None,
                "judge_justification": f"Judge error: {e}",
            }

    def _parse_response(self, text: str) -> dict:
        """
        Parse the judge's JSON response into structured scores.

        Handles cases where the model wraps JSON in markdown code fences
        or adds surrounding text.

        Args:
            text: Raw response text from Gemini.

        Returns:
            Parsed scores dict with an overall average.
        """
        if not text:
            return self._empty_scores("Empty judge response")

        # Extract JSON object from the response (handles code fences/extra text)
        json_match = re.search(r"\{.*\}", text, re.DOTALL)
        if not json_match:
            return self._empty_scores(f"No JSON in response: {text[:100]}")

        try:
            data = json.loads(json_match.group(0))
        except json.JSONDecodeError:
            return self._empty_scores(f"Invalid JSON: {text[:100]}")

        correctness = data.get("correctness")
        completeness = data.get("completeness")
        faithfulness = data.get("faithfulness")

        # Compute overall average of the three dimensions
        valid_scores = [
            s for s in (correctness, completeness, faithfulness)
            if isinstance(s, (int, float))
        ]
        overall = round(sum(valid_scores) / len(valid_scores), 2) if valid_scores else None

        return {
            "judge_correctness": correctness,
            "judge_completeness": completeness,
            "judge_faithfulness": faithfulness,
            "judge_overall": overall,
            "judge_justification": data.get("justification", ""),
        }

    def _empty_scores(self, reason: str) -> dict:
        """Return a scores dict with None values and a reason."""
        return {
            "judge_correctness": None,
            "judge_completeness": None,
            "judge_faithfulness": None,
            "judge_overall": None,
            "judge_justification": reason,
        }
