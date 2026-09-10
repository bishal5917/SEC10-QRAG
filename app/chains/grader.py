"""
Document Relevance Grading Chain.

Implements a lightweight grading step that evaluates whether retrieved
documents are actually relevant to the user's query. This prevents
the LLM from generating answers based on irrelevant context, which
is a common failure mode in RAG systems.

Architecture (in the LangGraph workflow):
    Retrieve → Grade → (relevant docs only) → Generate

    If no documents pass the relevance threshold, the workflow can
    either retry with a reformulated query or inform the user that
    no relevant information was found.

Design Choice:
    Uses Gemini for grading (simple yes/no classification) rather than
    a separate model, keeping the system simple while leveraging the
    LLM's understanding of relevance.
"""

from google import genai
from google.genai import types
from langchain_core.documents import Document

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# ─── Grading Prompt ───────────────────────────────────────────────────────────
GRADING_PROMPT = """You are a relevance grading assistant. Your task is to determine if a retrieved document is relevant to the user's question.

Evaluate based on:
- Does the document contain information that could help answer the question?
- Is there factual overlap between the document content and what the question asks about?
- For financial questions: does the document mention the relevant company, time period, or metrics?

Respond with ONLY one word: "relevant" or "irrelevant"

Document:
{document}

Question: {question}

Verdict:"""


class DocumentGrader:
    """
    Grades retrieved documents for relevance to the user query.

    Uses Gemini to perform binary classification (relevant/irrelevant)
    on each retrieved document. This filtering step improves answer
    quality by removing noisy retrievals.

    Usage:
        grader = DocumentGrader()
        relevant_docs = grader.grade_documents(documents, query)
    """

    def __init__(self):
        """Initialize the Gemini client for grading."""
        if not settings.gemini_api_key:
            raise ValueError("GEMINI_API_KEY required for document grading")

        self.client = genai.Client(api_key=settings.gemini_api_key)
        self.model = settings.gemini_model

    def grade_documents(
        self,
        documents: list[Document],
        query: str,
    ) -> list[Document]:
        """
        Filter documents by relevance to the query.

        Each document is independently evaluated by the LLM.
        Only documents classified as "relevant" are returned.

        Args:
            documents: List of retrieved Documents to grade.
            query: The user's question.

        Returns:
            Filtered list containing only relevant Documents.
        """
        if not documents:
            return []

        relevant_docs = []

        for doc in documents:
            # Truncate long documents for efficient grading
            content = doc.page_content[:1500]

            prompt = GRADING_PROMPT.format(document=content, question=query)

            try:
                # Rate limit: wait between calls to stay within free tier (5 RPM)
                import time
                time.sleep(1.5)

                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.0,  # Deterministic grading
                        max_output_tokens=10,
                    ),
                )

                # Handle None response (some models return None for short outputs)
                if response.text is None:
                    relevant_docs.append(doc)
                    continue

                verdict = response.text.strip().lower()

                if "relevant" in verdict and "irrelevant" not in verdict:
                    relevant_docs.append(doc)
                else:
                    source = doc.metadata.get("filename", "?")
                    page = doc.metadata.get("page_number", "?")
                    logger.debug(
                        f"Filtered out: {source} p.{page} (irrelevant to query)"
                    )

            except Exception as e:
                # On grading failure, include the document (fail-open)
                logger.warning(f"Grading failed, including document: {e}")
                relevant_docs.append(doc)

        logger.info(
            f"Grading: {len(relevant_docs)}/{len(documents)} documents deemed relevant"
        )
        return relevant_docs
