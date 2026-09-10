"""
Generation Chain Module.

Implements the answer-generation step as a LangChain LCEL chain:

    prompt | ChatGoogleGenerativeAI | StrOutputParser

Using LangChain's ChatGoogleGenerativeAI wrapper (instead of the raw Gemini
SDK) means the LLM call emits LangChain callback events — so the full prompt
and response show up in LangChain tracing (set_debug / Langfuse) automatically.

Design Choices:
    - Text-only chain (images are handled by the separate image path, currently
      disabled via settings.enable_images). Keeping this text-only makes the
      chain simple and fully traceable.
    - Low temperature for factual, consistent answers.
    - The system prompt / output format is preserved exactly from the prior
      implementation so answer quality and formatting are unchanged.
"""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from app.core.config import settings
from app.core.logging import get_logger
from app.retrieval.multimodal_retriever import RetrievalResult

logger = get_logger(__name__)

# ─── System Prompt ────────────────────────────────────────────────────────────
# Defines the LLM's role, constraints, and output expectations.
# This is critical for answer quality and grounding.

SYSTEM_PROMPT = """You are a financial analyst assistant specializing in analyzing company earnings reports (10-Q filings).

You answer questions using ONLY the provided context (text passages and tables).

RULES:
1. Base answers solely on the provided context. Never hallucinate or infer data not present.
2. Financial tables typically have a CURRENT period column and a PRIOR YEAR comparison column. Report whichever the QUESTION asks for:
   - If the question asks for a single value ("What was X?"), give the CURRENT period value.
   - If the question asks to COMPARE, examine CHANGES, or discuss TRENDS, include BOTH the current and prior-year figures so the comparison is complete.
   Let the question's intent decide — do not omit relevant comparison figures when the question calls for them.
3. If the answer cannot be determined from the context, state this clearly.
4. The source filename format is "{{year}} {{quarter}} {{ticker}}.pdf" (e.g., "2023 Q3 AAPL.pdf"). The year and quarter in the filename indicate which quarter the document reports on.
5. When the question asks about the DRIVERS, COMPONENTS, CAUSES, or what CONTRIBUTED to a change, break the answer down into the specific underlying line items with their values (e.g. for operating expenses: R&D and SG&A figures) — the component figures ARE the drivers. Do not answer with only the aggregate total when the components are available in the context. (This applies only to driver/component/cause questions; for a plain value question, just give the value.)
6. When the question asks to list something "for EACH period", "SEPARATELY per quarter", "ACROSS the reported quarters", or similar enumeration, enumerate EVERY period present in the context — do not report only a subset. If the context has data for four quarters, cover all four.
7. When the question refers to "the MOST RECENT 10-Q" (singular), focus the answer on the single latest-quarter document and cite only that one, unless the question also asks to compare with earlier quarters.

OUTPUT FORMAT — follow this strictly:

For questions about data across multiple periods/documents:
- Present each data point as a bullet (- ) with the specific value, period, and source filename.
- Format: "- For the quarterly period ended [date], [metric] was $X million. (SOURCE: [year] [quarter] [ticker].pdf)"
- After all data points, add a blank line then a brief 1-3 sentence analytical summary describing the trend or key insight.
- End with: SOURCE(S): [filename1], [filename2], ...

For questions about a single document:
- Present the answer in clear paragraphs or bullet points with specific values.
- Reference specific sections (e.g., "Note 7: Income Taxes") when applicable.
- End with: SOURCE(S): [filename]

For questions about segment/category breakdowns:
- Use a labeled format with the segment name followed by its metrics as sub-bullets.
- Add a brief comparative summary after all segments.
- End with: SOURCE(S): [filename(s)]

IMPORTANT:
- Answer according to what the QUESTION asks. For a single value, report the current period; for comparisons/changes/trends, include the prior-year figures too.
- Always use the format "[year] [quarter] [ticker].pdf" for source filenames (e.g., "2023 Q3 AAPL.pdf").
- Do NOT add elaborate markdown tables, headers, horizontal rules, or extra formatting.
- Keep answers factual, concise, and directly responsive to the question.
"""

# ─── Prompt template ──────────────────────────────────────────────────────────
# The system prompt above uses literal "{{...}}" so ChatPromptTemplate doesn't
# treat the filename example as a variable. Only {context} and {question} are
# real template variables.
GENERATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", """Here is the relevant context retrieved from financial documents:

{context}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Question: {question}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Provide a detailed answer based on the context above. Reference specific sources
(company, quarter, page) when citing data. If specific numbers are available in
the tables, include them in your response."""),
])


class GeminiMultiModalGenerator:
    """
    Answer generator built on a LangChain LCEL chain.

    Chain:  GENERATION_PROMPT | ChatGoogleGenerativeAI | StrOutputParser

    Because the LLM is a LangChain component, its prompt and response are
    captured by LangChain callbacks/tracing. The public generate() interface
    is unchanged so the rest of the pipeline is unaffected.

    Usage:
        generator = GeminiMultiModalGenerator()
        answer = generator.generate(retrieval_result)
    """

    def __init__(self):
        """
        Initialize the ChatGoogleGenerativeAI LLM and build the LCEL chain.

        Raises:
            ValueError: If GEMINI_API_KEY is not configured.
        """
        if not settings.gemini_api_key:
            raise ValueError(
                "GEMINI_API_KEY not configured. "
                "Get a free key at https://aistudio.google.com/apikey "
                "and add it to your .env file."
            )

        # LangChain wrapper around Gemini — this is what makes the call traceable.
        self.llm = ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            google_api_key=settings.gemini_api_key,
            temperature=settings.generation_temperature,
            max_output_tokens=settings.max_output_tokens,
        )

        # LCEL chain: prompt → llm → string output
        self.chain = GENERATION_PROMPT | self.llm | StrOutputParser()

        logger.info(f"Gemini generator (LangChain) initialized: model={settings.gemini_model}")

    def generate(
        self,
        retrieval_result: RetrievalResult,
        include_images: bool = False,
    ) -> str:
        """
        Generate an answer from retrieval results via the LCEL chain.

        Args:
            retrieval_result: Combined retrieval results (text + tables).
            include_images: Accepted for interface compatibility. The image path
                            is handled separately and is off by default; this
                            text chain ignores it.

        Returns:
            Generated answer string.
        """
        if not retrieval_result.has_results:
            return (
                "I couldn't find relevant information in the documents "
                "to answer your question. Please try rephrasing or check "
                "that the relevant documents have been ingested."
            )

        context_text = retrieval_result.get_text_context()

        # Log what is being sent (our own console trace; complements LangChain's)
        if settings.show_retrieval_trace:
            logger.info(
                f"🤖 GENERATE → sending to {settings.gemini_model}: "
                f"{len(retrieval_result.text_documents)} text + "
                f"{len(retrieval_result.table_documents)} tables | "
                f"~{len(context_text)} context chars | "
                f"temp={settings.generation_temperature}, "
                f"max_out={settings.max_output_tokens}"
            )

        try:
            answer = self.chain.invoke({
                "context": context_text,
                "question": retrieval_result.query,
            })

            if settings.show_retrieval_trace:
                logger.info(f"✅ GENERATE ← received response ({len(answer or '')} chars)")

            return answer

        except Exception as e:
            logger.error(f"Gemini generation failed: {e}")
            return f"Error generating response: {e}"
