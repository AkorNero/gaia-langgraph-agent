"""Environment, Langfuse client and the shared LLM."""

from dotenv import load_dotenv

load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langfuse import Langfuse
from langfuse.span_filter import is_default_export_span

# from langchain_openai import ChatOpenAI

NOISE_SPANS = {"__start__", "tools_condition", "route_agent", "ChatPromptTemplate"}


def should_export(span) -> bool:
    return is_default_export_span(span) and span.name not in NOISE_SPANS


# Initialize Langfuse client
# langfuse = get_client()
langfuse = Langfuse(should_export_span=should_export)

llm_gemini = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", temperature=0)

# llm_openai = ChatOpenAI(model="gpt-6-luna", temperature=0)
