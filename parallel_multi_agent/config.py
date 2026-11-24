"""Environment, Langfuse client and the shared LLM."""

from dotenv import load_dotenv

load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler
from langfuse.span_filter import is_default_export_span

# from langchain_openai import ChatOpenAI

NOISE_SPANS = {
    "__start__",
    "tools_condition",
    "route_agent",
    "ChatPromptTemplate",
    "dispatcher",
    "route_verdict",
    "collect",
}


def should_export(span) -> bool:
    return is_default_export_span(span) and span.name not in NOISE_SPANS


class GraphCallbackHandler(CallbackHandler):
    """CallbackHandler that keeps subgraph runs out of the main graph's Langfuse graph view.

    Langfuse draws the agent graph from metadata `langgraph_node` + `langgraph_step` only.
    A subgraph restarts at its own `__start__` and step 1, so its nodes get merged into the
    main graph's steps. Here every run inside a subgraph is relabelled with the parent node
    (e.g. `worker`) and the parent's step; the real node stays in `subgraph_node`.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._parent_steps: dict[str, int] = {}  # top-level task ns -> its step

    def _get_langchain_observation_metadata(self, *, metadata=None, **kwargs):
        if metadata:
            ns = (metadata).get("langgraph_checkpoint_ns")
            if ns:
                top = ns.split("|", 1)[0]  # e.g. "worker:<task_id>"
                if "|" not in ns:
                    self._parent_steps.setdefault(top, metadata["langgraph_step"])
                else:
                    metadata = {
                        **metadata,
                        "subgraph_node": metadata.get("langgraph_node"),
                        "subgraph_step": metadata.get("langgraph_step"),
                        "langgraph_node": top.split(":", 1)[0],
                        "langgraph_step": self._parent_steps.get(
                            top, metadata.get("langgraph_step")
                        ),
                    }
            return super()._get_langchain_observation_metadata(
                metadata=metadata, **kwargs
            )


# Initialize Langfuse client
# langfuse = get_client()
langfuse = Langfuse(should_export_span=should_export)

llm_gemini = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", temperature=0)

# llm_openai = ChatOpenAI(model="gpt-6-luna", temperature=0)
