"""Worker subgraph: runs one plan step with the agent that fits its task type."""

import base64
import mimetypes
from pathlib import Path
from typing import Literal

from langchain_core.messages import AnyMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from tools import (
    fetch_page,
    get_youtube_transcript,
    read_excel,
    read_file,
    read_stored,
    run_python,
    search_web,
    search_wiki,
    transcribe_audio,
    wikipedia_tables,
)

from .config import llm_gemini
from .data_models import StepResult, WorkerOutput, WorkerState
from .prompts import step_prompt

SEARCH_TOOLS = [
    search_web,
    fetch_page,
    search_wiki,
    wikipedia_tables,
    read_stored,
    get_youtube_transcript,
]
FILE_TOOLS = [transcribe_audio, read_excel, read_file]
CODE_TOOLS = [run_python]

search_llm = llm_gemini.bind_tools(SEARCH_TOOLS)
file_llm = llm_gemini.bind_tools(FILE_TOOLS)
code_llm = llm_gemini.bind_tools(CODE_TOOLS)

search_chain = step_prompt | search_llm
file_chain = step_prompt | file_llm
code_chain = step_prompt | code_llm
reason_chain = step_prompt | llm_gemini


def build_prompt_vars(
    state: WorkerState, extra: list[AnyMessage] | None = None
) -> dict:
    return {
        "question": state.question,
        "instruction": state.step.instruction,
        "context": f"Result from earlier steps:\n{state.context}\n"
        if state.context
        else "",
        "file": f"Attached file ({state.file_type}): {state.file_path}\n"
        if state.file_type != "none"
        else "",
        "messages": state.messages + (extra if extra is not None else []),
    }


def search_agent(state: WorkerState):
    reply = search_chain.invoke(build_prompt_vars(state))
    return {"messages": [reply]}


def file_agent(state: WorkerState):
    new = []
    if state.file_type == "image" and not state.messages:
        if state.file_path is None:
            raise ValueError("Image file path is required")
        with open(state.file_path, "rb") as f:
            data = base64.b64encode(f.read()).decode()
        mime = mimetypes.guess_type(state.file_path)[0] or "image/png"
        new.append(
            HumanMessage(
                content=[
                    {"type": "text", "text": "This is the attached image."},
                    {"type": "image", "base64": data, "mime_type": mime},
                ]
            )
        )
    reply = file_chain.invoke(build_prompt_vars(state, new))  # ty: ignore[invalid-argument-type]
    return {"messages": [*new, reply]}


def code_agent(state: WorkerState):
    reply = code_chain.invoke(build_prompt_vars(state))
    return {"messages": [reply]}


def reason_agent(state: WorkerState):
    reply = reason_chain.invoke(build_prompt_vars(state))
    return {"messages": [reply]}


def finalize(state: WorkerState):
    return {
        "step_results": [
            StepResult(step_id=state.step.id, output=state.messages[-1].text)
        ]
    }


def route_agent(
    state: WorkerState,
) -> Literal["search_agent", "file_agent", "code_agent", "reason_agent"]:
    return f"{state.step.task_type}_agent"  # type: ignore


worker = StateGraph(WorkerState, output_schema=WorkerOutput)
worker.add_conditional_edges(START, route_agent)

worker.add_node("search_agent", search_agent)
worker.add_node("file_agent", file_agent)
worker.add_node("code_agent", code_agent)
worker.add_node("reason_agent", reason_agent)
worker.add_node("finalize", finalize)

worker.add_node("search_tools", ToolNode(SEARCH_TOOLS, handle_tool_errors=True))
worker.add_node("file_tools", ToolNode(FILE_TOOLS, handle_tool_errors=True))
worker.add_node("code_tools", ToolNode(CODE_TOOLS, handle_tool_errors=True))

worker.add_conditional_edges(
    "search_agent", tools_condition, {"tools": "search_tools", END: "finalize"}
)
worker.add_conditional_edges(
    "file_agent", tools_condition, {"tools": "file_tools", END: "finalize"}
)
worker.add_conditional_edges(
    "code_agent", tools_condition, {"tools": "code_tools", END: "finalize"}
)

worker.add_edge("search_tools", "search_agent")
worker.add_edge("file_tools", "file_agent")
worker.add_edge("code_tools", "code_agent")
worker.add_edge("reason_agent", "finalize")
worker.add_edge("finalize", END)

worker_graph = worker.compile(name="step-agent")


if __name__ == "__main__":
    from langfuse import propagate_attributes
    from langfuse.langchain import CallbackHandler

    from .config import langfuse
    from .data_models import Step

    # Verify connection
    if langfuse.auth_check():
        print("Langfuse client is authenticated and ready!")
    else:
        print("Authentication failed. Please check your credentials and host.")

    worker_graph.get_graph().draw_mermaid_png(
        output_file_path=str(Path(__file__).with_name("worker.png"))
    )

    question = "How many studio albums did Mercedes Sosa publish from 2000 to 2009?"
    step = Step(
        id=1,
        task_type="search",
        instruction="Find the studio albums of Mercedes Sosa from 2000 to 2009",
        depends_on=[],
    )

    with (
        langfuse.start_as_current_observation(
            as_type="agent",  # this is an agent run
            name="test-worker-step",
            input={"question": question, "step": step.instruction},
            metadata={"step_id": step.id, "task_type": step.task_type},
        ) as root,
        propagate_attributes(
            # trace_name="test-worker-step", # trace name is inherited from the observation name
            session_id="test-session",
            tags=["langgraph", "worker-sub-graph"],
        ) as p,
    ):
        # Initialize Langfuse CallbackHandler for Langchain (tracing)
        langfuse_handler = CallbackHandler()

        out = worker_graph.invoke(
            WorkerState(step=step, question=question),
            config={
                "callbacks": [langfuse_handler],
                # "configurable": {
                #   "thread_id": "test"
                # }
            },
        )
        answer = out["step_results"]
        root.update(output=answer)  # clean output
        print(out)
    langfuse.flush()
