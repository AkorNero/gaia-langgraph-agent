import base64
from typing import Annotated, Literal, cast

from dotenv import load_dotenv
from langchain_core.messages import (
  AIMessage,
  AnyMessage,
  HumanMessage,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

# from langchain_openai import ChatOpenAI
from langfuse.model import ChatMessageDict

# from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel, Field

from tools import (
  get_youtube_transcript,
  read_excel,
  read_file,
  run_python,
  search_web,
  search_wiki,
  transcribe_audio,
  wikipedia_tables,
)

load_dotenv()

from langfuse import Langfuse, propagate_attributes
from langfuse.langchain import CallbackHandler
from langfuse.span_filter import is_default_export_span

NOISE_SPANS = {"__start__", "tools_condition", "route_agent", "ChatPromptTemplate"}


def should_export(span) -> bool:
  return is_default_export_span(span) and span.name not in NOISE_SPANS


# Initialize Langfuse client
# langfuse = get_client()
langfuse = Langfuse(should_export_span=should_export)

llm_gemini = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", temperature=0)

# llm_openai = ChatOpenAI(model="gpt-6-luna", temperature=0)   # check the current model names


TaskType = Literal["search", "file", "code", "reason"]
FileType = Literal["none", "image", "audio", "python", "excel", "other"]
Verdict = Literal["correct", "incorrect", "partial"]

EXT_TO_TYPE: dict[str, FileType] = {
  ".png": "image",
  ".jpg": "image",
  ".mp3": "audio",
  ".py": "python",
  ".xlsx": "excel",
}


class Step(BaseModel):
  id: int
  task_type: TaskType
  instruction: str
  depends_on: list[int] | None = None


class Plan(BaseModel):
  plan: str
  steps: list[Step]


class Check(BaseModel):
  verdict: Verdict
  feedback: str


class StepResult(BaseModel):
  step_id: int
  output: str


def results_reducer(
  old: list[StepResult], new: list[StepResult] | None
) -> list[StepResult]:
  if new is None:
    return []
  return old + new


class AgentState(BaseModel):
  # input
  task_id: str
  question: str
  file_path: str | None = None
  file_type: FileType | None = Field(default="none")

  # control
  plan: Plan | None = None
  verdict: Verdict | None = None
  feedback: str | None = None
  retries: int = Field(default=0)
  max_retries: int = Field(lt=5)

  # work
  step_results: Annotated[list[str], results_reducer] = Field(default_factory=list)

  # output
  draft_answer: str | None = None
  answer: str | None = None


class WorkerState(BaseModel):
  step: Step
  question: str
  file_path: str | None = None
  file_type: FileType | None = Field(default="none")
  context: str | None = None  # results of depends_on
  messages: Annotated[list[AnyMessage], add_messages] = Field(default_factory=list)

  # When you add a compiled subgraph as a node, LangGraph copies only the keys that both states have back to the main state.
  step_results: Annotated[list[StepResult], results_reducer] = Field(
    default_factory=list
  )


SEARCH_TOOLS = [search_web, search_wiki, wikipedia_tables, get_youtube_transcript]
FILE_TOOLS = [transcribe_audio, read_excel, read_file]
CODE_TOOLS = [run_python]

planner_llm = llm_gemini.with_structured_output(Plan)
verifier_llm = llm_gemini.with_structured_output(Check)

search_llm = llm_gemini.bind_tools(SEARCH_TOOLS)
file_llm = llm_gemini.bind_tools(FILE_TOOLS)
code_llm = llm_gemini.bind_tools(CODE_TOOLS)

STEP_FALLBACK = cast(
  list[ChatMessageDict],
  [
    {
      "role": "system",
      "content": "You do one step of a bigger task.\nFull question: {{question}}",
    },
    {
      "role": "user",
      "content": "Your step: {{instruction}}\n{{context}}{{file}}Do only your step. When done, reply with the step result only.",
    },
    {"type": "placeholder", "name": "messages"},
  ],
)

lf_step_prompt = langfuse.get_prompt(
  "worker-sub-graph/step", type="chat", fallback=STEP_FALLBACK
)
step_prompt = ChatPromptTemplate(
  lf_step_prompt.get_langchain_prompt(),
  metadata={"langfuse_prompt": lf_step_prompt},
)

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
    new.append(
      HumanMessage(
        content=[
          {"type": "text", "text": "This is the attached image."},
          {"type": "image", "base64": data, "mime_type": "image/png"},
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


def route_agent(
  state: WorkerState,
) -> Literal["search_agent", "file_agent", "code_agent", "reason_agent"]:
  return f"{state.step.task_type}_agent"  # type: ignore


worker = StateGraph(WorkerState)
worker.add_conditional_edges(START, route_agent)

worker.add_node("search_agent", search_agent)
worker.add_node("file_agent", file_agent)
worker.add_node("code_agent", code_agent)
worker.add_node("reason_agent", reason_agent)

worker.add_node("search_tools", ToolNode(SEARCH_TOOLS, handle_tool_errors=True))
worker.add_node("file_tools", ToolNode(FILE_TOOLS, handle_tool_errors=True))
worker.add_node("code_tools", ToolNode(CODE_TOOLS, handle_tool_errors=True))

worker.add_conditional_edges(
  "search_agent", tools_condition, {"tools": "search_tools", END: END}
)
worker.add_conditional_edges(
  "file_agent", tools_condition, {"tools": "file_tools", END: END}
)
worker.add_conditional_edges(
  "code_agent", tools_condition, {"tools": "code_tools", END: END}
)

worker.add_edge("search_tools", "search_agent")
worker.add_edge("file_tools", "file_agent")
worker.add_edge("code_tools", "code_agent")
worker.add_edge("reason_agent", END)

# checkpointer = InMemorySaver()

# worker_graph = worker.compile(name="step-agent", checkpointer=checkpointer)

worker_graph = worker.compile(name="step-agent")


def planner(state: AgentState) -> dict:
  prompt = f"Question: {state.question}" + (
    f"\nFile given with the above question: \n\t- File type: {state.file_type}.\n\t- File path: {state.file_path}"
    if state.file_type != "none"
    else ""
  )
  if state.feedback is not None:
    prompt += "Last try failed: " + state.feedback

  res = cast(Plan, planner_llm.invoke(prompt))

  mod_res = [
    f"\n\t- Task Type: {i.task_type}\n\t\t- Instruction: {i.instruction}"
    for i in res.steps
  ]

  return {
    "plan": res,
    "messages": [
      HumanMessage(content=state.question),
      AIMessage(content="Steps:\n" + "".join(mod_res)),
    ],
  }


flow = StateGraph(AgentState)
flow.add_node("planner", planner)
flow.add_edge(START, "planner")
flow.add_edge("planner", END)

if __name__ == "__main__":
  # Verify connection
  if langfuse.auth_check():
    print("Langfuse client is authenticated and ready!")
  else:
    print("Authentication failed. Please check your credentials and host.")

  worker_graph.get_graph().draw_mermaid_png(output_file_path="worker.png")

  question = "How many studio albums did Mercedes Sosa publish from 2000 to 2009?"
  step = Step(
    id=1,
    task_type="search",
    instruction="Find the studio albums of Mercedes Sosa from 2000 to 2009",
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

    answer = out["messages"][-1].text
    root.update(output=answer)  # clean output
    print(answer)

  langfuse.flush()

  # app = flow.compile(checkpointer=checkpointer)

  # config = {"configurable": {"thread_id": "customer_123"}}

  # result = app.invoke(
  #   AgentState(
  #     task_id="dadsdsw1",
  #     question="Actor who played Ray in Polish Raymond → role in Magda M.?",
  #     max_retries=2
  #   ),
  #   config
  # )

  # print(result)
