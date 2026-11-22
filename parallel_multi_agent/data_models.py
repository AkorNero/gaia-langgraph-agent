"""Types and state models shared by the main graph and the worker subgraph."""

from typing import Annotated, Literal

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

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


class StepResult(BaseModel):
    step_id: int
    output: str


def results_reducer(
    old: list[StepResult], new: list[StepResult] | None
) -> list[StepResult]:
    if new is None:
        return []
    return old + new


class WorkerOutput(BaseModel):
    # When you add a compiled subgraph as a node, LangGraph copies only the keys that both states have back to the main state.
    step_results: Annotated[list[StepResult], results_reducer] = Field(
        default_factory=list
    )


class WorkerState(WorkerOutput):
    step: Step
    question: str
    file_path: str | None = None
    file_type: FileType | None = Field(default="none")
    context: str | None = None  # results of depends_on
    messages: Annotated[list[AnyMessage], add_messages] = Field(default_factory=list)


class Check(BaseModel):
    verdict: Verdict
    feedback: str


class AgentState(BaseModel):
    # input
    task_id: str
    question: str
    file_path: str | None = None
    file_type: FileType | None = Field(default="none")

    # control
    plan: Plan | None = None
    steps: list[Step] = Field(default_factory=list)
    verdict: Verdict | None = None
    feedback: str | None = None
    retries: int = Field(default=0)
    max_retries: int = Field(default=2, lt=5)

    # work
    step_results: Annotated[list[StepResult], results_reducer] = Field(
        default_factory=list
    )

    # output
    draft_answer: str | None = None
    answer: str | None = None
