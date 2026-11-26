"""Main graph: plans the question into steps (work in progress)."""

from typing import Literal, cast

from langchain_core.messages import AIMessage, HumanMessage
from langfuse import propagate_attributes
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from .config import GraphCallbackHandler, langfuse, llm_gemini, llm_openai
from .data_models import AgentState, Check, Plan, WorkerState
from .prompts import (
    formatter_prompt,
    planner_prompt,
    verifier_prompt,
)
from .worker_graph import worker_graph

planner_llm = llm_gemini.with_structured_output(Plan)
verifier_llm = llm_openai.with_structured_output(Check)

planner_chain = planner_prompt | planner_llm
verifier_chain = verifier_prompt | verifier_llm
formatter_chain = formatter_prompt | llm_gemini


def planner(state: AgentState) -> dict:
    previous = ""
    if state.feedback and state.plan:
        old_steps = "\n".join(
            f"{s.id}. ({s.task_type}, depends on {s.depends_on}) {s.instruction}"
            for s in state.plan.steps
        )
        previous = (
            f"\nThe last attempt failed. Make a different plan.\n"
            f"Last plan: {state.plan.plan}\n{old_steps}\n"
            f"Draft answer it gave: {state.draft_answer}\n"
            f"Verifier feedback: {state.feedback}\n"
        )

    prompt_vars = {
        "question": state.question,
        "file": f"Attached file ({state.file_type}): {state.file_path}\n"
        if state.file_type != "none"
        else "",
        "previous_attempt": previous,
    }

    res = cast(
        Plan,
        planner_chain.invoke(prompt_vars),
    )

    mod_res = [
        f"\n\t- Task ID: {i.id} ({i.task_type}{f"; Depends on Step ID's: {i.depends_on}):" if i.depends_on else '):'})\n\t\t- Instruction: {i.instruction}"
        for i in res.steps
    ]

    return {
        "plan": res,
        "messages": [
            HumanMessage(
                content=f"Question: {prompt_vars['question']}\n{prompt_vars['file']}{prompt_vars['previous_attempt']}"
            ),
            AIMessage(content=f"Plan:\n{res.plan}\nSteps:" + "".join(mod_res)),
        ],
        "step_results": None,
    }


def dispatcher(state: AgentState) -> list[Send] | Literal["verifier"]:
    done = {r.step_id for r in state.step_results}
    if not (state.plan):
        return "verifier"
    ready = [
        s
        for s in state.plan.steps
        if s.id not in done and all(d in done for d in s.depends_on)
    ]
    if not ready:
        skipped = [s.id for s in state.plan.steps if s.id not in done]
        if skipped:
            print(f"steps never ran (bad depends_on): {skipped}")
        return "verifier"
    return [
        Send(
            "worker",
            WorkerState(
                question=state.question,
                file_type=state.file_type,
                file_path=state.file_path,
                step=item,
                context="\n---\n".join(
                    f"Step {r.step_id}: {r.output}"
                    for r in state.step_results
                    if r.step_id in item.depends_on
                )
                or None,
            ),
        )
        for item in ready
    ]


def verifier(state: AgentState) -> dict:
    assert state.plan is not None, "verifier ran before planner"
    results = state.step_results
    steps = {s.id: s for s in state.plan.steps}
    draft = results[-1].output if results else ""
    res = cast(
        Check,
        verifier_chain.invoke(
            {
                "question": state.question,
                "file": f"Attached file ({state.file_type}): {state.file_path}\n"
                if state.file_type != "none"
                else "",
                "plan": state.plan.plan,
                "step_results": "\n".join(
                    f"Step {r.step_id} ({steps[r.step_id].task_type}: {steps[r.step_id].instruction}):\n{r.output}"
                    for r in results
                ),
                "draft": draft,
            }
        ),
    )
    update = {"draft_answer": draft, "verdict": res.verdict, "feedback": res.feedback}
    if res.verdict != "correct":
        update["retries"] = state.retries + 1
    return update


def route_verdict(state: AgentState) -> Literal["planner", "formatter"]:
    if state.verdict == "correct":
        return "formatter"
    if state.retries >= state.max_retries:
        return "formatter"
    else:
        return "planner"


def formatter(state: AgentState) -> dict:
    reply = formatter_chain.invoke(
        {"question": state.question, "draft": state.draft_answer or ""}
    )
    answer = reply.text.strip().strip('"').rstrip(".")
    return {"answer": answer or "unknown"}


def collect(state: AgentState) -> dict:
    return {}  # join: runs once after all parallel workers have merged


flow = StateGraph(AgentState)
flow.add_node("planner", planner)
flow.add_node("worker", worker_graph)
flow.add_node("collect", collect)
flow.add_node("verifier", verifier)
flow.add_node("formatter", formatter)

flow.add_edge(START, "planner")
flow.add_conditional_edges("planner", dispatcher, ["worker", "verifier"])
flow.add_edge("worker", "collect")
flow.add_conditional_edges("collect", dispatcher, ["worker", "verifier"])
flow.add_conditional_edges("verifier", route_verdict, ["planner", "formatter"])
flow.add_edge("formatter", END)

app = flow.compile(name="gaia-agent", checkpointer=InMemorySaver())


def run(state: AgentState, session_id: str = "dev-parallel-main-graph") -> dict:
    """Answer one question with a Langfuse trace: one trace per task, grouped by session."""
    with (
        langfuse.start_as_current_observation(
            as_type="agent",
            name="gaia-question",
            input={"question": state.question, "file": state.file_path},
            metadata={"task_id": state.task_id, "file_type": state.file_type},
        ) as root,
        propagate_attributes(
            session_id=session_id,
            tags=["langgraph", "main-graph"],
            metadata={"task_id": state.task_id},
        ),
    ):
        result = app.invoke(
            state,
            config={
                "callbacks": [
                    GraphCallbackHandler()
                ],  # passed down to workers and subgraph
                "configurable": {"thread_id": state.task_id},
                "recursion_limit": 50,
            },
        )
        root.update(
            output={"answer": result.get("answer")},
            metadata={
                "verdict": result.get("verdict"),
                "retries": result.get("retries"),
            },
        )
    return result


if __name__ == "__main__":
    try:
        out = run(
            AgentState(
                task_id="dadsdsw1",
                question="Actor who played Ray in Polish Raymond → role in Magda M.?",
            )
        )
        print(out["answer"])
    finally:
        langfuse.flush()  # send buffered spans before the process exits
