"""Main graph: plans the question into steps (work in progress)."""

from typing import cast

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph

# from langgraph.checkpoint.memory import InMemorySaver
from .config import llm_gemini
from .data_models import AgentState, Check, Plan

planner_llm = llm_gemini.with_structured_output(Plan)
verifier_llm = llm_gemini.with_structured_output(Check)


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
    pass
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
