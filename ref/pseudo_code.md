TYPES:
  TaskType = search | file | code | reason
  FileType = none | image | audio | python | excel | other
  Verdict  = correct | incorrect | partial

MODELS:
  Step:       id: int, task_type: TaskType, instruction: str,
              depends_on: Optional[list[int]] = None
  Plan:       plan: str, steps: list[Step]
  Check:      verdict: Verdict, feedback: str
  StepResult: step_id: int, output: str

REDUCER results_reducer(old, new):
  IF new is None → []            # reset on re-plan
  RETURN old + new

────────────────────────────────────────────
MAIN STATE (AgentState):
  input:   task_id, question, file_path, file_type
  control: plan, steps: list[Step], retries_loop = 0, verdict, feedback
  work:    step_results: Annotated[list[StepResult], results_reducer] = []
  output:  draft_answer, answer

WORKER STATE (WorkerState):
  step: Step, question, file_path, file_type
  context: str                                  # results of depends_on
  messages: Annotated[list[AnyMessage], add_messages] = []
  step_results: Annotated[list[StepResult], results_reducer] = []

────────────────────────────────────────────
WORKER SUBGRAPH

NODE search_agent / file_agent / code_agent / reason_agent (w) → dict:
  sys = System(AGENT_PROMPT
               + "Question: " + w.question
               + "Your step: " + w.step.instruction
               + "Results from earlier steps: " + w.context
               + "Do only this step. Give the step result.")
  extra = [Human(image)] IF file_agent AND file_type == image AND messages empty
  reply = agent_llm.invoke([sys] + w.messages + extra)
  RETURN messages = extra + [reply]

NODE search_tools = ToolNode(SEARCH_TOOLS, handle_tool_errors=True)
NODE file_tools   = ToolNode(FILE_TOOLS,   handle_tool_errors=True)
NODE code_tools   = ToolNode(CODE_TOOLS,   handle_tool_errors=True)

NODE finish(w) → dict:
  RETURN step_results = [StepResult(w.step.id, w.messages[-1].content)]

ROUTE route_agent(w) → search_agent | file_agent | code_agent | reason_agent:
  RETURN w.step.task_type + "_agent"

WORKER GRAPH:
  add_conditional_edges: START → route_agent
  add_conditional_edges: search_agent → tools_condition, {"tools": search_tools, END: finish}
  add_conditional_edges: file_agent   → tools_condition, {"tools": file_tools,   END: finish}
  add_conditional_edges: code_agent   → tools_condition, {"tools": code_tools,   END: finish}
  add_edge: reason_agent → finish
  add_edge: search_tools → search_agent
  add_edge: file_tools   → file_agent
  add_edge: code_tools   → code_agent
  add_edge: finish → END
  worker = compile()

────────────────────────────────────────────
MAIN GRAPH

NODE planner(state) → dict:
  prompt = question + file_type + file_path
  IF feedback: prompt += "Last try failed: " + feedback
  res = planner_llm(Plan).invoke(prompt)
  FOR s IN res.steps:
    IF s.depends_on is None:
      s.depends_on = [s.id - 1] IF s.id > 1 ELSE []    # sequential by default
  RETURN plan, steps = res.steps, step_results = None  # reset

ROUTE dispatch(state) → list[Send] | "verifier":     # conditional edge condition
  done  = {r.step_id FOR r IN step_results}
  ready = [s FOR s IN steps
           IF s.id NOT IN done AND all(d IN done FOR d IN s.depends_on)]
  IF ready is empty → "verifier"
  RETURN [Send("worker", {
            step: s, question, file_path, file_type,
            context: join(r.output FOR r IN step_results IF r.step_id IN s.depends_on)
          }) FOR s IN ready]

NODE worker = worker subgraph

NODE verifier(state) → dict:
  draft = output of the last step
  res = verifier_llm(Check).invoke(question, plan, step_results, draft)
  IF res.verdict != correct:
    RETURN draft_answer = draft, verdict, feedback, retries_loop + 1
  RETURN draft_answer = draft, verdict

ROUTE route_verdict(state) → planner | formatter:
  IF verdict == correct  → formatter
  IF retries_loop >= 2   → formatter
  ELSE                   → planner

NODE formatter(state) → dict:
  answer = strip(format_llm.invoke(FORMAT_RULES + question + draft_answer))
  RETURN answer = answer OR "unknown"

MAIN GRAPH:
  add_edge: START → planner
  add_conditional_edges: planner  → dispatch, ["worker", "verifier"]
  add_conditional_edges: worker   → dispatch, ["worker", "verifier"]
  add_conditional_edges: verifier → route_verdict, ["planner", "formatter"]
  add_edge: formatter → END
  app = compile(checkpointer = InMemorySaver)

────────────────────────────────────────────
EXAMPLES

Sequential (chess):
  1 file   "Read board, give FEN"            depends_on [] 
  2 code   "Best move for black from FEN"    depends_on [1] (default)
  3 reason "Give move in algebraic notation" depends_on [2] (default)
  planner → worker(1) → worker(2) → worker(3) → verifier → formatter

Parallel (two lookups):
  1 search "Find X"                          depends_on []
  2 search "Find Y"                          depends_on []
  3 reason "Combine X and Y"                 depends_on [1, 2]
  planner → [worker(1) ∥ worker(2)] → worker(3) → verifier → formatter

────────────────────────────────────────────
RUN
  FOR q IN questions:
    config = {thread_id: q.task_id, recursion_limit: 50}
    result = app.invoke(AgentState(...), config)
    save(result["answer"])
