# Revisit: error handling

Inside the graph, tool errors can't crash anything: every `ToolNode` uses `handle_tool_errors=True`, so a failing tool returns an error message to the LLM. The risks are in code outside the tools.

## What can stop a run or the whole app

1. **Startup in `variables.py`:**
   - `OpenAI()` raises if `OPENAI_API_KEY` isn't set.
   - `TavilySearch()` and `TavilyExtract()` raise if `TAVILY_API_KEY` isn't set.

   Either one stops the app on import. On the Hugging Face Space, both keys must be set as Space secrets. The OpenAI client is only used by `transcribe_audio`.
2. **LLM calls in the agent nodes:**
   - `search_agent`, `file_agent`, `code_agent`, `reason_agent` and `planner` call `.invoke()` with no error handling. A Gemini 429 rate limit, a timeout or a safety block raises an exception and ends the whole graph run.
   - In `planner`, structured output can come back as `None`, and then `res.steps` fails with `AttributeError`.
3. **Too many steps:** if the agent keeps calling tools, LangGraph's step limit (25 by default) raises `GraphRecursionError`. The search tools are designed to be called several times (outline, then section, then keyword), which makes this more likely.
4. **File handling in `file_agent`:** the `raise ValueError` and the `open()` of the image run in the agent node, not in a tool, so a missing file stops the run.
5. **`api.py` `get_questions`:** `raise_for_status()` is intended. If the course API is down, there's nothing to run anyway.

## Suggested fixes

- Give the LLMs automatic retries: `max_retries` on `ChatGoogleGenerativeAI`, or `.with_retry()` on the chains.
- Pass a `recursion_limit` in the run config.
- Wrap each question's `invoke` in a `try/except` that records a fallback answer, so one failed question doesn't end the whole evaluation.
- Create the OpenAI client only when `transcribe_audio` first needs it.
