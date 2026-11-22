"""Prompts, loaded from Langfuse with an in-code fallback."""

from typing import cast

from langchain_core.prompts import ChatPromptTemplate
from langfuse.model import ChatMessageDict

from .config import langfuse

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
