"""The task agent: create_agent with the task tools, per-session memory and model fallback."""

from datetime import date
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, ModelRequest, dynamic_prompt
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from app import tools
from app.llm import build_model
from app.tools import AgentContext

SYSTEM_PROMPT = """\
You are TaskFlow's assistant. You manage the user's tasks. Today is {today} ({weekday}).

Rules:
- Act only through your tools. Never invent tasks, IDs or results.
- Convert dates like "tomorrow" or "Friday" to ISO dates (YYYY-MM-DD) using today's date above.
- Describe a task to update or delete in the tool's `target`. Pass `task_id` only when you copy it
  from a candidate list a tool returned earlier.
- When a tool returns several candidate tasks, do not act. Ask the user which one they mean.
- To mark a task done, use update_task with status=done.
- Deleting needs confirmation: after delete_task asks, ask the user, and call delete_task again
  only after the user has confirmed in their next message.
- Never say an action succeeded unless a tool result said so. If a tool returned an error or
  "not found", tell the user plainly.
- If the user gives several instructions in one message, make one tool call for each.
- Keep replies short.
"""


def build_system_prompt(today: date) -> str:
    return SYSTEM_PROMPT.format(today=today.isoformat(), weekday=today.strftime("%A"))


@dynamic_prompt
def _system_prompt(request: ModelRequest) -> str:
    # Built on every model call so "today" is right even if the process runs past midnight.
    return build_system_prompt(date.today())


# One in-memory checkpointer for the whole process: history is kept per session_id (thread_id).
_checkpointer = InMemorySaver()
_default_agent: Any = None


def _build_agent(model: BaseChatModel, fallback: list[AgentMiddleware[Any, Any, Any]]) -> Any:
    return create_agent(
        model=model,
        tools=tools.TOOLS,
        # The prompt middleware goes first (outermost) so fallback models get the same prompt.
        middleware=[_system_prompt, *fallback],
        checkpointer=_checkpointer,
        context_schema=AgentContext,
    )


def _get_agent(model: BaseChatModel | None) -> Any:
    global _default_agent
    if model is not None:
        return _build_agent(model, [])
    if _default_agent is None:
        primary, fallback = build_model()
        _default_agent = _build_agent(primary, fallback)
    return _default_agent


def run_agent(message: str, session_id: str, model: BaseChatModel | None = None) -> str:
    """Send one user message to the agent and return its reply.

    `model` is for tests; normally the chain from build_model() is used. Provider errors that
    survive the whole fallback chain are raised to the caller.
    """
    agent = _get_agent(model)
    tools.next_turn(session_id)
    result = agent.invoke(
        {"messages": [HumanMessage(message)]},
        config={"configurable": {"thread_id": session_id}},
        context=AgentContext(session_id=session_id),
    )
    return result["messages"][-1].text or "(The assistant returned no text.)"
