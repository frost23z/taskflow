import sqlite3
from collections.abc import Iterator
from datetime import date
from pathlib import Path
from typing import Any

import pytest
from chromadb.api.models.Collection import Collection
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.checkpoint.sqlite import SqliteSaver
from pydantic import ConfigDict
from sqlalchemy.engine import Engine
from sqlmodel import Session

from app import agent, services, tools
from app.agent import build_system_prompt, run_agent
from app.schemas import TaskCreate


class ScriptedModel(BaseChatModel):
    """Replies with the given messages in order and records what it was shown. No network."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    script: list[AIMessage]
    position: int = 0
    seen: list[list[BaseMessage]] = []
    bound_tools: list[Any] = []

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _generate(
        self, messages: list[BaseMessage], stop: Any = None, run_manager: Any = None, **kwargs: Any
    ) -> ChatResult:
        self.seen.append(list(messages))
        reply = self.script[self.position]
        self.position += 1
        return ChatResult(generations=[ChatGeneration(message=reply)])

    def bind_tools(self, tools: Any, **kwargs: Any) -> ScriptedModel:
        self.bound_tools = list(tools)
        return self


def tool_call(name: str, call_id: str, **args: Any) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


@pytest.fixture
def env(session: Session, fake_chroma: Collection) -> Iterator[Session]:
    engine = session.get_bind()
    assert isinstance(engine, Engine)
    tools.set_engine(engine)
    tools.reset_sessions()
    yield session
    tools.set_engine(None)
    tools.reset_sessions()


def test_run_agent_calls_a_tool_and_returns_the_final_reply(env: Session) -> None:
    model = ScriptedModel(
        script=[
            tool_call("create_task", "c1", title="Call mom", due_date="2026-10-29"),
            AIMessage(content="Added 'Call mom' for tomorrow."),
        ]
    )

    reply = run_agent("add call mom tomorrow", "s-create", model=model)

    assert reply == "Added 'Call mom' for tomorrow."
    [task] = services.list_tasks(env)
    assert (task.title, task.due_date) == ("Call mom", date(2026, 10, 29))
    # The tool result went back to the model before it wrote the reply.
    assert "Created task" in str(model.seen[1][-1].content)


def test_system_prompt_has_todays_date_and_the_rules(env: Session) -> None:
    model = ScriptedModel(script=[AIMessage(content="hi")])

    run_agent("hello", "s-prompt", model=model)

    system = model.seen[0][0]
    assert isinstance(system, SystemMessage)
    assert system.content == build_system_prompt(date.today())
    assert date.today().isoformat() in str(system.content)
    assert "only after the user has confirmed" in str(system.content).replace("\n  ", " ")


def test_the_model_cannot_see_or_fill_session_id(env: Session) -> None:
    model = ScriptedModel(script=[AIMessage(content="hi")])

    run_agent("hello", "s-schema", model=model)

    schemas = [
        convert_to_openai_tool(t)["function"]["parameters"]["properties"] for t in model.bound_tools
    ]
    assert len(schemas) == 5
    for properties in schemas:
        assert "session_id" not in properties
        assert "runtime" not in properties


def test_delete_needs_a_later_turn_through_the_whole_agent(env: Session) -> None:
    task_id = services.create_task(env, TaskCreate(title="Pay rent")).id
    assert task_id is not None
    model = ScriptedModel(
        script=[
            tool_call("delete_task", "d1", target="pay rent"),
            AIMessage(content="Delete 'Pay rent'? Please confirm."),
            tool_call("delete_task", "d2", target="pay rent"),
            AIMessage(content="Deleted 'Pay rent'."),
        ]
    )

    first = run_agent("delete the rent task", "s-delete", model=model)
    env.expire_all()
    assert first == "Delete 'Pay rent'? Please confirm."
    assert services.get_task(env, task_id) is not None  # nothing deleted in the requesting turn

    second = run_agent("yes", "s-delete", model=model)
    env.expire_all()
    assert second == "Deleted 'Pay rent'."
    assert services.get_task(env, task_id) is None
    # History was kept for the session: the second turn saw the first turn's messages.
    assert any("Please confirm" in str(m.content) for m in model.seen[-1])


def test_a_model_that_deletes_twice_in_one_turn_is_refused(env: Session) -> None:
    task_id = services.create_task(env, TaskCreate(title="Pay rent")).id
    assert task_id is not None
    model = ScriptedModel(
        script=[
            tool_call("delete_task", "d1", target="pay rent"),
            tool_call("delete_task", "d2", target="pay rent"),  # ignores the confirmation step
            AIMessage(content="Please confirm."),
        ]
    )

    run_agent("delete the rent task", "s-twice", model=model)

    env.expire_all()
    assert services.get_task(env, task_id) is not None
    assert "already waiting" in str(model.seen[-1][-1].content)


def test_the_conversation_survives_a_restart(env: Session, tmp_path: Path) -> None:
    db_file = tmp_path / "checkpoints.db"

    def new_saver() -> SqliteSaver:
        return SqliteSaver(sqlite3.connect(db_file, check_same_thread=False))

    agent.set_checkpointer(new_saver())
    run_agent("remember the number 42", "s-restart", model=ScriptedModel(script=[AIMessage("ok")]))

    # A restart: a new process opens the same file and has no state in memory.
    agent.set_checkpointer(new_saver())
    model = ScriptedModel(script=[AIMessage("You said 42.")])
    run_agent("what number did I say?", "s-restart", model=model)

    shown = " ".join(str(message.content) for message in model.seen[0])
    assert "remember the number 42" in shown
    assert "what number did I say?" in shown


def test_clear_thread_forgets_one_session_only(env: Session) -> None:
    run_agent("zebra-note", "s-a", model=ScriptedModel(script=[AIMessage("a")]))
    run_agent("walrus-note", "s-b", model=ScriptedModel(script=[AIMessage("b")]))

    agent.clear_thread("s-a")

    again_a = ScriptedModel(script=[AIMessage("fresh")])
    run_agent("third message", "s-a", model=again_a)
    again_b = ScriptedModel(script=[AIMessage("still here")])
    run_agent("fourth message", "s-b", model=again_b)
    assert "zebra-note" not in " ".join(str(m.content) for m in again_a.seen[0])
    assert "walrus-note" in " ".join(str(m.content) for m in again_b.seen[0])
