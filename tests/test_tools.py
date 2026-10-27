from collections.abc import Iterator
from datetime import date
from typing import Any

import pytest
from chromadb.api.models.Collection import Collection
from langchain.tools import ToolRuntime
from sqlalchemy.engine import Engine
from sqlmodel import Session

from app import services, tools
from app.models import Priority, Status
from app.schemas import TaskCreate

SID = "session-1"


@pytest.fixture
def env(session: Session, fake_chroma: Collection) -> Iterator[Session]:
    """Tools pointed at the test database and the offline Chroma collection."""
    engine = session.get_bind()
    assert isinstance(engine, Engine)
    tools.set_engine(engine)
    tools.reset_sessions()
    yield session
    tools.set_engine(None)
    tools.reset_sessions()


def runtime(session_id: str = SID) -> ToolRuntime[tools.AgentContext]:
    return ToolRuntime(
        state={},
        context=tools.AgentContext(session_id),
        config={},
        stream_writer=lambda _: None,
        tool_call_id="call-1",
        store=None,
    )


def call(tool: Any, session_id: str = SID, **args: Any) -> str:
    """Run a tool's function the way the agent does, with the session in the run context."""
    if "runtime" in tool.args_schema.model_fields or tool.name in {"update_task", "delete_task"}:
        args["runtime"] = runtime(session_id)
    return tool.func(**args)


def new_task(session: Session, title: str, **fields: Any) -> int:
    task = services.create_task(session, TaskCreate(title=title, **fields))
    assert task.id is not None
    return task.id


def titles(session: Session) -> list[str]:
    return [task.title for task in services.list_tasks(session)]


# --- create / list / search ---------------------------------------------------------------------


def test_create_task_saves_with_validated_fields(env: Session) -> None:
    result = call(
        tools.create_task,
        title="Buy groceries",
        priority="HIGH",
        due_date="2026-11-02",
    )

    assert result.startswith("Created task [")
    [task] = services.list_tasks(env)
    assert (task.title, task.priority, task.status) == ("Buy groceries", Priority.HIGH, Status.TODO)
    assert task.due_date == date(2026, 11, 2)


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        ({"title": "x", "due_date": "next friday"}, "due_date must be an ISO date"),
        ({"title": "x", "priority": "urgent"}, "priority must be one of: low, medium, high"),
        ({"title": "x", "status": "finished"}, "status must be one of: todo, in_progress, done"),
        ({"title": "   "}, "invalid input"),
    ],
)
def test_invalid_create_input_is_returned_as_an_error_string(
    env: Session, args: dict[str, str], expected: str
) -> None:
    result = call(tools.create_task, **args)

    assert result.startswith("Error:")
    assert expected in result
    assert services.list_tasks(env) == []


def test_list_tasks_filters_and_reports_empty(env: Session) -> None:
    new_task(env, "Write report", status=Status.DONE)
    new_task(env, "Pay rent", priority=Priority.HIGH)

    assert "Pay rent" in call(tools.list_tasks, status="todo")
    assert "Write report" not in call(tools.list_tasks, status="todo")
    assert call(tools.list_tasks, priority="low") == "No tasks found."
    assert call(tools.list_tasks, status="later").startswith("Error: status must be one of")


def test_search_tasks_respects_include_done_and_minimum_similarity(env: Session) -> None:
    new_task(env, "Write documentation", status=Status.DONE)
    new_task(env, "Buy groceries")

    assert "Write documentation" in call(tools.search_tasks, query="documentation")
    assert "No tasks match" in call(tools.search_tasks, query="documentation", include_done=False)
    assert "No tasks match" in call(tools.search_tasks, query="xylophone")


# --- update / delete target resolution ----------------------------------------------------------


def test_update_with_an_exact_match_acts(env: Session) -> None:
    task_id = new_task(env, "Write quarterly report")
    new_task(env, "Buy groceries")

    result = call(tools.update_task, target="write quarterly report", status="done")

    assert result.startswith("Updated task")
    env.expire_all()
    task = services.get_task(env, task_id)
    assert task is not None and task.status == Status.DONE


def test_update_with_no_match_returns_not_found_and_changes_nothing(env: Session) -> None:
    new_task(env, "Buy groceries")

    result = call(tools.update_task, target="xylophone lesson", status="done")

    assert result.startswith("Not found")
    env.expire_all()
    assert [t.status for t in services.list_tasks(env)] == [Status.TODO]


def test_ambiguous_match_asks_for_clarification_and_does_not_act(env: Session) -> None:
    api = new_task(env, "Write documentation for API")
    cli = new_task(env, "Write documentation for CLI")
    new_task(env, "Buy groceries")

    result = call(tools.update_task, target="write documentation", priority="high")

    assert "Ask the user which one" in result
    assert f"[{api}]" in result and f"[{cli}]" in result
    assert "Buy groceries" not in result
    env.expire_all()
    assert {t.priority for t in services.list_tasks(env)} == {Priority.MEDIUM}
    assert tools.get_state(SID).candidate_ids == {api, cli}


def test_task_id_from_a_candidate_list_is_accepted(env: Session) -> None:
    api = new_task(env, "Write documentation for API")
    new_task(env, "Write documentation for CLI")
    call(tools.update_task, target="write documentation", priority="high")  # offers candidates

    result = call(tools.update_task, target="write documentation", task_id=api, priority="high")

    assert result.startswith("Updated task")
    env.expire_all()
    task = services.get_task(env, api)
    assert task is not None and task.priority == Priority.HIGH


def test_invented_task_id_is_rejected(env: Session) -> None:
    real = new_task(env, "Write documentation for API")
    new_task(env, "Write documentation for CLI")

    # Never offered in this session, even though the task exists.
    result = call(tools.update_task, target="docs", task_id=real, priority="high")

    assert result.startswith("Error: task_id")
    assert "Describe the task" in result
    env.expire_all()
    task = services.get_task(env, real)
    assert task is not None and task.priority == Priority.MEDIUM


def test_candidate_ids_belong_to_their_own_session(env: Session) -> None:
    api = new_task(env, "Write documentation for API")
    new_task(env, "Write documentation for CLI")
    call(tools.update_task, "session-a", target="write documentation", priority="high")

    result = call(tools.update_task, "session-b", target="docs", task_id=api, priority="high")

    assert result.startswith("Error: task_id")


def test_update_can_clear_a_due_date_and_rejects_empty_changes(env: Session) -> None:
    task_id = new_task(env, "Pay rent", due_date=date(2026, 11, 1))

    assert call(tools.update_task, target="pay rent").startswith("Error: nothing to change")
    assert call(tools.update_task, target="pay rent", clear_due_date=True).startswith("Updated")
    assert call(tools.update_task, target="pay rent", due_date="soon").startswith("Error:")
    env.expire_all()
    task = services.get_task(env, task_id)
    assert task is not None and task.due_date is None


# --- delete confirmation ------------------------------------------------------------------------


def test_delete_is_refused_in_the_turn_it_was_requested(env: Session) -> None:
    task_id = new_task(env, "Pay rent")
    tools.next_turn(SID)

    first = call(tools.delete_task, target="pay rent")
    second = call(tools.delete_task, target="pay rent")  # model tries again, same user turn

    assert "Confirmation needed" in first
    assert "already waiting" in second
    assert services.get_task(env, task_id) is not None


def test_delete_runs_when_called_again_in_the_next_turn(env: Session) -> None:
    task_id = new_task(env, "Pay rent")
    tools.next_turn(SID)
    call(tools.delete_task, target="pay rent")

    tools.next_turn(SID)  # the user confirmed
    result = call(tools.delete_task, target="pay rent")

    assert result.startswith("Deleted task")
    env.expire_all()
    assert services.get_task(env, task_id) is None


def test_delete_confirmation_works_with_the_task_id_from_the_question(env: Session) -> None:
    task_id = new_task(env, "Pay rent")
    tools.next_turn(SID)
    call(tools.delete_task, target="pay rent")

    tools.next_turn(SID)
    result = call(tools.delete_task, target="that task", task_id=task_id)

    assert result.startswith("Deleted task")


def test_a_stale_pending_delete_does_not_authorise_a_later_delete(env: Session) -> None:
    task_id = new_task(env, "Pay rent")
    tools.next_turn(SID)
    call(tools.delete_task, target="pay rent")
    tools.next_turn(SID)  # the user talked about something else
    tools.next_turn(SID)

    result = call(tools.delete_task, target="pay rent")

    assert "Confirmation needed" in result
    assert services.get_task(env, task_id) is not None


def test_confirming_a_different_task_does_not_delete_it(env: Session) -> None:
    rent = new_task(env, "Pay rent")
    groceries = new_task(env, "Buy groceries")
    tools.next_turn(SID)
    call(tools.delete_task, target="pay rent")

    tools.next_turn(SID)
    result = call(tools.delete_task, target="buy groceries")

    assert "Confirmation needed" in result
    assert services.get_task(env, rent) is not None
    assert services.get_task(env, groceries) is not None


def test_delete_with_ambiguous_target_only_lists_candidates(env: Session) -> None:
    new_task(env, "Write documentation for API")
    new_task(env, "Write documentation for CLI")
    tools.next_turn(SID)

    result = call(tools.delete_task, target="write documentation")

    assert "Ask the user which one" in result
    assert tools.get_state(SID).pending_delete is None
    assert len(titles(env)) == 2


# --- never raise --------------------------------------------------------------------------------


def test_unexpected_failures_become_error_strings(env: Session) -> None:
    new_task(env, "Pay rent")  # indexed, so update_task gets as far as the database

    class BrokenEngine:
        def __getattr__(self, name: str) -> Any:
            raise RuntimeError("database is down")

    tools.set_engine(BrokenEngine())  # type: ignore[arg-type]

    assert call(tools.list_tasks).startswith("Error:")
    assert call(tools.create_task, title="x").startswith("Error:")
    assert call(tools.update_task, target="pay rent", status="done").startswith("Error:")
