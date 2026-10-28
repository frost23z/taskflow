from collections.abc import Iterator
from datetime import date, timedelta
from typing import Any

import pytest
from chromadb.api.models.Collection import Collection
from langchain.tools import ToolRuntime
from sqlalchemy.engine import Engine
from sqlmodel import Session

from app import services, tools
from app.models import ChatSessionState, Priority, Status
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


def test_list_tasks_filters_by_dates_and_overdue(env: Session) -> None:
    today = date.today()
    new_task(env, "Old", due_date=today - timedelta(days=2))
    new_task(env, "Old done", status=Status.DONE, due_date=today - timedelta(days=2))
    new_task(env, "Soon", due_date=today + timedelta(days=2))
    new_task(env, "Whenever")

    overdue = call(tools.list_tasks, overdue=True)
    assert "Old" in overdue and "Old done" not in overdue and "Soon" not in overdue

    by_friday = call(tools.list_tasks, due_before=(today + timedelta(days=2)).isoformat())
    assert "Soon" in by_friday and "Whenever" not in by_friday

    after = call(tools.list_tasks, due_after=today.isoformat())
    assert "Soon" in after and "Old" not in after

    assert call(tools.list_tasks, has_due_date=False).count("[") == 1
    assert "Whenever" in call(tools.list_tasks, has_due_date=False)


def test_list_tasks_sorts(env: Session) -> None:
    today = date.today()
    new_task(env, "Low later", priority=Priority.LOW, due_date=today + timedelta(days=9))
    new_task(env, "High soon", priority=Priority.HIGH, due_date=today + timedelta(days=1))
    new_task(env, "Medium undated")

    def order(**args: Any) -> list[str]:
        lines = call(tools.list_tasks, **args).splitlines()
        return [line.split("] ", 1)[1].split(" (")[0] for line in lines]

    assert order(sort_by="due_date") == ["High soon", "Low later", "Medium undated"]
    assert order(sort_by="due_date", order="desc") == ["Low later", "High soon", "Medium undated"]
    assert order(sort_by="priority", order="desc") == ["High soon", "Medium undated", "Low later"]
    assert order(sort_by="PRIORITY", order="ASC")[0] == "Low later"  # case-insensitive


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        ({"due_before": "next friday"}, "Error: due_before must be an ISO date"),
        ({"due_after": "2026-13-45"}, "Error: due_after must be an ISO date"),
        ({"sort_by": "title"}, "Error: sort_by must be one of: id, due_date, priority, created_at"),
        ({"order": "up"}, "Error: order must be one of: asc, desc"),
    ],
)
def test_list_tasks_rejects_bad_filter_and_sort_input(
    env: Session, args: dict[str, str], expected: str
) -> None:
    new_task(env, "Anything")

    assert call(tools.list_tasks, **args).startswith(expected)


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


# --- search limit -----------------------------------------------------------------------------


def test_search_tasks_says_when_the_list_is_cut_short(env: Session) -> None:
    for number in range(tools.SEARCH_LIMIT + 2):
        new_task(env, f"docs page {number}")

    result = call(tools.search_tasks, query="docs")

    lines = result.splitlines()
    assert len(lines) == tools.SEARCH_LIMIT + 1
    assert "not complete" in lines[-1]


def test_search_tasks_has_no_cut_off_note_when_everything_fits(env: Session) -> None:
    new_task(env, "docs page one")
    new_task(env, "docs page two")

    result = call(tools.search_tasks, query="docs")

    assert len(result.splitlines()) == 2
    assert "not complete" not in result


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

    tools.next_turn(SID, "yes")  # the user confirmed
    result = call(tools.delete_task, target="pay rent")

    assert result.startswith("Deleted task")
    env.expire_all()
    assert services.get_task(env, task_id) is None


def test_delete_confirmation_works_with_the_task_id_from_the_question(env: Session) -> None:
    task_id = new_task(env, "Pay rent")
    tools.next_turn(SID)
    call(tools.delete_task, target="pay rent")

    tools.next_turn(SID, "yes please")
    result = call(tools.delete_task, target="that task", task_id=task_id)

    assert result.startswith("Deleted task")


@pytest.mark.parametrize(
    "reply", ["no", "nope", "cancel", "keep it", "are you sure?", "yes but don't", "not sure", ""]
)
def test_delete_is_not_run_when_the_users_reply_is_not_a_clear_yes(
    env: Session, reply: str
) -> None:
    task_id = new_task(env, "Pay rent")
    tools.next_turn(SID, "delete the rent task")
    call(tools.delete_task, target="pay rent")

    tools.next_turn(SID, reply)  # the model calls delete_task again anyway
    result = call(tools.delete_task, target="pay rent")

    assert result.startswith("Not confirmed")
    env.expire_all()
    assert services.get_task(env, task_id) is not None
    assert tools.get_state(SID).pending_delete is None


def test_a_refused_delete_needs_a_fresh_confirmation_later(env: Session) -> None:
    task_id = new_task(env, "Pay rent")
    tools.next_turn(SID, "delete the rent task")
    call(tools.delete_task, target="pay rent")
    tools.next_turn(SID, "no")
    call(tools.delete_task, target="pay rent")

    tools.next_turn(SID, "yes")  # a late yes does not revive the old request
    result = call(tools.delete_task, target="pay rent")

    assert "Confirmation needed" in result
    assert services.get_task(env, task_id) is not None


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("yes", True),
        ("Yes, please delete it", True),
        ("Y", True),
        ("ok", True),
        ("go ahead", True),
        ("yes and also mark the passport task as done", True),
        ("no", False),
        ("yes, I'd say no", False),
        ("yes but don\u2019t", False),
        ("are you sure?", False),
        ("the video one", False),
        ("yes " + "x" * 80, False),
    ],
)
def test_is_confirmation(message: str, expected: bool) -> None:
    assert tools.is_confirmation(message) is expected


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

    tools.next_turn(SID, "yes")
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


# --- session state survives a restart -----------------------------------------------------------


def restart() -> None:
    """What a server restart does to the tools: the in-memory state is gone."""
    tools.reset_sessions()


def test_offered_candidates_survive_a_restart(env: Session) -> None:
    first = new_task(env, "Write demo script")
    new_task(env, "Write demo slides")
    tools.next_turn(SID, "delete the demo")
    assert "Several tasks match" in call(tools.delete_task, target="write demo")
    tools.save_state(SID)

    restart()
    tools.next_turn(SID, "the first one")
    result = call(tools.delete_task, target="write demo", task_id=first)

    assert result.startswith("Confirmation needed")  # the ID was still known as a candidate


def test_a_pending_delete_survives_a_restart(env: Session) -> None:
    new_task(env, "Pay rent")
    tools.next_turn(SID, "delete the rent task")
    assert call(tools.delete_task, target="rent").startswith("Confirmation needed")
    tools.save_state(SID)

    restart()
    tools.next_turn(SID, "yes")

    assert call(tools.delete_task, target="rent").startswith("Deleted task")
    assert titles(env) == []


def test_unsaved_state_is_lost_and_an_invented_id_is_still_rejected(env: Session) -> None:
    task_id = new_task(env, "Pay rent")
    tools.next_turn(SID, "hi")

    restart()
    result = call(tools.delete_task, target="rent", task_id=task_id)

    assert result.startswith("Error: task_id")


def test_saved_state_belongs_to_its_own_session(env: Session) -> None:
    new_task(env, "Pay rent")
    tools.next_turn("session-a", "delete the rent task")
    call(tools.delete_task, session_id="session-a", target="rent")
    tools.save_state("session-a")

    restart()

    assert tools.get_state("session-a").pending_delete is not None
    assert tools.get_state("session-b").pending_delete is None
    assert tools.get_state("session-b").turn == 0


def test_clear_state_forgets_the_session_everywhere(env: Session) -> None:
    new_task(env, "Pay rent")
    tools.next_turn(SID, "delete the rent task")
    call(tools.delete_task, target="rent")
    tools.save_state(SID)

    tools.clear_state(env, SID)
    restart()

    state = tools.get_state(SID)
    assert (state.turn, state.pending_delete, state.candidate_ids) == (0, None, set())


def test_unreadable_saved_state_starts_fresh_instead_of_failing(env: Session) -> None:
    env.add(ChatSessionState(session_id=SID, data="not json"))
    env.commit()

    assert tools.get_state(SID).turn == 0
