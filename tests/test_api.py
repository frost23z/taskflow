import importlib
import tomllib
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from chromadb.api.models.Collection import Collection
from fastapi.testclient import TestClient
from sqlalchemy.engine import Engine
from sqlmodel import Session, create_engine

from app import agent, database, main, services, tools
from app import chat as chat_history
from app.database import get_session
from app.llm import LLMConfigError
from app.models import ChatSessionState, Priority, Status
from app.schemas import TaskCreate

SID = "session-abc"


@pytest.fixture
def client(session: Session, fake_chroma: Collection) -> Iterator[TestClient]:
    """The API on the test database. No `with`, so the startup (init_db/reindex) does not run."""
    main.app.dependency_overrides[get_session] = lambda: session
    yield TestClient(main.app)
    main.app.dependency_overrides.clear()


def test_list_tasks_returns_task_read_objects(client: TestClient, session: Session) -> None:
    services.create_task(
        session, TaskCreate(title="Pay rent", priority=Priority.HIGH, status=Status.IN_PROGRESS)
    )
    services.create_task(session, TaskCreate(title="Buy groceries", description="Milk"))

    response = client.get("/api/tasks")

    assert response.status_code == 200
    tasks = response.json()
    assert [t["title"] for t in tasks] == ["Pay rent", "Buy groceries"]
    first = tasks[0]
    assert (first["priority"], first["status"], first["due_date"]) == ("high", "in_progress", None)
    assert {"id", "created_at", "updated_at", "description"} <= first.keys()


def test_list_tasks_empty(client: TestClient) -> None:
    assert client.get("/api/tasks").json() == []


def test_chat_returns_the_reply_and_the_full_current_task_list(
    client: TestClient, session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    services.create_task(session, TaskCreate(title="Existing task"))
    calls: list[tuple[str, str]] = []

    def fake_run_agent(message: str, session_id: str, model: Any = None) -> str:
        calls.append((message, session_id))
        services.create_task(session, TaskCreate(title="Call mom"))  # what the agent's tool does
        return "Added 'Call mom'."

    monkeypatch.setattr(main, "run_agent", fake_run_agent)

    response = client.post("/api/chat", json={"message": "  add call mom  ", "session_id": SID})

    assert response.status_code == 200
    body = response.json()
    assert body["reply"] == "Added 'Call mom'."
    assert [t["title"] for t in body["tasks"]] == ["Existing task", "Call mom"]
    assert calls == [("add call mom", SID)]  # trimmed message, same session id


@pytest.mark.parametrize(
    "payload",
    [
        {"message": "", "session_id": SID},
        {"message": "    ", "session_id": SID},
        {"message": "x" * 1001, "session_id": SID},
        {"message": "hi", "session_id": ""},
        {"message": "hi", "session_id": "s" * 65},
        {"message": "hi"},
        {"session_id": SID},
    ],
)
def test_invalid_chat_requests_are_rejected_without_calling_the_agent(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, payload: dict[str, str]
) -> None:
    def fail(*args: Any, **kwargs: Any) -> str:
        raise AssertionError("the agent must not run for an invalid request")

    monkeypatch.setattr(main, "run_agent", fail)

    response = client.post("/api/chat", json=payload)

    assert response.status_code == 422
    assert response.json()["detail"]


def test_chat_accepts_the_length_limits(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(main, "run_agent", lambda message, session_id, model=None: "ok")

    response = client.post("/api/chat", json={"message": "x" * 1000, "session_id": "s" * 64})

    assert response.status_code == 200


def test_agent_failure_returns_502_json_without_internals(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(message: str, session_id: str, model: Any = None) -> str:
        raise RuntimeError("secret internal detail: key=abc123")

    monkeypatch.setattr(main, "run_agent", broken)

    response = client.post("/api/chat", json={"message": "hi", "session_id": SID})

    assert response.status_code == 502
    assert response.headers["content-type"] == "application/json"
    assert "unavailable" in response.json()["error"]
    assert "abc123" not in response.text and "Traceback" not in response.text


def test_missing_llm_configuration_is_a_clear_502(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unconfigured(message: str, session_id: str, model: Any = None) -> str:
        raise LLMConfigError("No usable model in LLM_CHAIN: set at least one of GROQ_API_KEY")

    monkeypatch.setattr(main, "run_agent", unconfigured)

    response = client.post("/api/chat", json={"message": "hi", "session_id": SID})

    assert response.status_code == 502
    assert "GROQ_API_KEY" in response.json()["error"]


def test_index_serves_the_page(client: TestClient) -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "TASKFLOW" in response.text and "renderTasks" in response.text
    # Styles come from a compiled stylesheet the app serves itself, not a CDN script.
    assert 'href="/static/styles.css"' in response.text
    assert "cdn." not in response.text


def test_stylesheet_is_served(client: TestClient) -> None:
    response = client.get("/static/styles.css")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/css")
    assert "@container" in response.text  # the adaptive task rows made it into the build


def test_fastapi_entrypoint_points_at_the_app_not_a_route_function() -> None:
    """`fastapi dev` with no path reads this. A function there is treated as an app factory."""
    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    entrypoint = tomllib.loads(pyproject.read_text())["tool"]["fastapi"]["entrypoint"]
    module_name, _, attribute = entrypoint.partition(":")

    assert getattr(importlib.import_module(module_name), attribute) is main.app


def test_startup_initialises_the_database_and_reindexes(monkeypatch: pytest.MonkeyPatch) -> None:
    steps: list[str] = []
    monkeypatch.setattr(main, "init_db", lambda: steps.append("init_db"))
    monkeypatch.setattr(main, "reindex_all", lambda session: steps.append("reindex_all"))

    with TestClient(main.app):  # entering the client runs the lifespan
        assert steps == ["init_db", "reindex_all"]


def test_fresh_setup_creates_the_database_on_startup(
    tmp_path: Path, fake_chroma: Collection, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nothing exists yet (not even the data folder): startup alone must make the API usable."""
    db_path = tmp_path / "not-created-yet" / "tasks.db"
    fresh = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
    monkeypatch.setattr(database, "engine", fresh)
    monkeypatch.setattr(main, "engine", fresh)
    assert not db_path.parent.exists()

    with TestClient(main.app) as client:  # entering the client runs the lifespan
        assert db_path.exists()
        response = client.get("/api/tasks")

    fresh.dispose()
    assert response.status_code == 200
    assert response.json() == []


# --- chat history -------------------------------------------------------------------------------


def history(client: TestClient, session_id: str = SID) -> list[dict[str, Any]]:
    response = client.get("/api/chat/history", params={"session_id": session_id})
    assert response.status_code == 200
    return response.json()


def test_history_is_empty_for_a_new_session(client: TestClient) -> None:
    assert history(client) == []


def test_chat_exchanges_are_stored_and_returned_oldest_first(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    replies = iter(["first reply", "second reply"])
    monkeypatch.setattr(main, "run_agent", lambda message, session_id, model=None: next(replies))

    client.post("/api/chat", json={"message": "one", "session_id": SID})
    client.post("/api/chat", json={"message": "two", "session_id": SID})

    messages = history(client)
    assert [(m["role"], m["content"]) for m in messages] == [
        ("user", "one"),
        ("assistant", "first reply"),
        ("user", "two"),
        ("assistant", "second reply"),
    ]
    assert messages[0]["created_at"]


def test_history_is_separate_per_session(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(main, "run_agent", lambda message, session_id, model=None: "ok")
    client.post("/api/chat", json={"message": "mine", "session_id": "session-a"})
    client.post("/api/chat", json={"message": "theirs", "session_id": "session-b"})

    assert [m["content"] for m in history(client, "session-a")] == ["mine", "ok"]
    assert [m["content"] for m in history(client, "session-b")] == ["theirs", "ok"]


def test_failed_agent_runs_are_not_stored(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(message: str, session_id: str, model: Any = None) -> str:
        raise RuntimeError("boom")

    monkeypatch.setattr(main, "run_agent", broken)
    client.post("/api/chat", json={"message": "hi", "session_id": SID})

    assert history(client) == []


def test_history_returns_only_the_latest_messages(
    client: TestClient, session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(chat_history, "HISTORY_LIMIT", 3)
    for number in range(3):
        chat_history.save_exchange(session, SID, f"q{number}", f"a{number}")

    assert [m["content"] for m in history(client)] == ["a1", "q2", "a2"]


@pytest.mark.parametrize("params", [{}, {"session_id": ""}, {"session_id": "s" * 65}])
def test_history_requires_a_valid_session_id(client: TestClient, params: dict[str, str]) -> None:
    assert client.get("/api/chat/history", params=params).status_code == 422


# --- /clear -------------------------------------------------------------------------------------


@pytest.fixture
def no_agent(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Fails if the agent runs; records which sessions had their agent memory cleared."""
    cleared: list[str] = []

    def fail(*args: Any, **kwargs: Any) -> str:
        raise AssertionError("/clear must not call the agent")

    monkeypatch.setattr(main, "run_agent", fail)
    monkeypatch.setattr(agent, "clear_thread", cleared.append)
    return cleared


@pytest.mark.parametrize("message", ["/clear", "  /CLEAR  ", "/Clear"])
def test_clear_empties_the_chat_without_calling_the_agent(
    client: TestClient, session: Session, no_agent: list[str], message: str
) -> None:
    chat_history.save_exchange(session, SID, "hello", "hi")
    chat_history.save_exchange(session, "other-session", "keep", "me")
    services.create_task(session, TaskCreate(title="Pay rent"))
    session.add(ChatSessionState(session_id=SID, data="{}"))
    session.commit()

    response = client.post("/api/chat", json={"message": message, "session_id": SID})

    assert response.status_code == 200
    body = response.json()
    assert body["cleared"] is True
    assert "Chat cleared" in body["reply"]
    assert [t["title"] for t in body["tasks"]] == ["Pay rent"]  # tasks are never touched
    assert history(client) == []  # and the reply itself is not stored
    assert [m["content"] for m in history(client, "other-session")] == ["keep", "me"]
    assert no_agent == [SID]  # the agent's memory was cleared for this session only
    assert session.get(ChatSessionState, SID) is None


def test_clear_drops_a_pending_delete(
    client: TestClient, session: Session, no_agent: list[str]
) -> None:
    engine = session.get_bind()
    assert isinstance(engine, Engine)
    tools.set_engine(engine)  # tool state is saved to the test database, not the real one
    try:
        tools.get_state(SID).pending_delete = tools.PendingDelete(task_id=1, turn=1)
        tools.save_state(SID)

        client.post("/api/chat", json={"message": "/clear", "session_id": SID})

        tools.reset_sessions()  # a restart: the state must not come back from the database
        assert tools.get_state(SID).pending_delete is None
    finally:
        tools.set_engine(None)
        tools.reset_sessions()


def test_other_messages_are_not_commands(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(main, "run_agent", lambda message, session_id, model=None: "ok")

    for message in ["please /clear", "/clear everything", "clear"]:
        body = client.post("/api/chat", json={"message": message, "session_id": SID}).json()
        assert body["cleared"] is False
        assert body["reply"] == "ok"


def test_a_failing_clear_returns_a_plain_error(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(session: Session, session_id: str) -> None:
        raise RuntimeError("secret internal detail")

    monkeypatch.setattr(chat_history, "clear_session", broken)

    response = client.post("/api/chat", json={"message": "/clear", "session_id": SID})

    assert response.status_code == 500
    assert "Could not clear" in response.json()["error"]
    assert "secret" not in response.text
