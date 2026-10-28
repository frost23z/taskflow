"""The search index is a copy of SQLite: it must never make a task operation fail, must filter
before the limit, and must heal itself. Uses the offline word-hash embedding (no downloads)."""

import logging
from typing import Any

import pytest
from chromadb.api.models.Collection import Collection
from sqlmodel import Session

from app import search, services
from app.models import Status, Task
from app.schemas import TaskCreate, TaskUpdate


def make(session: Session, title: str, **fields: Any) -> Task:
    return services.create_task(session, TaskCreate(title=title, **fields))


def break_index(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("chroma is down")

    monkeypatch.setattr(search, "index_task", fail)
    monkeypatch.setattr(search, "remove_task", fail)


# --- a failed index sync must not fail the task operation ----------------------------------------


def test_create_succeeds_when_the_index_is_down_and_search_heals_later(
    session: Session, fake_chroma: Collection, monkeypatch: pytest.MonkeyPatch
) -> None:
    with monkeypatch.context() as broken:
        break_index(broken)
        task = make(session, "Write quarterly report")  # does not raise

    assert task.id is not None
    assert services.get_task(session, task.id) is not None
    assert search.is_stale()
    assert fake_chroma.count() == 0  # the index really missed it

    found = services.find_tasks(session, "quarterly report")  # rebuilds first

    assert [t.id for t, _ in found] == [task.id]
    assert not search.is_stale()


def test_update_and_delete_succeed_when_the_index_is_down(
    session: Session, fake_chroma: Collection, monkeypatch: pytest.MonkeyPatch
) -> None:
    keep = make(session, "Pay rent")
    drop = make(session, "Buy groceries")
    assert keep.id is not None and drop.id is not None

    break_index(monkeypatch)
    updated = services.update_task(session, keep.id, TaskUpdate(title="Pay the rent today"))
    deleted = services.delete_task(session, drop.id)

    assert updated is not None and updated.title == "Pay the rent today"
    assert deleted is True
    assert services.get_task(session, drop.id) is None
    assert search.is_stale()


def test_a_failed_sync_is_logged(
    session: Session,
    fake_chroma: Collection,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    break_index(monkeypatch)
    with caplog.at_level(logging.ERROR, logger="app.services"):
        make(session, "Pay rent")

    assert "Search index update failed" in caplog.text


def test_a_healthy_sync_leaves_the_index_fresh(session: Session, fake_chroma: Collection) -> None:
    make(session, "Pay rent")

    assert not search.is_stale()
    assert fake_chroma.count() == 1


# --- "unfinished" is filtered before the limit ----------------------------------------------------


def test_include_done_false_is_not_crowded_out_by_done_tasks(
    session: Session, fake_chroma: Collection
) -> None:
    for number in range(3):  # exact matches for the query, but finished
        make(session, "documentation guide", description=f"copy {number}", status=Status.DONE)
    open_task = make(session, "documentation api")

    with_done = services.find_tasks(session, "documentation guide", include_done=True, limit=3)
    without_done = services.find_tasks(session, "documentation guide", include_done=False, limit=2)

    assert open_task not in [t for t, _ in with_done]  # the limit really was filled by done tasks
    assert [t.id for t, _ in without_done] == [open_task.id]


def test_include_done_false_never_returns_done_tasks(
    session: Session, fake_chroma: Collection
) -> None:
    make(session, "Write docs", status=Status.DONE)
    make(session, "Write docs for cli")

    found = services.find_tasks(session, "write docs", include_done=False)

    assert found
    assert all(task.status != Status.DONE for task, _ in found)


def test_marking_a_task_done_updates_what_the_search_filters_on(
    session: Session, fake_chroma: Collection
) -> None:
    task = make(session, "Write docs")
    assert task.id is not None
    assert services.find_tasks(session, "write docs", include_done=False)

    services.update_task(session, task.id, TaskUpdate(status=Status.DONE))

    assert services.find_tasks(session, "write docs", include_done=False) == []


# --- reindex_all heals every kind of drift --------------------------------------------------------


def test_reindex_fixes_a_task_edited_behind_the_index(
    session: Session, fake_chroma: Collection
) -> None:
    task = make(session, "Pay rent")
    task.title = "Renew passport"  # edited directly in SQLite, so the index still says "Pay rent"
    session.add(task)
    session.commit()
    assert services.find_tasks(session, "renew passport", limit=1)[0][1] < 0.99

    search.reindex_all(session)

    best, score = services.find_tasks(session, "renew passport", limit=1)[0]
    assert best.id == task.id and score > 0.99


def test_reindex_adds_missing_removes_stale_and_fixes_old_entries(
    session: Session, fake_chroma: Collection
) -> None:
    kept = make(session, "Pay rent")
    missing = make(session, "Buy groceries")
    assert kept.id is not None and missing.id is not None
    fake_chroma.delete(ids=[str(missing.id)])  # never indexed
    fake_chroma.upsert(ids=["999"], documents=["ghost task"])  # deleted from SQLite long ago
    fake_chroma.upsert(  # an entry from before status metadata existed
        ids=[str(kept.id)], documents=[search._document(kept)]
    )

    search.reindex_all(session)

    stored = fake_chroma.get(include=["metadatas"])
    assert set(stored["ids"]) == {str(kept.id), str(missing.id)}
    assert all(meta == {"status": "todo"} for meta in stored["metadatas"] or [])


def test_reindex_does_not_rewrite_entries_that_are_already_correct(
    session: Session, fake_chroma: Collection, monkeypatch: pytest.MonkeyPatch
) -> None:
    make(session, "Pay rent")
    calls: list[Any] = []
    original = fake_chroma.upsert
    monkeypatch.setattr(fake_chroma, "upsert", lambda **kw: calls.append(kw) or original(**kw))

    search.reindex_all(session)

    assert calls == []
