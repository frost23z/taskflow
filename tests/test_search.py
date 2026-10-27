from chromadb.api.models.Collection import Collection
from sqlmodel import Session, select

from app import search, services
from app.models import Status, Task
from app.schemas import TaskCreate
from app.seed import SEED_TASKS, seed


def titles(results: list[tuple[Task, float]]) -> list[str]:
    return [task.title for task, _ in results]


def test_seed_inserts_once_and_indexes_everything(session: Session, chroma: Collection) -> None:
    assert seed(session) == len(SEED_TASKS)
    assert seed(session) == 0

    assert len(session.exec(select(Task)).all()) == len(SEED_TASKS)
    assert chroma.count() == len(SEED_TASKS)


def test_search_returns_similarities_best_first(session: Session, chroma: Collection) -> None:
    seed(session)

    hits = search.search("documentation", limit=5)

    assert len(hits) == 5
    similarities = [similarity for _, similarity in hits]
    assert similarities == sorted(similarities, reverse=True)
    assert all(0 < similarity <= 1 for similarity in similarities)


def test_search_on_empty_index_returns_nothing(chroma: Collection) -> None:
    assert search.search("anything") == []


def test_documentation_finds_the_docs_task(session: Session, chroma: Collection) -> None:
    seed(session)

    results = services.find_tasks(session, "documentation", limit=3)

    assert "Update API documentation" in titles(results)


def test_unfinished_docs_excludes_done_tasks(session: Session, chroma: Collection) -> None:
    seed(session)
    everything = services.find_tasks(session, "unfinished docs", limit=len(SEED_TASKS))
    unfinished = services.find_tasks(
        session, "unfinished docs", include_done=False, limit=len(SEED_TASKS)
    )

    assert "Write README quickstart" in titles(everything)  # a done docs task
    assert all(task.status != Status.DONE for task, _ in unfinished)
    assert "Update API documentation" in titles(unfinished)[:3]


def test_near_duplicate_demo_tasks_are_both_found(session: Session, chroma: Collection) -> None:
    seed(session)

    results = services.find_tasks(session, "demo", limit=5)

    assert {"Finish demo slides", "Record demo video"} <= set(titles(results))


def test_find_tasks_skips_ids_missing_from_sqlite(session: Session, chroma: Collection) -> None:
    kept = services.create_task(session, TaskCreate(title="Pay rent"))
    stale = services.create_task(session, TaskCreate(title="Pay rent online"))
    session.delete(stale)  # removed behind the index's back
    session.commit()
    assert chroma.count() == 2

    results = services.find_tasks(session, "pay rent")

    assert [task.id for task, _ in results] == [kept.id]


def test_reindex_all_adds_missing_and_removes_stale(session: Session, chroma: Collection) -> None:
    kept = services.create_task(session, TaskCreate(title="Kept task"))
    stale = services.create_task(session, TaskCreate(title="Stale task"))
    session.delete(stale)
    session.commit()
    missing = Task.model_validate(TaskCreate(title="Never indexed"))
    session.add(missing)
    session.commit()

    search.reindex_all(session)

    assert set(chroma.get(include=[])["ids"]) == {str(kept.id), str(missing.id)}
