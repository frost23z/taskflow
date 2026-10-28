"""Task operations. Every write commits to SQLite first, then syncs the search index."""

import logging
from collections.abc import Callable

from sqlmodel import Session, col, select

from app import search
from app.models import Priority, Status, Task
from app.schemas import TaskCreate, TaskUpdate

logger = logging.getLogger(__name__)


def _sync_index(action: Callable[..., None], *args: object) -> None:
    """Mirror a committed change into the search index without failing the change itself.

    SQLite is the source of truth and is already committed here. If Chroma fails (for example the
    embedding model cannot be downloaded), the user must not be told the task operation failed:
    they would retry and create a duplicate. Mark the index stale instead, and the next search
    rebuilds it from SQLite.
    """
    try:
        action(*args)
    except Exception:
        logger.exception("Search index update failed; it will be rebuilt before the next search")
        search.mark_stale()


def create_task(session: Session, data: TaskCreate) -> Task:
    task = Task.model_validate(data)
    session.add(task)
    session.commit()
    session.refresh(task)
    _sync_index(search.index_task, task)
    return task


def get_task(session: Session, task_id: int) -> Task | None:
    return session.get(Task, task_id)


def list_tasks(
    session: Session, status: Status | None = None, priority: Priority | None = None
) -> list[Task]:
    statement = select(Task).order_by(col(Task.id))
    if status is not None:
        statement = statement.where(Task.status == status)
    if priority is not None:
        statement = statement.where(Task.priority == priority)
    return list(session.exec(statement).all())


def update_task(session: Session, task_id: int, data: TaskUpdate) -> Task | None:
    task = session.get(Task, task_id)
    if task is None:
        return None
    # exclude_unset keeps an explicit due_date=None, which clears the date.
    task.sqlmodel_update(data.model_dump(exclude_unset=True))
    session.add(task)
    session.commit()
    session.refresh(task)
    _sync_index(search.index_task, task)
    return task


def delete_task(session: Session, task_id: int) -> bool:
    task = session.get(Task, task_id)
    if task is None:
        return False
    session.delete(task)
    session.commit()
    _sync_index(search.remove_task, task_id)
    return True


def find_tasks(
    session: Session, query: str, include_done: bool = True, limit: int = 5
) -> list[tuple[Task, float]]:
    """Semantic search, loaded from SQLite, in similarity order (best first)."""
    if search.is_stale():
        try:
            search.reindex_all(session)
        except Exception:
            logger.exception("Could not rebuild the search index; searching the old one")
    hits = search.search(query, limit, include_done)
    if not hits:
        return []
    tasks = session.exec(select(Task).where(col(Task.id).in_([id_ for id_, _ in hits]))).all()
    # The index filtered on status already, but SQLite has the final say.
    by_id = {task.id: task for task in tasks}
    return [
        (by_id[id_], similarity)
        for id_, similarity in hits
        if id_ in by_id and (include_done or by_id[id_].status != Status.DONE)
    ]
