"""Task operations. Every write commits to SQLite first, then syncs the search index."""

import logging
from collections.abc import Callable
from datetime import date
from enum import StrEnum

from sqlalchemy import case
from sqlmodel import Session, col, select

from app import search
from app.models import Priority, Status, Task
from app.schemas import TaskCreate, TaskUpdate

logger = logging.getLogger(__name__)


class SortBy(StrEnum):
    ID = "id"
    DUE_DATE = "due_date"
    PRIORITY = "priority"
    CREATED_AT = "created_at"


class SortOrder(StrEnum):
    ASC = "asc"  # soonest due, oldest created, lowest priority first
    DESC = "desc"  # latest due, newest created, highest priority first


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
    session: Session,
    status: Status | None = None,
    priority: Priority | None = None,
    *,
    due_before: date | None = None,
    due_after: date | None = None,
    overdue: bool = False,
    has_due_date: bool | None = None,
    sort_by: SortBy = SortBy.ID,
    order: SortOrder = SortOrder.ASC,
) -> list[Task]:
    """Filter and sort in SQL. due_before and due_after are inclusive; overdue skips done tasks.

    Tasks without a due date come last when sorting by due date, in both directions. Priority is
    sorted by meaning (low < medium < high), not alphabetically. Ties fall back to id.
    """
    statement = select(Task)
    if status is not None:
        statement = statement.where(Task.status == status)
    if priority is not None:
        statement = statement.where(Task.priority == priority)
    if due_before is not None:
        statement = statement.where(col(Task.due_date) <= due_before)
    if due_after is not None:
        statement = statement.where(col(Task.due_date) >= due_after)
    if overdue:
        statement = statement.where(col(Task.due_date) < date.today(), Task.status != Status.DONE)
    if has_due_date is not None:
        no_date = col(Task.due_date).is_(None)
        statement = statement.where(~no_date if has_due_date else no_date)

    descending = order == SortOrder.DESC
    if sort_by == SortBy.DUE_DATE:
        due = col(Task.due_date)
        statement = statement.order_by(due.is_(None), due.desc() if descending else due.asc())
    elif sort_by == SortBy.PRIORITY:
        rank = case(
            (col(Task.priority) == Priority.HIGH, 3),
            (col(Task.priority) == Priority.MEDIUM, 2),
            else_=1,
        )
        statement = statement.order_by(rank.desc() if descending else rank.asc())
    elif sort_by == SortBy.CREATED_AT:
        created = col(Task.created_at)
        statement = statement.order_by(created.desc() if descending else created.asc())
    elif descending:
        statement = statement.order_by(col(Task.id).desc())
    return list(session.exec(statement.order_by(col(Task.id))).all())


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
