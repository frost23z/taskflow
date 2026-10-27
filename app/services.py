"""Task operations. Every write commits to SQLite first, then syncs the search index."""

from sqlmodel import Session, col, select

from app import search
from app.models import Priority, Status, Task
from app.schemas import TaskCreate, TaskUpdate


def create_task(session: Session, data: TaskCreate) -> Task:
    task = Task.model_validate(data)
    session.add(task)
    session.commit()
    session.refresh(task)
    search.index_task(task)
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
    search.index_task(task)
    return task


def delete_task(session: Session, task_id: int) -> bool:
    task = session.get(Task, task_id)
    if task is None:
        return False
    session.delete(task)
    session.commit()
    search.remove_task(task_id)
    return True


def find_tasks(
    session: Session, query: str, include_done: bool = True, limit: int = 5
) -> list[tuple[Task, float]]:
    """Semantic search, loaded from SQLite, in similarity order (best first)."""
    hits = search.search(query, limit)
    if not hits:
        return []
    tasks = session.exec(select(Task).where(col(Task.id).in_([id_ for id_, _ in hits]))).all()
    by_id = {task.id: task for task in tasks}
    return [
        (by_id[id_], similarity)
        for id_, similarity in hits
        if id_ in by_id and (include_done or by_id[id_].status != Status.DONE)
    ]
