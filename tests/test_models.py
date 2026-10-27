import pytest
from pydantic import ValidationError
from sqlalchemy import text
from sqlmodel import Session

from app.models import Priority, Status, Task
from app.schemas import TaskCreate


def add_task(session: Session, **fields) -> Task:
    task = Task.model_validate(TaskCreate(**fields))
    session.add(task)
    session.commit()
    session.refresh(task)
    return task


def test_defaults_on_create(session) -> None:
    task = add_task(session, title="Write tests")

    assert task.id is not None
    assert task.description == ""
    assert task.status == Status.TODO
    assert task.priority == Priority.MEDIUM
    assert task.due_date is None
    assert task.created_at is not None
    assert task.updated_at is not None


def test_status_is_stored_as_value(session) -> None:
    task = add_task(session, title="Write tests", status=Status.IN_PROGRESS)

    stored = session.connection().execute(text("SELECT status FROM task")).scalar_one()

    assert stored == "in_progress"
    assert task.status is Status.IN_PROGRESS


def test_id_is_not_reused_after_delete(session) -> None:
    add_task(session, title="first")
    second = add_task(session, title="second")
    deleted_id = second.id

    session.delete(second)
    session.commit()
    third = add_task(session, title="third")

    assert third.id is not None
    assert deleted_id is not None
    assert third.id > deleted_id


@pytest.mark.parametrize("title", ["", "   "])
def test_empty_title_is_rejected(title) -> None:
    with pytest.raises(ValidationError):
        TaskCreate(title=title)
