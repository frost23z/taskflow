from datetime import date

from chromadb.api.models.Collection import Collection
from sqlmodel import Session

from app import services
from app.models import Priority, Status
from app.schemas import TaskCreate, TaskUpdate


def indexed_ids(chroma: Collection) -> set[str]:
    return set(chroma.get(include=[])["ids"])


def indexed_text(chroma: Collection, task_id: int) -> str:
    documents = chroma.get(ids=[str(task_id)])["documents"] or []
    return documents[0]


def test_create_task_saves_and_indexes(session: Session, chroma: Collection) -> None:
    task = services.create_task(
        session, TaskCreate(title="Write report", description="Quarterly numbers")
    )

    assert task.id is not None
    assert task.status == Status.TODO
    assert services.get_task(session, task.id) is not None
    assert indexed_ids(chroma) == {str(task.id)}
    assert indexed_text(chroma, task.id) == "Write report\nQuarterly numbers"


def test_get_task_returns_none_when_missing(session: Session, chroma: Collection) -> None:
    assert services.get_task(session, 999) is None


def test_list_tasks_is_ordered_by_id_and_filtered(session: Session, chroma: Collection) -> None:
    first = services.create_task(session, TaskCreate(title="a", priority=Priority.HIGH))
    second = services.create_task(session, TaskCreate(title="b", status=Status.DONE))
    third = services.create_task(
        session, TaskCreate(title="c", status=Status.DONE, priority=Priority.HIGH)
    )

    def ids(**filters) -> list[int | None]:
        return [task.id for task in services.list_tasks(session, **filters)]

    assert ids() == [first.id, second.id, third.id]
    assert ids(status=Status.DONE) == [second.id, third.id]
    assert ids(priority=Priority.HIGH) == [first.id, third.id]
    assert ids(status=Status.DONE, priority=Priority.HIGH) == [third.id]
    assert ids(status=Status.IN_PROGRESS) == []


def test_update_task_changes_only_provided_fields(session: Session, chroma: Collection) -> None:
    task = services.create_task(
        session,
        TaskCreate(title="Old title", description="Keep me", priority=Priority.LOW),
    )
    assert task.id is not None

    updated = services.update_task(session, task.id, TaskUpdate(status=Status.IN_PROGRESS))

    assert updated is not None
    assert updated.status == Status.IN_PROGRESS
    assert updated.title == "Old title"
    assert updated.description == "Keep me"
    assert updated.priority == Priority.LOW


def test_update_task_syncs_the_index(session: Session, chroma: Collection) -> None:
    task = services.create_task(session, TaskCreate(title="Old title", description="Old text"))
    assert task.id is not None

    services.update_task(session, task.id, TaskUpdate(title="New title"))

    assert indexed_ids(chroma) == {str(task.id)}
    assert indexed_text(chroma, task.id) == "New title\nOld text"


def test_update_task_can_set_and_clear_due_date(session: Session, chroma: Collection) -> None:
    task = services.create_task(session, TaskCreate(title="Ship it"))
    assert task.id is not None

    with_date = services.update_task(session, task.id, TaskUpdate(due_date=date(2030, 1, 15)))
    assert with_date is not None
    assert with_date.due_date == date(2030, 1, 15)

    cleared = services.update_task(session, task.id, TaskUpdate(due_date=None))
    assert cleared is not None
    assert cleared.due_date is None

    untouched = services.update_task(session, task.id, TaskUpdate(title="Ship it now"))
    assert untouched is not None
    assert untouched.due_date is None


def test_update_task_returns_none_when_missing(session: Session, chroma: Collection) -> None:
    assert services.update_task(session, 999, TaskUpdate(title="x")) is None
    assert indexed_ids(chroma) == set()


def test_delete_task_removes_row_and_index_entry(session: Session, chroma: Collection) -> None:
    keep = services.create_task(session, TaskCreate(title="keep"))
    gone = services.create_task(session, TaskCreate(title="gone"))
    assert gone.id is not None
    gone_id = gone.id

    assert services.delete_task(session, gone_id) is True

    assert services.get_task(session, gone_id) is None
    assert indexed_ids(chroma) == {str(keep.id)}


def test_delete_task_returns_false_when_missing(session: Session, chroma: Collection) -> None:
    assert services.delete_task(session, 999) is False
