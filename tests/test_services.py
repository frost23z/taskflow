from datetime import date, timedelta

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


# --- filtering and sorting -----------------------------------------------------------------------


def make_tasks(session: Session) -> dict[str, int]:
    """Five tasks that exercise every filter and sort. Dates are relative to today."""
    today = date.today()
    rows = [
        ("late", Priority.LOW, Status.TODO, today - timedelta(days=3)),
        ("late-done", Priority.HIGH, Status.DONE, today - timedelta(days=2)),
        ("today", Priority.MEDIUM, Status.IN_PROGRESS, today),
        ("later", Priority.HIGH, Status.TODO, today + timedelta(days=5)),
        ("undated", Priority.MEDIUM, Status.TODO, None),
    ]
    ids = {}
    for title, priority, status, due in rows:
        task = services.create_task(
            session, TaskCreate(title=title, priority=priority, status=status, due_date=due)
        )
        assert task.id is not None
        ids[title] = task.id
    return ids


def listed(session: Session, **options) -> list[str]:
    return [task.title for task in services.list_tasks(session, **options)]


def test_list_tasks_due_date_filters_are_inclusive(session: Session, chroma: Collection) -> None:
    make_tasks(session)
    today = date.today()

    assert listed(session, due_before=today) == ["late", "late-done", "today"]
    assert listed(session, due_after=today) == ["today", "later"]
    assert listed(session, due_after=today, due_before=today) == ["today"]


def test_list_tasks_overdue_ignores_done_and_undated_tasks(
    session: Session, chroma: Collection
) -> None:
    make_tasks(session)

    assert listed(session, overdue=True) == ["late"]


def test_list_tasks_has_due_date(session: Session, chroma: Collection) -> None:
    make_tasks(session)

    assert listed(session, has_due_date=False) == ["undated"]
    assert len(listed(session, has_due_date=True)) == 4


def test_list_tasks_filters_combine(session: Session, chroma: Collection) -> None:
    make_tasks(session)

    assert listed(session, status=Status.TODO, due_before=date.today()) == ["late"]
    assert listed(session, priority=Priority.HIGH, has_due_date=True, overdue=True) == []


def test_list_tasks_sorts_by_due_date_with_undated_last_in_both_directions(
    session: Session, chroma: Collection
) -> None:
    make_tasks(session)

    asc = listed(session, sort_by=services.SortBy.DUE_DATE)
    desc = listed(session, sort_by=services.SortBy.DUE_DATE, order=services.SortOrder.DESC)

    assert asc == ["late", "late-done", "today", "later", "undated"]
    assert desc == ["later", "today", "late-done", "late", "undated"]


def test_list_tasks_sorts_priority_by_meaning_not_alphabet(
    session: Session, chroma: Collection
) -> None:
    make_tasks(session)

    highest_first = listed(session, sort_by=services.SortBy.PRIORITY, order=services.SortOrder.DESC)
    lowest_first = listed(session, sort_by=services.SortBy.PRIORITY)

    # Ties (two high, two medium) keep id order.
    assert highest_first == ["late-done", "later", "today", "undated", "late"]
    assert lowest_first == ["late", "today", "undated", "late-done", "later"]


def test_list_tasks_sorts_by_id_and_created_at(session: Session, chroma: Collection) -> None:
    make_tasks(session)
    desc = services.SortOrder.DESC

    assert listed(session, sort_by=services.SortBy.ID, order=desc) == [
        "undated", "later", "today", "late-done", "late",
    ]  # fmt: skip
    assert listed(session, sort_by=services.SortBy.CREATED_AT)[0] == "late"
    assert listed(session, sort_by=services.SortBy.CREATED_AT, order=desc)[-1] == "late"


def test_list_tasks_filter_and_sort_work_together(session: Session, chroma: Collection) -> None:
    make_tasks(session)

    result = listed(
        session,
        status=Status.TODO,
        has_due_date=True,
        sort_by=services.SortBy.DUE_DATE,
        order=services.SortOrder.DESC,
    )

    assert result == ["later", "late"]
