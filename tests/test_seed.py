from chromadb.api.models.Collection import Collection
from sqlmodel import Session, select

from app.models import Priority, Status, Task
from app.seed import DEMO_TASKS, SEED_TASKS, seed


def test_demo_board_is_the_six_tasks_of_the_ui_reference(
    session: Session, fake_chroma: Collection
) -> None:
    assert seed(session, DEMO_TASKS) == 6
    assert seed(session, DEMO_TASKS) == 0  # only fills an empty table

    tasks = session.exec(select(Task)).all()
    assert [t.title for t in tasks] == [
        "Finish AI presentation",
        "Review pull request #82",
        "Send weekly report",
        "Update API documentation",
        "Set up database",
        "Write project README",
    ]
    assert fake_chroma.count() == 6  # the search index matches
    assert [t.status for t in tasks].count(Status.DONE) == 2
    assert {t.priority for t in tasks} == {Priority.HIGH, Priority.MEDIUM, Priority.LOW}


def test_demo_board_leaves_room_to_add_tasks_without_scrolling() -> None:
    # The board holds about seven rows on a laptop screen; the demo adds one and removes one.
    assert len(DEMO_TASKS) <= 6 < len(SEED_TASKS)
