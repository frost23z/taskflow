"""Demo data: `uv run python -m app.seed` creates the tables and fills them if empty."""

from datetime import date, timedelta

from sqlmodel import Session, select

from app.database import engine, init_db
from app.models import Priority, Status, Task
from app.schemas import TaskCreate
from app.search import reindex_all

TODO, DOING, DONE = Status.TODO, Status.IN_PROGRESS, Status.DONE
LOW, MEDIUM, HIGH = Priority.LOW, Priority.MEDIUM, Priority.HIGH

# (title, description, status, priority, due in days from today; negative = overdue, None = no date)
SEED_TASKS: list[tuple[str, str, Status, Priority, int | None]] = [
    # Dev
    (
        "Fix login redirect bug",
        "Users land on a blank page after logging in via the password reset link.",
        DOING,
        HIGH,
        1,
    ),
    (
        "Add pagination to the orders API",
        "Return 20 orders per page with next/previous cursors.",
        TODO,
        MEDIUM,
        6,
    ),
    (
        "Upgrade Python to 3.14",
        "Bump the runtime in CI and the Dockerfile, then fix deprecation warnings.",
        TODO,
        LOW,
        30,
    ),
    (
        "Review Sam's pull request",
        "Auth middleware refactor, about 400 changed lines.",
        TODO,
        HIGH,
        0,
    ),
    (
        "Set up error monitoring",
        "Connect Sentry to the backend and send alerts to Slack.",
        DONE,
        MEDIUM,
        -5,
    ),
    (
        "Write unit tests for the payment module",
        "Cover refunds, failed cards and currency rounding.",
        DOING,
        MEDIUM,
        4,
    ),
    (
        "Optimize slow dashboard query",
        "The weekly report query takes 12 seconds; add an index and check the query plan.",
        TODO,
        HIGH,
        3,
    ),
    (
        "Rotate production API keys",
        "Replace the keys that were shared in the old wiki page.",
        TODO,
        HIGH,
        -2,
    ),
    # Docs
    (
        "Update API documentation",
        "Keep the documentation in sync with the new orders endpoints.",
        TODO,
        MEDIUM,
        7,
    ),
    (
        "Write README quickstart",
        "Documentation: how to install, configure and run the project.",
        DONE,
        MEDIUM,
        -10,
    ),
    (
        "Draft onboarding guide for new developers",
        "Cover the dev environment setup, branching rules and code review etiquette.",
        TODO,
        LOW,
        21,
    ),
    (
        "Write release notes for v2.3",
        "Summarize features, fixes and breaking changes for customers.",
        DOING,
        MEDIUM,
        2,
    ),
    # Meetings
    (
        "Prepare sprint planning agenda",
        "List candidate stories and capacity for next sprint.",
        TODO,
        MEDIUM,
        1,
    ),
    (
        "Schedule 1:1 with Priya",
        "Monthly check-in; ask about her growth goals and the mentoring idea.",
        TODO,
        LOW,
        5,
    ),
    (
        "Send meeting notes from Monday's design review",
        "Share decisions and action items with the team.",
        DONE,
        LOW,
        -3,
    ),
    (
        "Finish demo slides",
        "Slide deck for the Friday customer demo: problem, solution, roadmap and pricing.",
        DOING,
        HIGH,
        3,
    ),
    (
        "Record demo video",
        "Screen recording of the product walkthrough with voice-over, under five minutes.",
        TODO,
        HIGH,
        4,
    ),
    (
        "Book conference room for retrospective",
        "Need a room for 8 people on Thursday afternoon.",
        DONE,
        LOW,
        -1,
    ),
    # Personal
    (
        "Renew passport",
        "Book an appointment and bring two photos and the old passport.",
        TODO,
        MEDIUM,
        40,
    ),
    (
        "Buy groceries for the week",
        "Vegetables, rice, eggs, olive oil and coffee beans.",
        TODO,
        LOW,
        0,
    ),
    (
        "Book dentist appointment",
        "Routine checkup and cleaning, preferably early morning.",
        TODO,
        LOW,
        14,
    ),
    (
        "Plan weekend hiking trip",
        "Pick a trail, check the weather and reserve a parking spot.",
        TODO,
        LOW,
        9,
    ),
    (
        "Pay electricity bill",
        "Due at the end of the month; pay through the bank app.",
        TODO,
        HIGH,
        2,
    ),
    (
        "Call Mom for her birthday",
        "Order flowers to be delivered in the morning.",
        DONE,
        MEDIUM,
        -7,
    ),
    ("Clean out the garage", "Donate old furniture and sort the boxes of cables.", TODO, LOW, None),
]


def seed(session: Session) -> int:
    """Insert the demo tasks if the table is empty, then sync the search index.

    Returns the number of tasks inserted.
    """
    inserted = 0
    if session.exec(select(Task).limit(1)).first() is None:
        today = date.today()
        for title, description, status, priority, days in SEED_TASKS:
            due_date = None if days is None else today + timedelta(days=days)
            data = TaskCreate(
                title=title,
                description=description,
                status=status,
                priority=priority,
                due_date=due_date,
            )
            session.add(Task.model_validate(data))
        session.commit()
        inserted = len(SEED_TASKS)
    reindex_all(session)
    return inserted


def main() -> None:
    init_db()
    with Session(engine) as session:
        inserted = seed(session)
    print(f"Seeded {inserted} tasks." if inserted else "Tasks already exist; index synced.")


if __name__ == "__main__":
    main()
