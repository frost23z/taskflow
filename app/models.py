from datetime import UTC, date, datetime
from enum import StrEnum
from typing import Annotated

from pydantic import StringConstraints
from sqlalchemy import Enum
from sqlmodel import Field, SQLModel


class Status(StrEnum):
    TODO = "todo"
    IN_PROGRESS = "in_progress"
    DONE = "done"


class Priority(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


def utcnow() -> datetime:
    return datetime.now(UTC)


def _values(enum_class: type[StrEnum]) -> list[str]:
    # Persist the enum values ("in_progress"), not the member names ("IN_PROGRESS").
    return [member.value for member in enum_class]


Title = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1), Field(max_length=200)
]


class TaskBase(SQLModel):
    title: Title
    description: str = ""
    status: Status = Field(
        default=Status.TODO,
        index=True,
        sa_type=Enum(Status, values_callable=_values, native_enum=False),
    )
    priority: Priority = Field(
        default=Priority.MEDIUM,
        index=True,
        sa_type=Enum(Priority, values_callable=_values, native_enum=False),
    )
    due_date: date | None = Field(default=None, index=True)


class Task(TaskBase, table=True):
    # AUTOINCREMENT stops SQLite from reusing the id of a deleted row.
    __table_args__ = {"sqlite_autoincrement": True}

    id: int | None = Field(default=None, primary_key=True)
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow, sa_column_kwargs={"onupdate": utcnow})


class ChatMessage(SQLModel, table=True):
    """One line of the visible chat. The agent's own memory is kept separately (see app.agent)."""

    __table_args__ = {"sqlite_autoincrement": True}

    id: int | None = Field(default=None, primary_key=True)
    session_id: str = Field(index=True, max_length=64)
    role: str  # "user" or "assistant"
    content: str
    created_at: datetime = Field(default_factory=utcnow)


class ChatSessionState(SQLModel, table=True):
    """The tools' per-session state (candidate ids, pending delete), saved as JSON."""

    session_id: str = Field(primary_key=True, max_length=64)
    data: str
