"""API contracts: what clients send to and receive from the task endpoints."""

from datetime import date, datetime
from typing import Any

from pydantic import field_validator
from sqlmodel import SQLModel

from app.models import Priority, Status, TaskBase, Title


class TaskCreate(TaskBase):
    pass


class TaskUpdate(SQLModel):
    title: Title | None = None
    description: str | None = None
    status: Status | None = None
    priority: Priority | None = None
    due_date: date | None = None

    # Only runs for values that were passed, so omitting a field is fine but an
    # explicit null is rejected. due_date is nullable, so null clears it.
    @field_validator("title", "description", "status", "priority")
    @classmethod
    def reject_null(cls, value: Any) -> Any:
        if value is None:
            raise ValueError("may not be null")
        return value


class TaskRead(TaskBase):
    id: int
    created_at: datetime
    updated_at: datetime
