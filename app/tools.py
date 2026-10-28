"""LangChain tools that let the agent work on tasks through app.services.

Safety rules enforced here in code, not left to the prompt:
- update/delete take a natural-language `target`, resolved with semantic search. No match means
  "not found", several close matches mean a clarification question, one clear match means act.
- A `task_id` is accepted only if it was in a candidate list shown to the same session earlier.
- delete needs confirmation: the first call only records a pending delete, and it runs only when
  called again in the very next user turn AND that user message is a clear yes (checked here, by
  `is_confirmation`, not by the model).
- session_id comes from the run context (ToolRuntime), never from a model-filled argument.
- Tools return strings, including for errors, and never raise to the model.
"""

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from enum import StrEnum
from functools import wraps
from typing import Any

from langchain.tools import ToolRuntime, tool
from pydantic import ValidationError
from sqlalchemy.engine import Engine
from sqlmodel import Session

from app import database, services
from app.models import Priority, Status, Task
from app.schemas import TaskCreate, TaskUpdate

logger = logging.getLogger(__name__)

# Semantic-search tuning. Similarity is 1 - cosine distance (see app.search), best match first.
MIN_SIMILARITY = 0.30  # below this a task does not count as a match at all
AMBIGUITY_MARGIN = 0.05  # matches within this of the best one are all plausible: ask the user
MAX_CANDIDATES = 5  # most candidates shown in a clarification question
MAX_LISTED = 50  # most tasks list_tasks prints
SEARCH_LIMIT = 10  # most tasks search_tasks prints; the reply says when more matched


# --- Run context and per-session state ---------------------------------------------------------


@dataclass
class AgentContext:
    """Passed as `context=` on every agent run; tools read it through ToolRuntime."""

    session_id: str


@dataclass
class PendingDelete:
    task_id: int
    turn: int


@dataclass
class SessionState:
    turn: int = 0
    confirmed: bool = False  # the latest user message was a clear "yes" (see is_confirmation)
    candidate_ids: set[int] = field(default_factory=set)
    pending_delete: PendingDelete | None = None


_sessions: dict[str, SessionState] = {}


def get_state(session_id: str) -> SessionState:
    return _sessions.setdefault(session_id, SessionState())


_YES = re.compile(
    r"\b(yes|y|yep|yeah|yup|sure|ok|okay|confirm|confirmed|proceed|go ahead|do it|delete it)\b"
)
_NO = re.compile(
    r"\b(no|nope|nah|not|never|cancel|stop|wait|abort|nevermind|never mind|do not)\b|n['’]t\b"
)
MAX_CONFIRMATION_LENGTH = 80  # a long message is an instruction, not a reply to "delete it?"


def is_confirmation(message: str) -> bool:
    """True only for a short, clear yes. Questions, negations and long messages never count.

    Deleting is irreversible, so this errs on the side of asking again: "are you sure?", "yes
    but don't", and "no" all return False.
    """
    text = message.strip().lower()
    if not text or len(text) > MAX_CONFIRMATION_LENGTH or "?" in text:
        return False
    return bool(_YES.search(text)) and not _NO.search(text)


def next_turn(session_id: str, message: str = "") -> int:
    """Start a new user turn for this session. run_agent calls this once per message."""
    state = get_state(session_id)
    state.turn += 1
    state.confirmed = is_confirmation(message)
    return state.turn


def reset_sessions() -> None:
    _sessions.clear()


# --- Database access (overridable in tests) ----------------------------------------------------

_engine: Engine | None = None


def set_engine(engine: Engine | None) -> None:
    """Point the tools at another database (None goes back to app.database.engine)."""
    global _engine
    _engine = engine


def _open_session() -> Session:
    # database.engine is looked up at call time so it can also be patched there.
    return Session(_engine or database.engine)


# The Chroma collection is overridden with app.search.set_collection, which services already use.


# --- Input handling and the never-raise guard --------------------------------------------------


class ToolInputError(Exception):
    """Bad input from the model; the message goes back to it as an error string."""


def _guard(func: Callable[..., str]) -> Callable[..., str]:
    """Turn every failure into an error string so a tool never raises to the model."""

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> str:
        try:
            return func(*args, **kwargs)
        except ToolInputError as exc:
            return f"Error: {exc}"
        except ValidationError as exc:
            return f"Error: invalid input: {_describe_validation_error(exc)}"
        except Exception:
            logger.exception("Tool %s failed", func.__name__)
            return "Error: the task operation failed unexpectedly. Tell the user it did not work."

    return wrapper


def _describe_validation_error(exc: ValidationError) -> str:
    return "; ".join(
        f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}" for error in exc.errors()
    )


def _parse_enum[E: StrEnum](enum_class: type[E], value: str | None, name: str) -> E | None:
    if value is None:
        return None
    try:
        return enum_class(value.strip().lower())
    except ValueError:
        allowed = ", ".join(member.value for member in enum_class)
        raise ToolInputError(f"{name} must be one of: {allowed} (got {value!r})") from None


def _parse_date(value: str | None, name: str = "due_date") -> date | None:
    if value is None:
        return None
    try:
        return date.fromisoformat(value.strip())
    except ValueError:
        raise ToolInputError(
            f"{name} must be an ISO date like 2026-10-31 (got {value!r})"
        ) from None


def _line(task: Task) -> str:
    due = task.due_date.isoformat() if task.due_date else "no due date"
    return f"[{task.id}] {task.title} ({task.status}, {task.priority} priority, {due})"


# --- Resolving a natural-language target -------------------------------------------------------


@dataclass
class Resolution:
    """Either the task to act on, or the message to send back to the model."""

    task: Task | None = None
    message: str = ""


def _resolve(session: Session, state: SessionState, target: str, task_id: int | None) -> Resolution:
    if task_id is not None:
        if task_id not in state.candidate_ids:
            return Resolution(
                message=(
                    f"Error: task_id {task_id} was not offered in this conversation. "
                    "Describe the task in `target` instead of guessing an ID."
                )
            )
        task = services.get_task(session, task_id)
        if task is None:
            state.candidate_ids.discard(task_id)
            return Resolution(message=f"Not found: task {task_id} no longer exists.")
        return Resolution(task=task)

    if not target.strip():
        return Resolution(message="Error: describe the task in `target`.")

    found = services.find_tasks(session, target, limit=MAX_CANDIDATES)
    matches = [(task, score) for task, score in found if score >= MIN_SIMILARITY]
    if not matches:
        return Resolution(message=f"Not found: no task matches {target!r}.")

    best = matches[0][1]
    plausible = [task for task, score in matches if best - score <= AMBIGUITY_MARGIN]
    if len(plausible) == 1:
        return Resolution(task=plausible[0])

    state.candidate_ids.update(task.id for task in plausible if task.id is not None)
    listing = "\n".join(_line(task) for task in plausible)
    return Resolution(
        message=(
            f"Several tasks match {target!r}. Do not act yet. Ask the user which one they mean:\n"
            f"{listing}\n"
            "After they choose, call the tool again with that task's task_id."
        )
    )


# --- Tools -------------------------------------------------------------------------------------


@tool
@_guard
def create_task(
    title: str,
    description: str = "",
    priority: str = "medium",
    status: str = "todo",
    due_date: str | None = None,
) -> str:
    """Create a new task.

    Args:
        title: Short task title.
        description: Optional longer details.
        priority: One of low, medium, high.
        status: One of todo, in_progress, done.
        due_date: Optional due date as an ISO string, e.g. 2026-10-31. Resolve words like
            "tomorrow" or "Friday" to a date first.
    """
    data = TaskCreate(
        title=title,
        description=description,
        priority=_parse_enum(Priority, priority, "priority") or Priority.MEDIUM,
        status=_parse_enum(Status, status, "status") or Status.TODO,
        due_date=_parse_date(due_date),
    )
    with _open_session() as session:
        task = services.create_task(session, data)
        return f"Created task {_line(task)}"


@tool
@_guard
def list_tasks(
    status: str | None = None,
    priority: str | None = None,
    due_before: str | None = None,
    due_after: str | None = None,
    overdue: bool = False,
    has_due_date: bool | None = None,
    sort_by: str = "id",
    order: str = "asc",
) -> str:
    """List tasks, filtered and sorted. Use search_tasks to find tasks by topic instead.

    Args:
        status: Optional filter: todo, in_progress or done.
        priority: Optional filter: low, medium or high.
        due_before: Only tasks due on or before this ISO date (e.g. "by Friday").
        due_after: Only tasks due on or after this ISO date.
        overdue: True for tasks past their due date that are not done.
        has_due_date: True for tasks with a due date, false for tasks without one.
        sort_by: id (default), due_date, priority or created_at.
        order: asc or desc. asc = soonest due, oldest created or lowest priority first;
            desc = latest due, newest created or highest priority first. For "highest
            priority first" use sort_by=priority, order=desc. Tasks without a due date come
            last when sorting by due date.
    """
    wanted_status = _parse_enum(Status, status, "status")
    wanted_priority = _parse_enum(Priority, priority, "priority")
    wanted_sort = _parse_enum(services.SortBy, sort_by, "sort_by") or services.SortBy.ID
    wanted_order = _parse_enum(services.SortOrder, order, "order") or services.SortOrder.ASC
    with _open_session() as session:
        tasks = services.list_tasks(
            session,
            wanted_status,
            wanted_priority,
            due_before=_parse_date(due_before, "due_before"),
            due_after=_parse_date(due_after, "due_after"),
            overdue=overdue,
            has_due_date=has_due_date,
            sort_by=wanted_sort,
            order=wanted_order,
        )
        if not tasks:
            return "No tasks found."
        lines = [_line(task) for task in tasks[:MAX_LISTED]]
        if len(tasks) > MAX_LISTED:
            lines.append(f"... and {len(tasks) - MAX_LISTED} more (narrow the filter to see them).")
        return "\n".join(lines)


@tool
@_guard
def search_tasks(query: str, include_done: bool = True) -> str:
    """Find tasks related to a topic or phrase (semantic search), best match first.

    Args:
        query: What to look for, e.g. "documentation" or "things about the demo".
        include_done: Set false to leave out tasks that are already done.
    """
    if not query.strip():
        raise ToolInputError("query must not be empty")
    with _open_session() as session:
        # One extra, so the reply can say when the list was cut short.
        found = services.find_tasks(
            session, query, include_done=include_done, limit=SEARCH_LIMIT + 1
        )
        matches = [(task, score) for task, score in found if score >= MIN_SIMILARITY]
        if not matches:
            return f"No tasks match {query!r}."
        lines = [f"{_line(task)} - match {score:.2f}" for task, score in matches[:SEARCH_LIMIT]]
        if len(matches) > SEARCH_LIMIT:
            lines.append(
                f"... more tasks match; these are the best {SEARCH_LIMIT}. "
                "Tell the user the list is not complete and suggest a narrower search."
            )
        return "\n".join(lines)


@tool
@_guard
def update_task(
    target: str,
    runtime: ToolRuntime[AgentContext],
    task_id: int | None = None,
    title: str | None = None,
    description: str | None = None,
    status: str | None = None,
    priority: str | None = None,
    due_date: str | None = None,
    clear_due_date: bool = False,
) -> str:
    """Change an existing task. Also use this to mark a task done (status=done).

    The task is found from `target`. If several tasks match, nothing is changed and the result
    lists candidates: ask the user which one, then call again with that task_id.

    Args:
        target: Natural-language description of the task to change, e.g. "the report task".
        task_id: Only an ID taken from a candidate list this tool returned earlier. Never guess.
        title: New title.
        description: New description.
        status: New status: todo, in_progress or done.
        priority: New priority: low, medium or high.
        due_date: New due date as an ISO string, e.g. 2026-10-31.
        clear_due_date: Set true to remove the due date.
    """
    changes: dict[str, Any] = {}
    if title is not None:
        changes["title"] = title
    if description is not None:
        changes["description"] = description
    if (new_status := _parse_enum(Status, status, "status")) is not None:
        changes["status"] = new_status
    if (new_priority := _parse_enum(Priority, priority, "priority")) is not None:
        changes["priority"] = new_priority
    if due_date is not None and clear_due_date:
        raise ToolInputError("give either due_date or clear_due_date, not both")
    if (new_due := _parse_date(due_date)) is not None:
        changes["due_date"] = new_due
    if clear_due_date:
        changes["due_date"] = None
    if not changes:
        raise ToolInputError("nothing to change: pass at least one field to update")
    update = TaskUpdate(**changes)  # validates before anything is looked up

    state = get_state(runtime.context.session_id)
    with _open_session() as session:
        resolution = _resolve(session, state, target, task_id)
        if resolution.task is None or resolution.task.id is None:
            return resolution.message
        task = services.update_task(session, resolution.task.id, update)
        if task is None:
            return f"Not found: task {resolution.task.id} no longer exists."
        return f"Updated task {_line(task)}"


@tool
@_guard
def delete_task(
    target: str,
    runtime: ToolRuntime[AgentContext],
    task_id: int | None = None,
) -> str:
    """Delete a task. Needs the user's confirmation, so it takes two steps.

    The first call only asks for confirmation and deletes nothing. Ask the user, and call this
    again with the same task only after the user confirms in their NEXT message.

    Args:
        target: Natural-language description of the task to delete.
        task_id: Only an ID taken from a candidate list this tool returned earlier. Never guess.
    """
    state = get_state(runtime.context.session_id)
    with _open_session() as session:
        resolution = _resolve(session, state, target, task_id)
        task = resolution.task
        if task is None or task.id is None:
            return resolution.message

        pending = state.pending_delete
        if pending is not None and pending.task_id == task.id:
            if pending.turn == state.turn:
                return (
                    f"Deleting {_line(task)} is already waiting for the user's confirmation. "
                    "Ask the user and wait for their reply; do not call delete_task again now."
                )
            if pending.turn + 1 == state.turn:
                state.pending_delete = None
                if not state.confirmed:
                    return (
                        f"Not confirmed: the user's reply was not a clear yes, so {_line(task)} "
                        "was NOT deleted. Tell the user it was kept. Ask again only if they "
                        "still want it deleted."
                    )
                if services.delete_task(session, task.id):
                    return f"Deleted task [{task.id}] {task.title}"
                return f"Not found: task {task.id} no longer exists."

        # No pending delete for this task, or it is stale: record a fresh one and ask.
        state.pending_delete = PendingDelete(task.id, state.turn)
        state.candidate_ids.add(task.id)
        return (
            f"Confirmation needed. Nothing was deleted. Ask the user: delete {_line(task)}? "
            "This cannot be undone. Only call delete_task again after they say yes in their "
            f"next message (use task_id={task.id})."
        )


TOOLS = [create_task, list_tasks, search_tasks, update_task, delete_task]
