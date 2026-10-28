"""The visible chat history, kept in SQLite so a page refresh or a restart does not lose it."""

from sqlmodel import Session, col, delete, select

from app.models import ChatMessage

HISTORY_LIMIT = 200  # most recent messages sent to the browser


def save_exchange(session: Session, session_id: str, user_message: str, reply: str) -> None:
    session.add(ChatMessage(session_id=session_id, role="user", content=user_message))
    session.add(ChatMessage(session_id=session_id, role="assistant", content=reply))
    session.commit()


def list_history(session: Session, session_id: str) -> list[ChatMessage]:
    """The latest messages of one session, oldest first."""
    statement = (
        select(ChatMessage)
        .where(ChatMessage.session_id == session_id)
        .order_by(col(ChatMessage.id).desc())
        .limit(HISTORY_LIMIT)
    )
    return list(reversed(session.exec(statement).all()))


def delete_history(session: Session, session_id: str) -> None:
    session.exec(delete(ChatMessage).where(col(ChatMessage.session_id) == session_id))
    session.commit()
