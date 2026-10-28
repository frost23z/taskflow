"""TaskFlow API: the task list, the chat endpoint that drives the agent, and the web page."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, StringConstraints
from sqlmodel import Session

from app import chat as chat_history
from app import services
from app.agent import run_agent
from app.database import engine, get_session, init_db
from app.llm import LLMConfigError
from app.schemas import ChatMessageRead, TaskRead
from app.search import reindex_all

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

SessionDep = Annotated[Session, Depends(get_session)]
SessionId = Annotated[str, Query(min_length=1, max_length=64)]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    init_db()
    with Session(engine) as session:
        reindex_all(session)
    yield


app = FastAPI(title="TaskFlow", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class ChatRequest(BaseModel):
    message: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=1000)]
    session_id: str = Field(min_length=1, max_length=64)


class ChatResponse(BaseModel):
    reply: str
    tasks: list[TaskRead]
    cleared: bool = False  # true after /clear: the page empties its chat log


@app.get("/api/chat/history")
def chat_history_endpoint(session_id: SessionId, session: SessionDep) -> list[ChatMessageRead]:
    return [
        ChatMessageRead.model_validate(message)
        for message in chat_history.list_history(session, session_id)
    ]


@app.get("/api/tasks")
def list_tasks(session: SessionDep) -> list[TaskRead]:
    return [TaskRead.model_validate(task) for task in services.list_tasks(session)]


# A plain `def`: run_agent blocks on the LLM, so FastAPI runs this in its threadpool.
@app.post(
    "/api/chat",
    response_model=ChatResponse,
    responses={
        500: {"description": "/clear failed"},
        502: {"description": "The agent failed"},
    },
)
def chat(body: ChatRequest, session: SessionDep) -> ChatResponse | JSONResponse:
    if body.message.lower() == chat_history.CLEAR_COMMAND:
        # Handled here, before the model: it is deterministic and costs no tokens.
        try:
            chat_history.clear_session(session, body.session_id)
        except Exception:
            logger.exception("Could not clear the chat")
            return JSONResponse(
                status_code=500, content={"error": "Could not clear the chat. Please try again."}
            )
        tasks = [TaskRead.model_validate(task) for task in services.list_tasks(session)]
        return ChatResponse(
            reply="Chat cleared. Your tasks were not touched.", tasks=tasks, cleared=True
        )
    try:
        reply = run_agent(body.message, body.session_id)
    except LLMConfigError as exc:
        # A setup problem (no usable key). The message names variables, never secrets.
        logger.error("LLM is not configured: %s", exc)
        return JSONResponse(status_code=502, content={"error": str(exc)})
    except Exception:
        logger.exception("Agent failed")
        return JSONResponse(
            status_code=502,
            content={"error": "The assistant is unavailable right now. Please try again shortly."},
        )
    chat_history.save_exchange(session, body.session_id, body.message, reply)
    tasks = [TaskRead.model_validate(task) for task in services.list_tasks(session)]
    return ChatResponse(reply=reply, tasks=tasks)


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")
