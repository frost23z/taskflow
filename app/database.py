import os
from collections.abc import Iterator
from pathlib import Path

from dotenv import load_dotenv
from sqlmodel import Session, SQLModel, create_engine

import app.models  # noqa: F401  (registers the tables on SQLModel.metadata)

load_dotenv()

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DATA_DIR / 'tasks.db'}")
APP_ENV = os.getenv("APP_ENV", "dev")

# SQL logging only in dev mode.
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
    echo=APP_ENV == "dev",
)


def get_session() -> Iterator[Session]:
    with Session(engine) as session:
        yield session


def init_db() -> None:
    # The SQLite file may live anywhere now that the URL comes from the environment.
    if engine.url.database and engine.url.database != ":memory:":
        Path(engine.url.database).parent.mkdir(parents=True, exist_ok=True)
    SQLModel.metadata.create_all(engine)
