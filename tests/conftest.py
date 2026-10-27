import os
import uuid
from collections.abc import Iterator
from pathlib import Path

import chromadb
import pytest
from chromadb.api.models.Collection import Collection
from dotenv import load_dotenv
from sqlmodel import Session, SQLModel, create_engine

from app import search
from app.database import DATABASE_URL

load_dotenv()

DEFAULT_TEST_DB = Path(__file__).resolve().parent.parent / "data" / "test.db"
TEST_DATABASE_URL = os.getenv("TEST_DATABASE_URL", f"sqlite:///{DEFAULT_TEST_DB}")


@pytest.fixture
def session() -> Iterator[Session]:
    # The tables are dropped below, so never let this point at the real database.
    if TEST_DATABASE_URL == DATABASE_URL:
        pytest.fail("TEST_DATABASE_URL must differ from DATABASE_URL", pytrace=False)

    engine = create_engine(TEST_DATABASE_URL)
    if engine.url.database:
        Path(engine.url.database).parent.mkdir(parents=True, exist_ok=True)
    SQLModel.metadata.drop_all(engine)
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        yield session
    engine.dispose()


@pytest.fixture
def chroma() -> Iterator[Collection]:
    # Ephemeral clients share state inside one process, so every test gets its own collection.
    client = chromadb.EphemeralClient()
    collection = search.make_collection(client, name=f"tasks-{uuid.uuid4().hex}")
    search.set_collection(collection)
    yield collection
    search.set_collection(None)
    client.delete_collection(collection.name)
