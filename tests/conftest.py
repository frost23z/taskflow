import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from dotenv import load_dotenv
from sqlmodel import Session, SQLModel, create_engine

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
