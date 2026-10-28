import hashlib
import math
import os
import re
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import chromadb
import numpy as np
import pytest
from chromadb import Documents, EmbeddingFunction, Embeddings
from chromadb.api.models.Collection import Collection
from dotenv import load_dotenv
from langgraph.checkpoint.memory import InMemorySaver
from sqlmodel import Session, SQLModel, create_engine

from app import agent, search
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


class WordHashEmbedding(EmbeddingFunction[Documents]):
    """Offline stand-in for the real embedding model: texts sharing words are similar.

    Identical texts score 1.0, unrelated texts 0.0, so tests can create clear, ambiguous and
    missing matches on purpose without downloading a model.
    """

    DIM = 256

    def __init__(self) -> None:
        pass

    def __call__(self, input: Documents) -> Embeddings:
        vectors: Embeddings = []
        for text in input:
            vector = [0.0] * self.DIM
            for word in re.findall(r"[a-z0-9]+", text.lower()):
                vector[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.DIM] += 1.0
            norm = math.sqrt(sum(x * x for x in vector)) or 1.0
            vectors.append(np.array([x / norm for x in vector], dtype=np.float32))
        return vectors

    @staticmethod
    def name() -> str:
        return "word-hash-test"

    def get_config(self) -> dict:
        return {}

    @staticmethod
    def build_from_config(config: dict) -> WordHashEmbedding:
        return WordHashEmbedding()


@pytest.fixture
def fake_chroma() -> Iterator[Collection]:
    """Like `chroma`, but with WordHashEmbedding so it works offline and similarities are known."""
    client = chromadb.EphemeralClient()
    collection = client.get_or_create_collection(
        name=f"tasks-{uuid.uuid4().hex}",
        configuration={"hnsw": {"space": "cosine"}},
        # Chroma types embedding functions as EmbeddingFunction[Embeddable]; ours takes text only.
        embedding_function=cast(Any, WordHashEmbedding()),
    )
    search.set_collection(collection)
    yield collection
    search.set_collection(None)
    client.delete_collection(collection.name)


@pytest.fixture(autouse=True)
def memory_checkpointer() -> Iterator[None]:
    """Tests never write the agent's memory to data/checkpoints.db."""
    agent.set_checkpointer(InMemorySaver())
    yield
    agent.set_checkpointer(None)
