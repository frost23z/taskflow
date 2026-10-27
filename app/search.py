"""Semantic search over tasks. ChromaDB is only an index; SQLite stays the source of truth."""

import os
from pathlib import Path

import chromadb
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection
from dotenv import load_dotenv
from sqlmodel import Session, select

from app.models import Task

load_dotenv()

CHROMA_PATH = os.getenv("CHROMA_PATH", str(Path(__file__).resolve().parent.parent / "data/chroma"))
COLLECTION_NAME = "tasks"

_collection: Collection | None = None


def make_collection(client: ClientAPI | None = None, name: str = COLLECTION_NAME) -> Collection:
    """Open (or create) the cosine-distance task collection. Pass an ephemeral client in tests."""
    client = client or chromadb.PersistentClient(path=CHROMA_PATH)
    return client.get_or_create_collection(name=name, configuration={"hnsw": {"space": "cosine"}})


def get_collection() -> Collection:
    global _collection
    if _collection is None:
        _collection = make_collection()
    return _collection


def set_collection(collection: Collection | None) -> None:
    """Swap the collection used by every function below (None resets to the default one)."""
    global _collection
    _collection = collection


def _document(task: Task) -> str:
    return f"{task.title}\n{task.description}".strip()


def index_task(task: Task) -> None:
    if task.id is None:
        raise ValueError("task must be saved before it can be indexed")
    get_collection().upsert(ids=[str(task.id)], documents=[_document(task)])


def remove_task(task_id: int) -> None:
    get_collection().delete(ids=[str(task_id)])


def search(query: str, limit: int = 5) -> list[tuple[int, float]]:
    """Return (task_id, similarity) pairs, best first. similarity = 1 - cosine distance."""
    collection = get_collection()
    count = collection.count()
    if count == 0:
        return []
    result = collection.query(query_texts=[query], n_results=min(limit, count))
    ids = result["ids"][0]
    distances = (result["distances"] or [[]])[0]
    return [(int(id_), 1 - distance) for id_, distance in zip(ids, distances, strict=True)]


def reindex_all(session: Session) -> None:
    """Make Chroma match SQLite: add missing tasks, remove stale ones."""
    collection = get_collection()
    tasks = {str(task.id): task for task in session.exec(select(Task)).all()}
    indexed = set(collection.get(include=[])["ids"])

    stale = sorted(indexed - tasks.keys())
    if stale:
        collection.delete(ids=stale)

    missing = sorted(tasks.keys() - indexed)
    if missing:
        collection.add(ids=missing, documents=[_document(tasks[id_]) for id_ in missing])
