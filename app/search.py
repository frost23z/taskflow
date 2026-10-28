"""Semantic search over tasks. ChromaDB is only an index; SQLite stays the source of truth."""

import os
from pathlib import Path

import chromadb
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection
from chromadb.api.types import Where
from dotenv import load_dotenv
from sqlmodel import Session, select

from app.models import Status, Task

load_dotenv()

CHROMA_PATH = os.getenv("CHROMA_PATH", str(Path(__file__).resolve().parent.parent / "data/chroma"))
COLLECTION_NAME = "tasks"

_collection: Collection | None = None

# True when a write to SQLite could not be mirrored into Chroma. The next search rebuilds the
# index from SQLite first (see services.find_tasks), so a failed sync never loses a task.
_stale = False


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
    global _collection, _stale
    _collection = collection
    _stale = False


def mark_stale() -> None:
    global _stale
    _stale = True


def is_stale() -> bool:
    return _stale


def _document(task: Task) -> str:
    return f"{task.title}\n{task.description}".strip()


def _metadata(task: Task) -> dict[str, str]:
    # Stored so "unfinished tasks" can be filtered inside the vector query, before the limit.
    return {"status": Status(task.status).value}


def index_task(task: Task) -> None:
    if task.id is None:
        raise ValueError("task must be saved before it can be indexed")
    get_collection().upsert(
        ids=[str(task.id)], documents=[_document(task)], metadatas=[_metadata(task)]
    )


def remove_task(task_id: int) -> None:
    get_collection().delete(ids=[str(task_id)])


def search(query: str, limit: int = 5, include_done: bool = True) -> list[tuple[int, float]]:
    """Return (task_id, similarity) pairs, best first. similarity = 1 - cosine distance.

    include_done=False leaves done tasks out inside the query, so the limit applies to open
    tasks only (filtering afterwards would let done tasks use up the limit).
    """
    collection = get_collection()
    count = collection.count()
    if count == 0:
        return []
    where: Where | None = None if include_done else {"status": {"$ne": Status.DONE.value}}
    result = collection.query(query_texts=[query], n_results=min(limit, count), where=where)
    ids = result["ids"][0]
    distances = (result["distances"] or [[]])[0]
    return [(int(id_), 1 - distance) for id_, distance in zip(ids, distances, strict=True)]


REINDEX_BATCH = 500


def reindex_all(session: Session) -> None:
    """Make Chroma match SQLite: add missing tasks, fix changed ones, remove stale ones."""
    global _stale
    collection = get_collection()
    tasks = {str(task.id): task for task in session.exec(select(Task)).all()}
    existing = collection.get(include=["documents", "metadatas"])
    documents = existing["documents"] or []
    metadatas = existing["metadatas"] or []
    indexed = {
        id_: (
            documents[i] if i < len(documents) else None,
            metadatas[i] if i < len(metadatas) else None,
        )
        for i, id_ in enumerate(existing["ids"])
    }

    stale = sorted(indexed.keys() - tasks.keys())
    if stale:
        collection.delete(ids=stale)

    # Missing from the index, edited behind its back, or indexed without current metadata.
    outdated = sorted(
        id_
        for id_, task in tasks.items()
        if id_ not in indexed
        or indexed[id_][0] != _document(task)
        or dict(indexed[id_][1] or {}) != _metadata(task)
    )
    for start in range(0, len(outdated), REINDEX_BATCH):
        batch = outdated[start : start + REINDEX_BATCH]
        collection.upsert(
            ids=batch,
            documents=[_document(tasks[id_]) for id_ in batch],
            metadatas=[_metadata(tasks[id_]) for id_ in batch],
        )
    _stale = False
