"""Measure how well semantic search finds tasks. Local only: no LLM calls, no database writes.

    uv run python -m scripts.eval_retrieval            # summary + every failing case
    uv run python -m scripts.eval_retrieval --verbose  # also print passing cases

It indexes the demo tasks from app.seed in a throwaway in-memory ChromaDB, runs the queries below
through app.search.search, and applies the same decision rule as app.tools._resolve (same
MIN_SIMILARITY and AMBIGUITY_MARGIN), so the numbers describe what the agent would actually do.

Each case has an expected outcome:
- act:  one task is the clear match, so the agent may update/delete it without asking.
- ask:  several tasks are plausible, so the agent must ask which one is meant. Passes when at
        least two tasks are plausible and all of them are expected ones (so the agent never acts
        alone, and never asks about unrelated tasks).
- list: a search/list query; every expected task must appear in the top results.
- none: nothing in the data matches, so the agent must answer "not found".

The expected outcomes are judgement calls about the seed data. If a case fails, look at the scores
before deciding whether the search or the expectation is wrong.
"""

import argparse
from dataclasses import dataclass

import chromadb
from chromadb.api.models.Collection import Collection

from app import search
from app.models import Task
from app.seed import SEED_TASKS
from app.tools import AMBIGUITY_MARGIN, MAX_CANDIDATES, MIN_SIMILARITY

LIST_LIMIT = 5  # how many results a list query is judged on


@dataclass(frozen=True)
class Case:
    query: str
    kind: str  # "act" | "ask" | "list" | "none"
    expected: tuple[str, ...] = ()  # task titles


CASES: list[Case] = [
    # Short targets the agent would pass for update/delete: one clear task.
    Case("login bug", "act", ("Fix login redirect bug",)),
    Case("pull request from Sam", "act", ("Review Sam's pull request",)),
    Case("Sentry", "act", ("Set up error monitoring",)),
    Case("payment tests", "act", ("Write unit tests for the payment module",)),
    Case("slow query", "act", ("Optimize slow dashboard query",)),
    Case("API keys", "act", ("Rotate production API keys",)),
    Case("release notes", "act", ("Write release notes for v2.3",)),
    Case("pagination", "act", ("Add pagination to the orders API",)),
    Case("python upgrade", "act", ("Upgrade Python to 3.14",)),
    Case("sprint agenda", "act", ("Prepare sprint planning agenda",)),
    Case("one on one with Priya", "act", ("Schedule 1:1 with Priya",)),
    Case("conference room", "act", ("Book conference room for retrospective",)),
    Case("passport", "act", ("Renew passport",)),
    Case("groceries", "act", ("Buy groceries for the week",)),
    Case("dentist", "act", ("Book dentist appointment",)),
    Case("hiking", "act", ("Plan weekend hiking trip",)),
    Case("electricity bill", "act", ("Pay electricity bill",)),
    Case("Mom's birthday", "act", ("Call Mom for her birthday",)),
    # Paraphrases with no shared keywords: the point of semantic search.
    Case("get my teeth cleaned", "act", ("Book dentist appointment",)),
    Case("pay the power bill", "act", ("Pay electricity bill",)),
    Case("buy food", "act", ("Buy groceries for the week",)),
    Case("monitor production errors", "act", ("Set up error monitoring",)),
    Case("slides for the customer presentation", "act", ("Finish demo slides",)),
    Case("screen recording walkthrough", "act", ("Record demo video",)),
    Case("fix the sign-in problem", "act", ("Fix login redirect bug",)),
    # Genuinely ambiguous: the agent must ask.
    Case("demo", "ask", ("Finish demo slides", "Record demo video")),
    Case("the demo task", "ask", ("Finish demo slides", "Record demo video")),
    # The assignment's own example ("change the priority of the API task"): three tasks mention
    # the API, so acting on one of them would be a wrong guess. Not checked against real
    # embeddings yet: run the eval and look at the scores before trusting this expectation.
    Case(
        "the API task",
        "ask",
        (
            "Add pagination to the orders API",
            "Rotate production API keys",
            "Update API documentation",
        ),
    ),
    # Search and list requests: every expected task must be in the top results.
    Case(
        "documentation",
        "list",
        ("Update API documentation", "Write README quickstart"),
    ),
    Case(
        "unfinished tasks related to documentation",
        "list",
        ("Update API documentation", "Write README quickstart"),
    ),
    Case(
        "meetings",
        "list",
        ("Schedule 1:1 with Priya", "Send meeting notes from Monday's design review"),
    ),
    Case("things to book", "list", ("Book dentist appointment", "Renew passport")),
    # Nothing relevant: must come back as "not found".
    Case("quantum physics homework", "none"),
    Case("buy a new car", "none"),
    Case("learn to play guitar", "none"),
]


def build_collection() -> Collection:
    """In-memory index of the seed tasks, using Chroma's default embedding model."""
    collection = search.make_collection(chromadb.EphemeralClient(), name="eval_tasks")
    tasks = [
        Task(id=i, title=title, description=description)
        for i, (title, description, *_rest) in enumerate(SEED_TASKS, start=1)
    ]
    collection.upsert(
        ids=[str(task.id) for task in tasks],
        documents=[search._document(task) for task in tasks],
    )
    return collection


@dataclass
class Result:
    case: Case
    ok: bool
    outcome: str  # what the agent would do, in words
    ranked: list[tuple[str, float]]  # (title, similarity), best first
    rank: int | None  # 1-based rank of the first expected task, if any


def decide(ranked: list[tuple[str, float]]) -> list[str]:
    """Titles the agent would treat as plausible. Mirrors app.tools._resolve."""
    matches = [
        (title, score) for title, score in ranked[:MAX_CANDIDATES] if score >= MIN_SIMILARITY
    ]
    if not matches:
        return []
    best = matches[0][1]
    return [title for title, score in matches if best - score <= AMBIGUITY_MARGIN]


def run_case(case: Case, titles_by_id: dict[int, str]) -> Result:
    hits = search.search(case.query, limit=LIST_LIMIT)
    ranked = [(titles_by_id[id_], similarity) for id_, similarity in hits]
    ranked_titles = [title for title, _ in ranked]
    rank = next((i for i, t in enumerate(ranked_titles, 1) if t in case.expected), None)
    plausible = decide(ranked)

    if case.kind == "act":
        ok = plausible == list(case.expected)
        outcome = f"acts on {plausible[0]!r}" if len(plausible) == 1 else _describe(plausible)
    elif case.kind == "ask":
        ok = len(plausible) >= 2 and set(plausible) <= set(case.expected)
        outcome = _describe(plausible)
    elif case.kind == "list":
        ok = all(title in ranked_titles for title in case.expected)
        missing = [t for t in case.expected if t not in ranked_titles]
        outcome = "all expected found" if ok else f"missing {missing}"
    else:  # none
        ok = not plausible
        outcome = "not found" if ok else f"false match {plausible}"
    return Result(case, ok, outcome, ranked, rank)


def _describe(plausible: list[str]) -> str:
    if not plausible:
        return "not found"
    if len(plausible) == 1:
        return f"acts on {plausible[0]!r}"
    return f"asks which of {plausible}"


def evaluate(collection: Collection) -> list[Result]:
    search.set_collection(collection)
    try:
        titles_by_id = {i: title for i, (title, *_rest) in enumerate(SEED_TASKS, start=1)}
        return [run_case(case, titles_by_id) for case in CASES]
    finally:
        search.set_collection(None)


def print_report(results: list[Result], verbose: bool) -> None:
    for result in results:
        if result.ok and not verbose:
            continue
        mark = "ok  " if result.ok else "FAIL"
        top = ", ".join(f"{title[:28]} {score:.2f}" for title, score in result.ranked[:3])
        print(f"{mark} [{result.case.kind:4}] {result.case.query!r}: {result.outcome}")
        print(f"       top 3: {top}")

    print("\nSummary")
    for kind, label in (
        ("act", "clear match, agent acts"),
        ("ask", "ambiguous, agent asks"),
        ("list", "list/search, all found"),
        ("none", "no match, not found"),
    ):
        group = [r for r in results if r.case.kind == kind]
        passed = sum(r.ok for r in group)
        print(f"  {label:26} {passed}/{len(group)}")
    total = sum(r.ok for r in results)
    print(f"  {'overall':26} {total}/{len(results)} ({100 * total / len(results):.0f}%)")

    ranked = [r for r in results if r.case.kind in ("act", "ask", "list")]
    hit1 = sum(r.rank == 1 for r in ranked)
    hit3 = sum(r.rank is not None and r.rank <= 3 for r in ranked)
    mrr = sum(1 / r.rank for r in ranked if r.rank) / len(ranked)
    print(f"\nRanking over {len(ranked)} queries with an expected task:")
    print(f"  top-1 hit {hit1}/{len(ranked)}   top-3 hit {hit3}/{len(ranked)}   MRR {mrr:.2f}")

    print_threshold_sweep(results)


def print_threshold_sweep(results: list[Result]) -> None:
    """Show why MIN_SIMILARITY is where it is: best scores of matching vs. non-matching queries."""
    found = [r.ranked[0][1] for r in results if r.case.kind != "none" and r.ranked]
    absent = [r.ranked[0][1] for r in results if r.case.kind == "none" and r.ranked]
    if not found or not absent:
        return
    print("\nMIN_SIMILARITY sweep (best score per query):")
    print(f"  queries with a real match: lowest best-score {min(found):.2f}")
    print(f"  queries with no match:     highest best-score {max(absent):.2f}")
    print("  cutoff   real matches kept   no-match queries rejected")
    for cutoff in (0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50):
        kept = sum(score >= cutoff for score in found)
        rejected = sum(score < cutoff for score in absent)
        current = "  <- current" if abs(cutoff - MIN_SIMILARITY) < 1e-9 else ""
        kept_text = f"{kept}/{len(found)}"
        print(f"  {cutoff:.2f}     {kept_text:<18} {rejected}/{len(absent)}{current}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--verbose", action="store_true", help="also print passing cases")
    args = parser.parse_args()
    print(
        f"Indexing {len(SEED_TASKS)} seed tasks. The first run downloads the embedding model.\n"
        f"Rule: similarity >= {MIN_SIMILARITY}, ambiguous within {AMBIGUITY_MARGIN} of the best.\n"
    )
    print_report(evaluate(build_collection()), args.verbose)


if __name__ == "__main__":
    main()
