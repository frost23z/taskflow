"""Chat with the agent in the terminal. Makes real LLM calls (uses your free-tier quota).

    uv run python -m app.seed          # optional: demo tasks
    uv run python -m scripts.chat_cli

Type "exit" (or press Ctrl-D) to quit. Each run is a new session with its own history.
"""

import logging
import uuid

from app import database
from app.agent import run_agent


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("app.llm").setLevel(logging.INFO)  # shows which model answered
    database.engine.echo = False  # no SQL noise in the chat
    database.init_db()

    session_id = uuid.uuid4().hex
    print(f"TaskFlow chat (session {session_id[:8]}). Type 'exit' to quit.")
    while True:
        try:
            message = input("you> ").strip()
        except EOFError, KeyboardInterrupt:
            print()
            break
        if message.lower() in {"exit", "quit"}:
            break
        if not message:
            continue
        try:
            print(f"bot> {run_agent(message, session_id)}")
        except Exception as exc:
            print(f"bot> [error] {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
