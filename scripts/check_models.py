"""Check every model in LLM_CHAIN on its own (no fallback) with a real tool-calling request.

Makes real API calls and spends free-tier quota, so run it by hand, not from pytest:

    uv run python -m scripts.check_models
"""

import os
import sys
import time

from dotenv import load_dotenv
from langchain_core.tools import tool

from app.llm import PROVIDER_KEY_ENV, ChainEntry, create_model, parse_chain

PROMPT = "What is 17 + 25? Use the tool."


@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


def check(entry: ChainEntry, key: str) -> bool:
    start = time.perf_counter()
    try:
        response = create_model(entry, key).bind_tools([add]).invoke(PROMPT)
        calls = [call for call in response.tool_calls if call["name"] == "add"]
        if not calls:
            raise RuntimeError(f"no add tool call (got {str(response.content)[:80]!r})")
        detail = f"add({calls[0]['args']})"
        ok = True
    except Exception as exc:
        detail = f"{type(exc).__name__}: {str(exc)[:200]}"
        ok = False
    elapsed = time.perf_counter() - start
    print(f"{'OK  ' if ok else 'FAIL'} {entry.label:<50} {elapsed:6.2f}s  {detail}")
    return ok


def main() -> int:
    load_dotenv()
    failures = 0
    for entry in parse_chain(os.getenv("LLM_CHAIN", "")):
        key_var = PROVIDER_KEY_ENV[entry.provider]
        key = os.getenv(key_var, "").strip()
        if not key:
            print(f"SKIP {entry.label:<50} {key_var} is not set")
        elif not check(entry, key):
            failures += 1
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
