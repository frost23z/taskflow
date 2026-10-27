import asyncio
import logging
from typing import Any

import httpx
import pytest
from langchain.agents import create_agent
from langchain_core.exceptions import ModelRateLimitError
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from pydantic import ConfigDict

from app import llm
from app.llm import (
    LLMConfigError,
    ProviderFallbackMiddleware,
    build_model,
    is_provider_failure,
    parse_chain,
)

KEYS = {"GOOGLE_API_KEY": "g", "GROQ_API_KEY": "q", "OPENROUTER_API_KEY": "o"}


class FakeChat(BaseChatModel):
    """Chat model that answers with its own name, or raises `error` when set. No network."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    label: str
    error: Exception | None = None
    calls: int = 0
    tools_bound: bool = False

    @property
    def _llm_type(self) -> str:
        return "fake"

    def _generate(
        self, messages: list[BaseMessage], stop: Any = None, run_manager: Any = None, **kwargs: Any
    ) -> ChatResult:
        self.calls += 1
        if self.error:
            raise self.error
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=f"from {self.label}"))]
        )

    def bind_tools(self, tools: Any, **kwargs: Any) -> FakeChat:
        self.tools_bound = True
        return self


class StatusError(Exception):
    """Stand-in for an SDK error carrying an HTTP status code."""

    def __init__(self, status_code: int, message: str = "boom") -> None:
        super().__init__(message)
        self.status_code = status_code


@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


def make_agent(*models: FakeChat):
    chain = [(f"fake:{m.label}", m) for m in models]
    return create_agent(
        model=models[0], tools=[add], middleware=[ProviderFallbackMiddleware(chain)]
    )


def ask(agent) -> str:
    return agent.invoke({"messages": [HumanMessage("hi")]})["messages"][-1].content


# --- chain parsing ------------------------------------------------------------------------------


def test_parse_splits_on_first_colon_only() -> None:
    [entry] = parse_chain("openrouter:google/gemma-4-31b-it:free")
    assert (entry.provider, entry.model) == ("openrouter", "google/gemma-4-31b-it:free")
    assert entry.label == "openrouter:google/gemma-4-31b-it:free"


def test_parse_allows_repeated_provider_and_keeps_order() -> None:
    entries = parse_chain(" google_genai:a , groq:b,google_genai:c ,, ")
    assert [e.label for e in entries] == ["google_genai:a", "groq:b", "google_genai:c"]


def test_parse_unknown_provider_is_a_clear_error() -> None:
    with pytest.raises(LLMConfigError, match="Unknown LLM provider 'nope'"):
        parse_chain("google_genai:a,nope:b")


@pytest.mark.parametrize("raw", ["", " , ", "groq", "groq:", ":model"])
def test_parse_rejects_empty_or_malformed(raw: str) -> None:
    with pytest.raises(LLMConfigError):
        parse_chain(raw)


@pytest.fixture
def created(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Replace init_chat_model with a fake factory and record how it was called."""
    calls: list[dict[str, Any]] = []

    def fake_init(model: str, **kwargs: Any) -> FakeChat:
        calls.append({"model": model, **kwargs})
        return FakeChat(label=model)

    monkeypatch.setattr(llm, "init_chat_model", fake_init)
    return calls


def test_build_model_skips_entries_without_a_key(created, caplog: pytest.LogCaptureFixture) -> None:
    env = {"GROQ_API_KEY": "q"}
    with caplog.at_level(logging.WARNING, logger="app.llm"):
        primary, middleware = build_model("google_genai:a,groq:b", env)
    assert [c["model"] for c in created] == ["b"]
    assert isinstance(primary, FakeChat) and primary.label == "b"
    assert "Skipping google_genai:a: GOOGLE_API_KEY is not set" in caplog.text
    assert len(middleware) == 1


def test_build_model_nothing_usable_is_a_clear_error(created) -> None:
    with pytest.raises(LLMConfigError, match="No usable model"):
        build_model("google_genai:a,groq:b", {})
    assert created == []


def test_build_model_passes_timeout_and_no_sdk_retries(created) -> None:
    build_model("groq:x,google_genai:y,openrouter:z:free", KEYS)
    by_model = {c["model"]: c for c in created}
    assert all(c["max_retries"] == 0 for c in created)
    assert by_model["x"]["timeout"] == llm.TIMEOUT_SECONDS
    assert by_model["z:free"]["timeout"] == llm.TIMEOUT_SECONDS * 1000  # OpenRouter uses ms
    assert by_model["z:free"]["model_provider"] == "openrouter"
    assert by_model["x"]["api_key"] == "q"


def test_build_model_repeated_provider_makes_separate_models(created) -> None:
    build_model("google_genai:a,google_genai:b", KEYS)
    assert [c["model"] for c in created] == ["a", "b"]


def chain_labels(middleware: list[Any]) -> list[str]:
    [fallback] = middleware
    assert isinstance(fallback, ProviderFallbackMiddleware)
    return [label for label, _ in fallback.chain]


def test_comma_separated_keys_expand_each_entry_in_order(created) -> None:
    env = {"GOOGLE_API_KEY": "k1, k2 ,", "GROQ_API_KEY": "q"}
    _, middleware = build_model("google_genai:a,groq:b,google_genai:c", env)
    assert chain_labels(middleware) == [
        "google_genai:a#1",
        "google_genai:a#2",
        "groq:b",
        "google_genai:c#1",
        "google_genai:c#2",
    ]
    assert [c["api_key"] for c in created] == ["k1", "k2", "q", "k1", "k2"]


def test_single_key_has_no_slot_suffix(created) -> None:
    _, middleware = build_model("google_genai:a", {"GOOGLE_API_KEY": "k1"})
    assert chain_labels(middleware) == ["google_genai:a"]


def test_blank_key_list_counts_as_missing(created) -> None:
    with pytest.raises(LLMConfigError, match="No usable model"):
        build_model("google_genai:a", {"GOOGLE_API_KEY": " , "})


# --- which errors count as provider failures ----------------------------------------------------


@pytest.mark.parametrize(
    "exc",
    [
        ModelRateLimitError("quota"),
        StatusError(429),
        StatusError(402),
        StatusError(503),
        StatusError(500),
        StatusError(413, "Request too large on tokens per minute (TPM)"),
        httpx.ConnectTimeout("slow"),
        httpx.ConnectError("down"),
        TimeoutError(),
        ConnectionError(),
    ],
)
def test_provider_failures_are_detected(exc: Exception) -> None:
    assert is_provider_failure(exc)


@pytest.mark.parametrize(
    "exc",
    [
        ValueError("bug"),
        KeyError("x"),
        StatusError(400),
        StatusError(401),
        StatusError(404),
        StatusError(413, "payload too large"),
    ],
)
def test_other_errors_are_not_provider_failures(exc: Exception) -> None:
    assert not is_provider_failure(exc)


def test_provider_error_in_cause_chain_counts() -> None:
    try:
        try:
            raise StatusError(429)
        except StatusError as inner:
            raise RuntimeError("wrapped") from inner
    except RuntimeError as wrapped:
        assert is_provider_failure(wrapped)


# --- fallback behaviour through create_agent ----------------------------------------------------


@pytest.mark.parametrize(
    "error", [StatusError(429), ModelRateLimitError("quota"), httpx.ReadTimeout("t")]
)
def test_rate_limited_primary_falls_through_to_next_model(error: Exception) -> None:
    primary, backup = FakeChat(label="a", error=error), FakeChat(label="b")
    assert ask(make_agent(primary, backup)) == "from b"
    assert (primary.calls, backup.calls) == (1, 1)
    assert backup.tools_bound  # tools are bound again on the fallback model


def test_fallback_walks_the_whole_chain_in_order() -> None:
    a, b = (FakeChat(label=n, error=StatusError(503)) for n in "ab")
    c = FakeChat(label="c")
    assert ask(make_agent(a, b, c)) == "from c"
    assert (a.calls, b.calls, c.calls) == (1, 1, 1)


def test_all_models_failing_raises_the_last_provider_error() -> None:
    a = FakeChat(label="a", error=StatusError(429, "first"))
    b = FakeChat(label="b", error=StatusError(503, "last"))
    with pytest.raises(StatusError, match="last"):
        ask(make_agent(a, b))


@pytest.mark.parametrize("error", [ValueError("bug"), StatusError(400)])
def test_non_provider_error_does_not_fall_through(error: Exception) -> None:
    primary, backup = FakeChat(label="a", error=error), FakeChat(label="b")
    with pytest.raises(type(error)):
        ask(make_agent(primary, backup))
    assert backup.calls == 0


def test_async_path_falls_back_and_respects_the_filter() -> None:
    async def run(agent) -> str:
        result = await agent.ainvoke({"messages": [HumanMessage("hi")]})
        return result["messages"][-1].content

    primary, backup = FakeChat(label="a", error=StatusError(429)), FakeChat(label="b")
    assert asyncio.run(run(make_agent(primary, backup))) == "from b"

    bad, other = FakeChat(label="a", error=ValueError("bug")), FakeChat(label="b")
    with pytest.raises(ValueError):
        asyncio.run(run(make_agent(bad, other)))
    assert other.calls == 0


# --- logging ------------------------------------------------------------------------------------


def test_logs_the_model_that_answered(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="app.llm"):
        ask(make_agent(FakeChat(label="a"), FakeChat(label="b")))
    assert "Model call handled by fake:a" in caplog.text


def test_logs_each_fallback_with_the_error_class(caplog: pytest.LogCaptureFixture) -> None:
    primary, backup = FakeChat(label="a", error=StatusError(429)), FakeChat(label="b")
    with caplog.at_level(logging.INFO, logger="app.llm"):
        ask(make_agent(primary, backup))
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "fake:a failed with StatusError; falling back to fake:b" in warnings[0].getMessage()
    assert "Model call handled by fake:b" in caplog.text
