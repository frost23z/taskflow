"""LLM configuration: an ordered chain of free-tier models with provider-failure fallback.

LLM_CHAIN lists "provider:model" entries, best first. The first usable entry is the primary model
and the rest are fallbacks. Use it like this:

    primary, middleware = build_model()
    agent = create_agent(model=primary, middleware=middleware, tools=[...])
"""

import logging
import os
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import groq
import httpx
from dotenv import load_dotenv
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain.chat_models import init_chat_model
from langchain_core.exceptions import (
    ModelAPIError,
    ModelConnectionError,
    ModelError,
    ModelRateLimitError,
    ModelTimeoutError,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage

load_dotenv()

logger = logging.getLogger(__name__)

# provider (as written in LLM_CHAIN, also the init_chat_model provider name) -> key variable.
PROVIDER_KEY_ENV = {
    "google_genai": "GOOGLE_API_KEY",
    "groq": "GROQ_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}

# Short on purpose: a rate-limited or stalled model should fail over quickly. max_retries=0 stops
# the SDK from sleeping and retrying on its own before we get to try the next model.
TIMEOUT_SECONDS = 20
MAX_RETRIES = 0


class LLMConfigError(ValueError):
    """LLM_CHAIN is malformed, names an unknown provider, or has no usable entry."""


@dataclass(frozen=True)
class ChainEntry:
    provider: str
    model: str
    # 1-based position among several comma-separated keys; None when the provider has one key.
    key_slot: int | None = None

    @property
    def label(self) -> str:
        base = f"{self.provider}:{self.model}"
        return base if self.key_slot is None else f"{base}#{self.key_slot}"


def parse_chain(raw: str) -> list[ChainEntry]:
    """Parse "provider:model,provider:model". Each entry splits on the first colon only."""
    entries: list[ChainEntry] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        provider, sep, model = item.partition(":")
        provider, model = provider.strip(), model.strip()
        if not sep or not provider or not model:
            raise LLMConfigError(f"Invalid LLM_CHAIN entry {item!r}: expected 'provider:model'")
        if provider not in PROVIDER_KEY_ENV:
            known = ", ".join(sorted(PROVIDER_KEY_ENV))
            raise LLMConfigError(f"Unknown LLM provider {provider!r} in LLM_CHAIN (known: {known})")
        entries.append(ChainEntry(provider, model))
    if not entries:
        raise LLMConfigError("LLM_CHAIN is empty; set it to e.g. 'google_genai:gemini-3.8-flash'")
    return entries


def entry_keys(entry: ChainEntry, env: Mapping[str, str]) -> list[tuple[ChainEntry, str]]:
    """Expand an entry into one (entry, key) pair per API key set for its provider.

    The key variable may hold several comma-separated keys (for example Gemini keys from separate
    Google Cloud projects, whose quotas are independent). Each key becomes its own model, tried in
    the order written, before the chain moves on to the next entry. Returns [] if no key is set.
    """
    keys = [k.strip() for k in env.get(PROVIDER_KEY_ENV[entry.provider], "").split(",")]
    keys = [k for k in keys if k]
    if len(keys) <= 1:
        return [(entry, key) for key in keys]
    return [(replace(entry, key_slot=slot), key) for slot, key in enumerate(keys, start=1)]


def usable_entries(
    entries: Sequence[ChainEntry], env: Mapping[str, str]
) -> list[tuple[ChainEntry, str]]:
    """Pair each entry with its API key(s). Entries without a key are skipped with a warning."""
    usable: list[tuple[ChainEntry, str]] = []
    for entry in entries:
        pairs = entry_keys(entry, env)
        if pairs:
            usable.extend(pairs)
        else:
            logger.warning(
                "Skipping %s: %s is not set", entry.label, PROVIDER_KEY_ENV[entry.provider]
            )
    if not usable:
        needed = ", ".join(sorted({PROVIDER_KEY_ENV[e.provider] for e in entries}))
        raise LLMConfigError(f"No usable model in LLM_CHAIN: set at least one of {needed}")
    return usable


def create_model(entry: ChainEntry, api_key: str) -> BaseChatModel:
    """Create one chat model with a short timeout and no SDK-level retries."""
    # Passing model_provider separately keeps colons in the model name (":free") intact.
    # Timeout units differ per integration: OpenRouter takes milliseconds, the others seconds.
    timeout = TIMEOUT_SECONDS * 1000 if entry.provider == "openrouter" else TIMEOUT_SECONDS
    return init_chat_model(
        entry.model,
        model_provider=entry.provider,
        api_key=api_key,
        timeout=timeout,
        max_retries=MAX_RETRIES,
    )


# --- Which errors are provider failures -------------------------------------------------------

# 402: credits or quota exhausted, 408: request timeout, 429: rate limited.
_FALLBACK_STATUS = {402, 408, 429}


def _is_provider_failure_one(exc: BaseException) -> bool:
    if isinstance(exc, ModelError):
        # Integrations that classify their errors (Gemini does) tell us directly.
        return isinstance(
            exc, ModelRateLimitError | ModelAPIError | ModelConnectionError | ModelTimeoutError
        )
    for attr in ("status_code", "code"):
        status = getattr(exc, attr, None)
        if isinstance(status, int):
            if status in _FALLBACK_STATUS or status >= 500:
                return True
            # Groq answers 413 when a request exceeds the free tokens-per-minute budget.
            return status == 413 and "tokens per minute" in str(exc).lower()
    return isinstance(
        exc, TimeoutError | ConnectionError | httpx.TransportError | groq.APIConnectionError
    )


def is_provider_failure(exc: BaseException) -> bool:
    """True for 429/quota, 402, timeouts, connection errors and 5xx; False for everything else.

    Anything else (validation errors, tool errors, bugs, 400/401/404) must surface instead of
    silently burning quota on the next model. The cause chain is checked because wrappers
    re-raise provider errors with `from`.
    """
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        if _is_provider_failure_one(current):
            return True
        seen.add(id(current))
        current = current.__cause__
    return False


# --- Fallback middleware ------------------------------------------------------------------------


class ProviderFallbackMiddleware(AgentMiddleware):
    """Try each model in order; move on only when the failure is a provider failure.

    The built-in ModelFallbackMiddleware catches every Exception, so it would hide real bugs.
    Overriding request.model makes create_agent bind the tools to that model again, so tool
    calling works on every model in the chain.
    """

    def __init__(self, chain: Sequence[tuple[str, BaseChatModel]]) -> None:
        super().__init__()
        if not chain:
            raise ValueError("ProviderFallbackMiddleware needs at least one model")
        self.chain = list(chain)

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse | AIMessage:
        last_error: Exception | None = None
        for index, (label, model) in enumerate(self.chain):
            try:
                response = handler(request.override(model=model))
            except Exception as exc:
                if not is_provider_failure(exc):
                    raise
                last_error = exc
                self._log_failure(index, label, exc)
                continue
            logger.info("Model call handled by %s", label)
            return response
        assert last_error is not None
        raise last_error

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse | AIMessage:
        last_error: Exception | None = None
        for index, (label, model) in enumerate(self.chain):
            try:
                response = await handler(request.override(model=model))
            except Exception as exc:
                if not is_provider_failure(exc):
                    raise
                last_error = exc
                self._log_failure(index, label, exc)
                continue
            logger.info("Model call handled by %s", label)
            return response
        assert last_error is not None
        raise last_error

    def _log_failure(self, index: int, label: str, exc: Exception) -> None:
        if index + 1 < len(self.chain):
            logger.warning(
                "%s failed with %s; falling back to %s",
                label,
                type(exc).__name__,
                self.chain[index + 1][0],
            )
        else:
            logger.error("%s failed with %s; no models left", label, type(exc).__name__)


# --- Public entry point -------------------------------------------------------------------------


def build_model(
    chain: str | None = None, env: Mapping[str, str] | None = None
) -> tuple[BaseChatModel, list[AgentMiddleware[Any, Any, Any]]]:
    """Return (primary_model, middleware) for create_agent(model=primary, middleware=middleware).

    The first usable entry is the primary; the rest are fallbacks tried in order. Arguments
    default to LLM_CHAIN and os.environ; they exist so tests need no environment patching.
    """
    raw = chain if chain is not None else os.getenv("LLM_CHAIN", "")
    usable = usable_entries(parse_chain(raw), env if env is not None else os.environ)
    models = [(entry.label, create_model(entry, key)) for entry, key in usable]
    logger.info("LLM chain: %s", " -> ".join(label for label, _ in models))
    return models[0][1], [ProviderFallbackMiddleware(models)]
