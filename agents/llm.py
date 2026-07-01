"""Shared OpenAI client with retries, token budget, and lazy initialization."""

from __future__ import annotations

import json
from typing import Optional

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from agents.config import get_settings
from agents.logging_config import log

_client: Optional[OpenAI] = None
_session_tokens = 0


def get_client() -> OpenAI:
    global _client
    if _client is None:
        settings = get_settings()
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY environment variable required")
        _client = OpenAI(api_key=settings.openai_api_key)
    return _client


def reset_session_tokens() -> None:
    global _session_tokens
    _session_tokens = 0


def get_session_tokens() -> int:
    return _session_tokens


def _track_usage(total_tokens: int) -> None:
    global _session_tokens
    _session_tokens += total_tokens
    settings = get_settings()
    if _session_tokens > settings.session_token_budget:
        raise RuntimeError(
            f"Session token budget exceeded ({_session_tokens} > {settings.session_token_budget})"
        )


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def chat_json(
    system: str,
    user: str,
    *,
    model: Optional[str] = None,
    max_tokens: int = 1500,
    temperature: Optional[float] = None,
) -> dict:
    settings = get_settings()
    model = model or settings.agent_model
    temperature = settings.agent_temperature if temperature is None else temperature

    response = get_client().chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    usage = response.usage
    if usage:
        log.info(
            "LLM tokens: %s (prompt=%s, completion=%s)",
            usage.total_tokens,
            usage.prompt_tokens,
            usage.completion_tokens,
        )
        _track_usage(usage.total_tokens)

    content = response.choices[0].message.content or "{}"
    try:
        return json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON from model: {exc}") from exc


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def chat_text(
    system: str,
    user: str,
    *,
    model: Optional[str] = None,
    max_tokens: int = 2000,
    temperature: Optional[float] = None,
) -> str:
    settings = get_settings()
    model = model or settings.supervisor_model
    temperature = settings.supervisor_temperature if temperature is None else temperature

    response = get_client().chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    usage = response.usage
    if usage:
        log.info(
            "LLM tokens: %s (prompt=%s, completion=%s)",
            usage.total_tokens,
            usage.prompt_tokens,
            usage.completion_tokens,
        )
        _track_usage(usage.total_tokens)

    return response.choices[0].message.content or ""


def chat_json_with_repair(
    system: str,
    user: str,
    *,
    max_tokens: int = 1500,
    temperature: Optional[float] = None,
) -> dict:
    """Call chat_json with one repair attempt on schema/parse failure."""
    try:
        data = chat_json(system, user, max_tokens=max_tokens, temperature=temperature)
        if not isinstance(data, dict):
            raise ValueError("Expected JSON object")
        return data
    except (ValueError, json.JSONDecodeError) as first_error:
        log.warning("JSON repair attempt after: %s", first_error)
        repair_user = (
            user
            + "\n\nIMPORTANT: Your previous response was invalid JSON. "
            "Return ONLY a valid JSON object matching the schema in the system prompt."
        )
        return chat_json(system, repair_user, max_tokens=max_tokens, temperature=0.1)
