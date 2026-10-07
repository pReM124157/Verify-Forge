from __future__ import annotations

import json
import os
import re
from typing import Protocol

DEFAULT_MODEL = "claude-sonnet-5-5"

# One system prompt per role. ReplayLLM routes recorded responses by these.
SYS_ARCHITECT = (
    "You are a software architect. Reply with exactly one JSON object and nothing else."
)
SYS_TIER1 = "You are the architect writing the acceptance tests. Reply with exactly one fenced python code block."
SYS_BUILDER = "You are a meticulous Python engineer. Reply with exactly one fenced python code block."
SYS_VERIFIER = "You are an independent adversarial verifier. Reply with exactly one fenced python code block."
SYS_REPAIR = "You are a meticulous Python engineer fixing a failing module. Reply with exactly one fenced python code block."


class LLM(Protocol):
    def complete(self, system: str, prompt: str) -> str: ...


class AnthropicLLM:
    def __init__(self, model: str | None = None, max_tokens: int = 8000):
        import anthropic

        self.client = anthropic.Anthropic()
        self.model = model or os.environ.get("VERIFYFORGE_MODEL", DEFAULT_MODEL)
        self.max_tokens = max_tokens

    def complete(self, system: str, prompt: str) -> str:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in resp.content if b.type == "text")


class ReplayLLM:
    """Replays saved model outputs per role. Only the network call is replaced; everything
    downstream (orchestrator, file writes, pytest) is real."""

    def __init__(self, responses: dict[str, list[str]]):
        self._queues = {k: list(v) for k, v in responses.items()}

    def complete(self, system: str, prompt: str) -> str:
        q = self._queues.get(system)
        if not q:
            raise RuntimeError(f"ReplayLLM has no recorded response left for role: {system[:40]!r}")
        return q.pop(0)


def extract_python(text: str) -> str:
    """Return the first fenced python block, or the whole text if none."""
    m = re.search(r"```(?:python)?\n(.*?)```", text, re.S)
    return (m.group(1) if m else text).strip() + "\n"


def extract_json(text: str) -> dict:
    """Parse a JSON object from a reply, tolerating code fences and surrounding prose."""
    text = text.strip()
    m = re.search(r"```(?:json)?\n(.*?)```", text, re.S)
    if m:
        text = m.group(1)
    else:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            text = text[start : end + 1]
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Model returned malformed JSON: {e}") from e
    if not isinstance(data, dict):
        raise ValueError("Model returned JSON that is not an object.")
    return data
