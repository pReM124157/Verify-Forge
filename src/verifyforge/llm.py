from __future__ import annotations

import os
import re
from typing import Protocol

DEFAULT_MODEL = "claude-sonnet-5-5"


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


def extract_python(text: str) -> str:
    """Return the first fenced python block, or the whole text if none."""
    m = re.search(r"```(?:python)?\n(.*?)```", text, re.S)
    return (m.group(1) if m else text).strip() + "\n"
