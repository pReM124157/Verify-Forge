from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
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


class ClaudeCLI:
    """Model transport via headless Claude Code (`claude -p`), using its existing login: no API key needed.

    Claude only produces text. Tools are disabled and it runs from an empty temp dir, so it cannot read the
    repo or touch the filesystem; VerifyForge's own runner does all execution.
    (`--bare` is deliberately not used: it ignores OAuth/keychain login and would require an API key.)"""

    def __init__(self, model: str | None = None, timeout: int = 300, binary: str = "claude"):
        self.model = model or os.environ.get("VERIFYFORGE_MODEL", DEFAULT_MODEL)
        self.timeout = timeout
        self.binary = binary

    def complete(self, system: str, prompt: str) -> str:
        cmd = [self.binary, "-p", prompt, "--output-format", "json", "--model", self.model,
               "--system-prompt", system, "--tools", "", "--no-session-persistence", "--disable-slash-commands"]
        with tempfile.TemporaryDirectory(prefix="vf_model_") as cwd:
            try:
                p = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout, cwd=cwd)
            except FileNotFoundError as e:
                raise RuntimeError("`claude` CLI not found; install Claude Code or use --provider api") from e
            except subprocess.TimeoutExpired as e:
                raise TimeoutError(f"claude CLI timed out after {self.timeout}s") from e
        try:
            payload = json.loads(p.stdout)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"claude CLI returned non-JSON (exit {p.returncode}): {(p.stdout or p.stderr)[:300]}") from e
        if p.returncode != 0 or payload.get("is_error"):
            raise RuntimeError(f"claude CLI error (exit {p.returncode}): {str(payload.get('result', p.stderr))[:300]}")
        result = payload.get("result")
        if not isinstance(result, str) or not result.strip():
            raise RuntimeError("claude CLI returned an empty result")
        return result


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
