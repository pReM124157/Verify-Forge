from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import time
from typing import Protocol

DEFAULT_MODEL = "claude-sonnet-5-5"

# One system prompt per role. ReplayLLM routes recorded responses by these.
SYS_ARCHITECT = (
    "You are a software architect. Reply with exactly one JSON object and nothing else."
)
SYS_TIER1 = "You are the architect writing the acceptance tests. Reply with exactly one fenced python code block."
SYS_BUILDER = "You are a meticulous Python engineer. Reply with exactly one fenced python code block."
SYS_VERIFIER = "You are an independent adversarial verifier. Reply with exactly one fenced python code block."
SYS_TRIAGE = "You are an impartial test auditor. Reply with exactly one JSON object and nothing else."
SYS_PRESERVATION = "You are an independent requirements auditor. Reply with exactly one JSON object and nothing else."
SYS_SCAN_REVIEW = "You are a software verification analyst. Reply with exactly one JSON object and nothing else."
SYS_SCAN_CONTRACT = "You are an evidence-bound requirements analyst. Reply with exactly one JSON object and nothing else."
SYS_REPAIR = "You are a meticulous Python engineer fixing a failing module. Reply with exactly one fenced python code block."


class LLM(Protocol):
    """The one model-provider interface every agent uses. Providers: AnthropicLLM (API), ClaudeCLI, ReplayLLM (offline)."""

    def complete(self, system: str, prompt: str) -> str: ...


ModelProvider = LLM  # the provider abstraction, by its conceptual name

# ---- model routing ---------------------------------------------------------------------------------------------------
# The role of a call is identified by its system prompt. Implementation work goes to the builder model; every role that
# judges (architect, auditors, verifier, triage, replacement-test authors) goes to the judge model.
ROLE_OF = {SYS_ARCHITECT: "architect", SYS_PRESERVATION: "auditor", SYS_VERIFIER: "verifier", SYS_TRIAGE: "triage",
           SYS_TIER1: "tier1_author", SYS_BUILDER: "builder", SYS_REPAIR: "repair",
           SYS_SCAN_REVIEW: "repo_review", SYS_SCAN_CONTRACT: "contract_inference"}
BUILDER_ROLES = {"builder", "repair"}


def resolve_models(model: str | None = None, judge: str | None = None, builder: str | None = None,
                   env: dict | None = None) -> tuple[str, str]:
    """(judge_model, builder_model). Precedence: explicit judge/builder > VERIFYFORGE_JUDGE_MODEL/BUILDER_MODEL >
    explicit model > VERIFYFORGE_DEFAULT_MODEL (or the older VERIFYFORGE_MODEL) > the built-in default."""
    env = os.environ if env is None else env
    default = model or env.get("VERIFYFORGE_DEFAULT_MODEL") or env.get("VERIFYFORGE_MODEL") or DEFAULT_MODEL
    return (judge or env.get("VERIFYFORGE_JUDGE_MODEL") or default, builder or env.get("VERIFYFORGE_BUILDER_MODEL") or default)


def _env_number(name: str, default: float) -> float:
    try:
        v = float(os.environ.get(name, ""))
        return v if v > 0 else default
    except ValueError:
        return default


# ---- errors and secrets ----------------------------------------------------------------------------------------------
class ProviderError(RuntimeError):
    """A model-provider failure with a stable machine-readable code (PROVIDER_AUTH_ERROR, PROVIDER_RATE_LIMIT,
    PROVIDER_TIMEOUT, PROVIDER_BILLING, PROVIDER_ERROR). The run fails closed with this code as its reason."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(f"{code}: {redact(message)}")


_KEY_SHAPE = re.compile(r"sk-ant-[A-Za-z0-9_\-]{6,}")


def redact(text: str) -> str:
    """Remove API-key-shaped strings and the configured key value from any text that may be stored or shown."""
    text = _KEY_SHAPE.sub("[REDACTED]", str(text))
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    return text.replace(key, "[REDACTED]") if len(key) >= 8 else text


# ---- usage telemetry (never records secrets) -------------------------------------------------------------------------
class _Telemetry:
    def _telemetry_init(self) -> None:
        self.usage: list[dict] = []

    def _record(self, role: str, model: str, input_tokens: int, output_tokens: int, seconds: float) -> None:
        self.usage.append({"role": role, "model": model, "input_tokens": int(input_tokens or 0),
                           "output_tokens": int(output_tokens or 0), "seconds": round(seconds, 3)})

    def usage_summary(self) -> dict:
        roles: dict[str, dict] = {}
        for u in self.usage:
            r = roles.setdefault(u["role"], {"model": u["model"], "calls": 0, "input_tokens": 0, "output_tokens": 0, "seconds": 0.0})
            r["calls"] += 1
            r["input_tokens"] += u["input_tokens"]
            r["output_tokens"] += u["output_tokens"]
            r["seconds"] = round(r["seconds"] + u["seconds"], 3)
            if u["model"] not in r["model"].split(" + "):
                r["model"] += " + " + u["model"]
        tot = {k: sum(r[k] for r in roles.values()) for k in ("calls", "input_tokens", "output_tokens")}
        tot["seconds"] = round(sum(r["seconds"] for r in roles.values()), 3)
        return {"roles": roles, "total": tot} if roles else {}


class _Routed:
    judge_model: str
    builder_model: str

    def _route(self, system: str) -> tuple[str, str]:
        role = ROLE_OF.get(system, "other")
        return role, (self.builder_model if role in BUILDER_ROLES else self.judge_model)

    @property
    def model_label(self) -> str:
        return self.judge_model if self.judge_model == self.builder_model else f"judge {self.judge_model} / builder {self.builder_model}"


class AnthropicLLM(_Telemetry, _Routed):
    """Official Anthropic API provider (SDK). Reads ANTHROPIC_API_KEY from the environment; never stores or logs it.

    Roles are separated properly: VerifyForge's system prompt is the API `system`, the task data is the user message.
    Transport retries (network errors, 429, 5xx) are the SDK's bounded retries; semantic failures are never retried."""

    provider = "api"

    def __init__(self, model: str | None = None, *, judge_model: str | None = None, builder_model: str | None = None,
                 max_tokens: int | None = None, timeout: float | None = None, max_retries: int = 2, client=None):
        self.judge_model, self.builder_model = resolve_models(model, judge_model, builder_model)
        self.model = self.judge_model
        self.max_tokens = int(max_tokens or _env_number("VERIFYFORGE_MAX_OUTPUT_TOKENS", 8000))
        self.timeout = float(timeout or _env_number("VERIFYFORGE_MODEL_TIMEOUT_SECONDS", 300))
        self._telemetry_init()
        if client is None:
            import anthropic

            key = os.environ.get("ANTHROPIC_API_KEY")
            if not key:
                raise ProviderError("PROVIDER_AUTH_ERROR", "ANTHROPIC_API_KEY is not set")
            client = anthropic.Anthropic(api_key=key, timeout=self.timeout, max_retries=max_retries)
        self.client = client

    def complete(self, system: str, prompt: str) -> str:
        role, model = self._route(system)
        t0 = time.monotonic()
        try:
            resp = self.client.messages.create(model=model, max_tokens=self.max_tokens, system=system,
                                               messages=[{"role": "user", "content": prompt}])
        except Exception as e:
            raise self._translate(e, model) from None  # the SDK exception chain is dropped: nothing to leak
        blocks = getattr(resp, "content", None) or []
        text = "".join(getattr(b, "text", "") or "" for b in blocks if getattr(b, "type", "") == "text")
        if not text.strip():
            raise ProviderError("PROVIDER_ERROR", f"the API returned no text content (stop_reason={getattr(resp, 'stop_reason', None)})")
        if getattr(resp, "stop_reason", None) == "max_tokens":
            raise ProviderError("PROVIDER_ERROR", f"the response was cut off at max_tokens={self.max_tokens}; raise VERIFYFORGE_MAX_OUTPUT_TOKENS")
        u = getattr(resp, "usage", None)
        self._record(role, model, getattr(u, "input_tokens", 0), getattr(u, "output_tokens", 0), time.monotonic() - t0)
        return text

    @staticmethod
    def _translate(e: Exception, model: str) -> ProviderError:
        import anthropic

        def is_a(*names):
            return any(isinstance(e, getattr(anthropic, n)) for n in names if hasattr(anthropic, n))

        msg = str(e)[:300]
        if is_a("AuthenticationError", "PermissionDeniedError"):
            return ProviderError("PROVIDER_AUTH_ERROR", f"authentication failed ({msg})")
        if is_a("RateLimitError"):
            return ProviderError("PROVIDER_RATE_LIMIT", f"rate limited ({msg})")
        if is_a("APITimeoutError") or isinstance(e, TimeoutError):
            return ProviderError("PROVIDER_TIMEOUT", "the request timed out")
        if is_a("NotFoundError"):
            return ProviderError("PROVIDER_ERROR", f"unsupported or unknown model id {model!r} ({msg})")
        if is_a("BadRequestError") and "credit" in msg.lower():
            return ProviderError("PROVIDER_BILLING", f"insufficient credits ({msg})")
        if is_a("APIStatusError"):
            return ProviderError("PROVIDER_ERROR", f"API error {getattr(e, 'status_code', '?')} ({msg})")
        if is_a("APIConnectionError"):
            return ProviderError("PROVIDER_ERROR", f"could not reach the API ({msg})")
        return ProviderError("PROVIDER_ERROR", f"{type(e).__name__}: {msg}")


class ClaudeCLI(_Telemetry, _Routed):
    """Model transport via headless Claude Code (`claude -p`), using its existing login: no API key needed.

    Claude only produces text. Tools are disabled and it runs from an empty temp dir, so it cannot read the
    repo or touch the filesystem; VerifyForge's own runner does all execution.
    (`--bare` is deliberately not used: it ignores OAuth/keychain login and would require an API key.)
    ANTHROPIC_API_KEY is removed from the subprocess environment, so this provider always uses your Claude login and
    never silently bills an API key."""

    provider = "cli"

    def __init__(self, model: str | None = None, timeout: float | None = None, binary: str = "claude", *,
                 judge_model: str | None = None, builder_model: str | None = None):
        self.judge_model, self.builder_model = resolve_models(model, judge_model, builder_model)
        self.model = self.judge_model
        self.timeout = float(timeout or _env_number("VERIFYFORGE_MODEL_TIMEOUT_SECONDS", 300))
        self.binary = binary
        self._telemetry_init()

    def complete(self, system: str, prompt: str) -> str:
        role, model = self._route(system)
        cmd = [self.binary, "-p", prompt, "--output-format", "json", "--model", model,
               "--system-prompt", system, "--tools", "", "--no-session-persistence", "--disable-slash-commands"]
        env = {k: v for k, v in os.environ.items() if k not in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")}
        t0 = time.monotonic()
        with tempfile.TemporaryDirectory(prefix="vf_model_") as cwd:
            try:
                p = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout, cwd=cwd, env=env)
            except FileNotFoundError as e:
                raise RuntimeError("`claude` CLI not found; install Claude Code or use --provider api") from e
            except OSError as e:  # e.g. E2BIG: the request is too large to pass on the command line
                raise RuntimeError(f"could not start the claude CLI ({type(e).__name__}: {e}); the request may be too large") from e
            except subprocess.TimeoutExpired as e:
                raise TimeoutError(f"claude CLI timed out after {self.timeout:g}s") from e
        try:
            try:
                payload = json.loads(p.stdout)
            except json.JSONDecodeError:  # tolerate warning lines printed before the JSON document
                payload = json.loads(next(l for l in reversed(p.stdout.strip().splitlines()) if l.lstrip().startswith("{")))
        except (json.JSONDecodeError, StopIteration) as e:
            raise RuntimeError(f"claude CLI returned non-JSON (exit {p.returncode}): {(p.stdout or p.stderr)[:300]}") from e
        if p.returncode != 0 or payload.get("is_error"):
            raise RuntimeError(f"claude CLI error (exit {p.returncode}): {str(payload.get('result', p.stderr))[:300]}")
        result = payload.get("result")
        if not isinstance(result, str) or not result.strip():
            raise RuntimeError("claude CLI returned an empty result")
        u = payload.get("usage") or {}
        self._record(role, model, u.get("input_tokens", 0), u.get("output_tokens", 0), time.monotonic() - t0)
        return result


AnthropicAPIProvider = AnthropicLLM
ClaudeCLIProvider = ClaudeCLI


def health_check(llm) -> dict:
    """A tiny round trip through the provider abstraction: proves auth, the routed model and latency."""
    t0 = time.monotonic()
    text = llm.complete("You are a connectivity check. Follow the instruction exactly.", "Reply with exactly:\nVERIFYFORGE_API_OK")
    seconds = time.monotonic() - t0
    last = (getattr(llm, "usage", None) or [{}])[-1]
    return {"ok": "VERIFYFORGE_API_OK" in text, "reply": text.strip()[:60], "seconds": round(seconds, 2),
            "model": last.get("model") or getattr(llm, "model", ""), "input_tokens": last.get("input_tokens"),
            "output_tokens": last.get("output_tokens")}


class ReplayLLM:
    """Replays saved model outputs per role (the offline provider). Only the network call is replaced; everything
    downstream (orchestrator, file writes, pytest) is real. Makes no network or CLI calls."""

    provider = "offline"
    model_label = "replayed"

    def __init__(self, responses: dict[str, list[str]]):
        self._queues = {k: list(v) for k, v in responses.items()}

    def complete(self, system: str, prompt: str) -> str:
        q = self._queues.get(system)
        if not q:
            raise RuntimeError(f"ReplayLLM has no recorded response left for role: {system[:40]!r}")
        return q.pop(0)


OfflineReplayProvider = ReplayLLM


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
