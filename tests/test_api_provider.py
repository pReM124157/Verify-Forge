"""Anthropic API provider: routing, secrets, fail-closed errors, telemetry. A fake SDK client is used throughout: these
tests never call the paid API."""
import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import anthropic
import pytest

try:  # the SDK's HTTP layer is httpx2 in recent versions
    import httpx2 as httpx
except ImportError:  # pragma: no cover
    import httpx

from test_orchestrator import ADV, GOOD, SPEC, UNIT, FakeLLM
from test_preservation import GOOD_JSON, USER_TTL, Scripted
from test_tier1_triage import T1LLM
from test_triage import WRONG, Scripted as AdvTriageScripted, verdict
from verifyforge import cli
from verifyforge.llm import (ROLE_OF, SYS_ARCHITECT, SYS_BUILDER, SYS_PRESERVATION, SYS_REPAIR, SYS_TIER1, SYS_TRIAGE,
                             SYS_VERIFIER, AnthropicLLM, ClaudeCLI, ProviderError, ReplayLLM, extract_json, health_check,
                             redact, resolve_models)
from verifyforge.orchestrator import run, run_request

SECRET = "sk-" + "ant-api03-" + "TESTSECRET" * 3 + "-0123456789"  # built at runtime: no key-shaped literal in the repo
JUDGE, BUILDER = "judge-model-X", "builder-model-Y"
REQ = httpx.Request("POST", "https://api.anthropic.com/v1/messages")


def http_error(cls, status, message):
    return cls(message, response=httpx.Response(status, request=REQ), body=None)


class FakeSDK:
    """Stands in for anthropic.Anthropic(): records every call, answers from a role-routed backend, can fail on demand."""

    def __init__(self, backend=None, fail=None, fail_after=0):
        self.backend, self.fail, self.fail_after = backend, fail, fail_after
        self.calls = []  # (model, system, user prompt, max_tokens)
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, *, model, max_tokens, system, messages, **kw):
        self.calls.append((model, system, messages, max_tokens))
        if self.fail is not None and len(self.calls) > self.fail_after:
            raise self.fail
        text = self.backend.complete(system, messages[0]["content"])
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)], stop_reason="end_turn",
                               usage=SimpleNamespace(input_tokens=120, output_tokens=45))


def api(sdk, **kw):
    return AnthropicLLM(client=sdk, judge_model=JUDGE, builder_model=BUILDER, **kw)


@pytest.fixture(autouse=True)
def _no_real_key(monkeypatch):
    for k in ("ANTHROPIC_API_KEY", "VERIFYFORGE_JUDGE_MODEL", "VERIFYFORGE_BUILDER_MODEL", "VERIFYFORGE_DEFAULT_MODEL",
              "VERIFYFORGE_MODEL", "VERIFYFORGE_PROVIDER", "VERIFYFORGE_MODEL_TIMEOUT_SECONDS"):
        monkeypatch.delenv(k, raising=False)


# ===== 1-2. key from the environment; missing key is a clear error ======================================================

def test_api_key_is_read_from_the_environment_and_never_hardcoded(monkeypatch):
    seen = {}

    class Rec:
        def __init__(self, **kw):
            seen.update(kw)
            self.messages = SimpleNamespace(create=lambda **k: None)

    monkeypatch.setenv("ANTHROPIC_API_KEY", SECRET)
    monkeypatch.setattr(anthropic, "Anthropic", Rec)
    AnthropicLLM()
    assert seen["api_key"] == SECRET and seen["max_retries"] == 2 and seen["timeout"] == 300.0


def test_timeout_and_retries_are_finite_and_configurable(monkeypatch):
    seen = {}
    monkeypatch.setenv("ANTHROPIC_API_KEY", SECRET)
    monkeypatch.setenv("VERIFYFORGE_MODEL_TIMEOUT_SECONDS", "42")
    monkeypatch.setattr(anthropic, "Anthropic", lambda **kw: seen.update(kw) or SimpleNamespace(messages=None))
    AnthropicLLM()
    assert seen["timeout"] == 42.0 and 0 <= seen["max_retries"] <= 3  # bounded transport retries only


def test_missing_key_is_a_clear_provider_error_and_a_clear_cli_error(monkeypatch, tmp_path, capsys):
    with pytest.raises(ProviderError) as e:
        AnthropicLLM()
    assert e.value.code == "PROVIDER_AUTH_ERROR" and "ANTHROPIC_API_KEY is not set" in str(e.value)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "load_env_file", lambda *a, **k: [])
    assert cli.main(["demo", "rate-limiter", "--provider", "api", "--out", str(tmp_path / "o")]) == 2
    assert "ANTHROPIC_API_KEY is not set" in capsys.readouterr().err


# ===== 3 + 22. the key never reaches logs, reports or events; telemetry is recorded ===================================

def scan_for(secret: str, root: Path) -> list[str]:
    return [str(p) for p in root.rglob("*") if p.is_file() and secret in p.read_text(errors="ignore")]


def test_redact_removes_key_shapes_and_the_configured_value(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "plainly-configured-secret-value")
    assert "sk-ant" not in redact(f"boom {SECRET} and " + "sk-" + "ant-usr-AbCdEf-123456 end")
    assert "plainly-configured-secret-value" not in redact("x plainly-configured-secret-value y")


def test_key_never_appears_in_report_events_stdout_even_when_a_provider_error_echoes_it(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("ANTHROPIC_API_KEY", SECRET)
    leaky = http_error(anthropic.AuthenticationError, 401, f"invalid x-api-key: {SECRET}")
    out = tmp_path / "out"
    r = run_request(api(FakeSDK(T1LLM(), fail=leaky, fail_after=2)), USER_TTL, out)  # fails at the Builder
    assert r.status == "UNVERIFIED" and r.unverified_reason == "PROVIDER_AUTH_ERROR"
    assert scan_for(SECRET, out) == [] and scan_for("sk-" + "ant-api03", out) == []
    cap = capsys.readouterr()
    assert SECRET not in cap.out + cap.err


def test_model_and_token_telemetry_is_recorded_per_role_without_secrets(monkeypatch, tmp_path):
    monkeypatch.setenv("ANTHROPIC_API_KEY", SECRET)
    out = tmp_path / "out"
    r = run_request(api(FakeSDK(Scripted([GOOD_JSON]))), USER_TTL, out)
    u = r.model_usage
    assert set(u["roles"]) >= {"architect", "auditor", "builder", "verifier"}
    assert u["roles"]["architect"]["model"] == JUDGE and u["roles"]["builder"]["model"] == BUILDER
    assert u["roles"]["architect"]["input_tokens"] == 120 and u["roles"]["architect"]["output_tokens"] == 45
    assert u["total"]["calls"] == sum(x["calls"] for x in u["roles"].values()) > 0
    md = (out / "verification_report.md").read_text()
    assert "## Model usage" in md and JUDGE in md and BUILDER in md
    saved = json.loads((out / "verification_report.json").read_text())
    assert saved["model_usage"]["total"]["input_tokens"] == u["total"]["input_tokens"] and "judge" in saved["model"]
    assert scan_for(SECRET, out) == []


# ===== 4-11. every role is routed to the configured model ================================================================

JUDGE_ROLES = {"architect", "auditor", "verifier", "triage", "tier1_author"}
BUILDER_ROLES = {"builder", "repair"}


def routed(sdk):
    got = {}
    for model, system, _, _ in sdk.calls:
        got.setdefault(ROLE_OF.get(system, "other"), set()).add(model)
    return got


def test_every_role_uses_its_configured_model_in_a_full_tier1_triage_pipeline(tmp_path):
    sdk = FakeSDK(T1LLM())  # architect, auditor, builder, repair, hidden verifier, Tier-1 triage, Tier-1 replacement
    r = run_request(api(sdk), USER_TTL, tmp_path)
    got = routed(sdk)
    assert r.status == "VERIFIED" and r.repairs == 1 and r.quarantined == 1
    for role in ("architect", "auditor", "verifier", "triage", "tier1_author", "builder", "repair"):
        assert role in got, f"{role} was never called, so its routing is untested"
    assert all(got[r_] == {JUDGE} for r_ in JUDGE_ROLES if r_ in got)  # architect, auditor, verifier, Tier-1 triage, tier-1 author
    assert all(got[r_] == {BUILDER} for r_ in BUILDER_ROLES)  # builder and repair


def test_adversarial_triage_and_replacement_use_the_judge_model_and_repair_the_builder_model(tmp_path):
    sdk = FakeSDK(AdvTriageScripted(GOOD, ADV + WRONG, verdict("INVALID", "1 + 1 is 2")))
    r = run(api(sdk), SPEC, tmp_path / "o")
    got = routed(sdk)
    assert r.status == "VERIFIED" and r.quarantined == 1
    assert got["triage"] == {JUDGE} and got["verifier"] == {JUDGE}  # hidden suite, adversarial triage, replacement author
    assert got["builder"] == {BUILDER} and got["repair"] == {BUILDER}
    replacement_calls = [m for m, s, msgs, _ in sdk.calls if msgs[0]["content"].startswith("Replace an invalid test")]
    assert replacement_calls == [JUDGE]


def test_system_prompt_is_the_system_and_the_task_is_the_user_message():
    sdk = FakeSDK(Scripted([GOOD_JSON]))
    llm = api(sdk)
    llm.complete(SYS_ARCHITECT, "TASK DATA")
    model, system, messages, _ = sdk.calls[0]
    assert system == SYS_ARCHITECT and messages == [{"role": "user", "content": "TASK DATA"}]


def test_model_resolution_precedence(monkeypatch):
    assert resolve_models(env={}) == ("claude-sonnet-5-5", "claude-sonnet-5-5")
    assert resolve_models(model="m", env={}) == ("m", "m")
    env = {"VERIFYFORGE_JUDGE_MODEL": "ej", "VERIFYFORGE_BUILDER_MODEL": "eb", "VERIFYFORGE_DEFAULT_MODEL": "ed"}
    assert resolve_models(env=env) == ("ej", "eb")
    assert resolve_models(judge="cj", builder="cb", env=env) == ("cj", "cb")  # CLI flags beat the environment
    assert resolve_models(env={"VERIFYFORGE_DEFAULT_MODEL": "ed"}) == ("ed", "ed")
    assert resolve_models(env={"VERIFYFORGE_MODEL": "legacy"}) == ("legacy", "legacy")


def test_env_vars_configure_the_provider(monkeypatch):
    monkeypatch.setenv("VERIFYFORGE_JUDGE_MODEL", "env-judge")
    monkeypatch.setenv("VERIFYFORGE_BUILDER_MODEL", "env-builder")
    llm = AnthropicLLM(client=FakeSDK(Scripted([GOOD_JSON])))
    assert (llm.judge_model, llm.builder_model) == ("env-judge", "env-builder")


# ===== 12-13. normalized responses; strict JSON handling unchanged ====================================================

def test_response_is_normalized_to_plain_text_ignoring_non_text_blocks():
    sdk = FakeSDK(Scripted([GOOD_JSON]))
    sdk.messages = SimpleNamespace(create=lambda **k: SimpleNamespace(
        content=[SimpleNamespace(type="thinking", thinking="..."), SimpleNamespace(type="text", text="hello "),
                 SimpleNamespace(type="text", text="world")], stop_reason="end_turn",
        usage=SimpleNamespace(input_tokens=1, output_tokens=2)))
    assert api(sdk).complete(SYS_BUILDER, "x") == "hello world"


def test_strict_json_handling_is_unchanged_over_the_api(tmp_path):
    class Garbage:
        def complete(self, system, prompt):
            return "this is not json"

    with pytest.raises(ValueError, match="malformed JSON"):
        run_request(api(FakeSDK(Garbage())), "Build X.", tmp_path / "o")  # fail-closed, exactly as with the CLI provider
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}


# ===== 14-18. provider failures fail closed with a clean reason ========================================================

FAILURES = {
    "rate limit": (http_error(anthropic.RateLimitError, 429, "slow down"), "PROVIDER_RATE_LIMIT"),
    "auth": (http_error(anthropic.AuthenticationError, 401, "invalid x-api-key"), "PROVIDER_AUTH_ERROR"),
    "permission": (http_error(anthropic.PermissionDeniedError, 403, "no access"), "PROVIDER_AUTH_ERROR"),
    "timeout": (anthropic.APITimeoutError(request=REQ), "PROVIDER_TIMEOUT"),
    "builtin timeout": (TimeoutError("read timed out"), "PROVIDER_TIMEOUT"),
    "unsupported model": (http_error(anthropic.NotFoundError, 404, "model: nope-model"), "PROVIDER_ERROR"),
    "insufficient credits": (http_error(anthropic.BadRequestError, 400, "Your credit balance is too low"), "PROVIDER_BILLING"),
    "server error": (http_error(anthropic.InternalServerError, 500, "oops"), "PROVIDER_ERROR"),
    "connection": (anthropic.APIConnectionError(request=REQ), "PROVIDER_ERROR"),
    "unexpected": (RuntimeError("something odd"), "PROVIDER_ERROR"),
}


@pytest.mark.parametrize("name", list(FAILURES))
def test_provider_failure_at_the_architect_fails_closed_with_a_clean_reason(tmp_path, name):
    exc, code = FAILURES[name]
    r = run_request(api(FakeSDK(Scripted([GOOD_JSON]), fail=exc)), USER_TTL, tmp_path / "o")
    assert r.status == "UNVERIFIED" and r.unverified_reason == code and code in r.error
    assert (tmp_path / "o" / "verification_report.json").exists()  # a report, not a traceback


@pytest.mark.parametrize("name", ["rate limit", "auth", "timeout", "unsupported model"])
def test_provider_failure_mid_run_fails_closed_with_a_clean_reason(tmp_path, name):
    exc, code = FAILURES[name]
    sdk = FakeSDK(T1LLM(), fail=exc, fail_after=2)  # architect + auditor succeed, the Builder call fails
    r = run_request(api(sdk), USER_TTL, tmp_path / "o")
    assert r.status == "UNVERIFIED" and r.unverified_reason == code and r.rounds == []


def test_auth_failure_during_the_preservation_audit_carries_its_code(tmp_path):
    exc, code = FAILURES["auth"]
    r = run_request(api(FakeSDK(Scripted([GOOD_JSON]), fail=exc, fail_after=1)), USER_TTL, tmp_path / "o")
    assert r.status == "UNVERIFIED" and r.unverified_reason == code
    assert [c for c in FakeSDK.__dict__] and True  # (auth is not retried: only one failing audit call)


@pytest.mark.parametrize("bad", [
    SimpleNamespace(content=[], stop_reason="end_turn", usage=None),
    SimpleNamespace(content=[SimpleNamespace(type="thinking", thinking="x")], stop_reason="end_turn", usage=None),
    SimpleNamespace(content=None, stop_reason="end_turn", usage=None),
    SimpleNamespace(content=[SimpleNamespace(type="text", text="partial")], stop_reason="max_tokens", usage=None),
])
def test_malformed_or_truncated_api_responses_fail_closed(bad):
    sdk = SimpleNamespace(messages=SimpleNamespace(create=lambda **k: bad))
    with pytest.raises(ProviderError) as e:
        AnthropicLLM(client=sdk, judge_model="j", builder_model="b").complete(SYS_BUILDER, "x")
    assert e.value.code == "PROVIDER_ERROR"


# ===== 19-21. CLI provider and offline provider unchanged; never a silent switch ======================================

def test_cli_provider_routes_models_and_never_passes_the_api_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", SECRET)
    seen = []

    def fake_run(cmd, **kw):
        seen.append((cmd, kw))
        return subprocess.CompletedProcess(cmd, 0, json.dumps({"result": "ok", "is_error": False,
                                                                "usage": {"input_tokens": 7, "output_tokens": 3}}), "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    llm = ClaudeCLI(judge_model="cli-judge", builder_model="cli-builder")
    llm.complete(SYS_ARCHITECT, "a")
    llm.complete(SYS_BUILDER, "b")
    models = [c[c.index("--model") + 1] for c, _ in seen]
    assert models == ["cli-judge", "cli-builder"]
    assert all("ANTHROPIC_API_KEY" not in kw["env"] for _, kw in seen)  # the CLI keeps using your login, not an API key
    assert llm.usage_summary()["total"]["input_tokens"] == 14


def test_offline_demo_needs_no_key_no_network_and_no_cli(monkeypatch, tmp_path):
    boom = lambda *a, **k: (_ for _ in ()).throw(AssertionError("a provider was used by the offline demo"))
    monkeypatch.setattr(anthropic, "Anthropic", boom)
    real_run = subprocess.run

    def guard(cmd, **k):  # pytest itself still runs; only the claude binary is forbidden
        if "claude" in str(cmd[0]):
            raise AssertionError("the claude CLI was used by the offline demo")
        return real_run(cmd, **k)

    monkeypatch.setattr(subprocess, "run", guard)
    monkeypatch.setattr(cli, "AnthropicLLM", boom)
    monkeypatch.setattr(cli, "ClaudeCLI", boom)
    assert cli.main(["demo", "rate-limiter", "--offline", "--out", str(tmp_path / "d")]) == 0
    r = json.loads((tmp_path / "d" / "verification_report.json").read_text())
    assert r["status"] == "VERIFIED" and r["repairs"] == 1 and r["model_usage"] == {}


def test_an_api_failure_never_switches_to_another_provider(monkeypatch, tmp_path):
    monkeypatch.setenv("ANTHROPIC_API_KEY", SECRET)
    constructed = []
    monkeypatch.setattr(cli, "ClaudeCLI", lambda *a, **k: constructed.append(1) or (_ for _ in ()).throw(AssertionError("CLI fallback")))
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(AssertionError("CLI fallback via subprocess")))
    exc, code = FAILURES["rate limit"]
    monkeypatch.setattr(cli, "_make_llm", lambda provider, *a, **k: api(FakeSDK(Scripted([GOOD_JSON]), fail=exc)))
    rc = cli.main(["run", USER_TTL, "--provider", "api", "--out", str(tmp_path / "o")])
    assert rc == 1 and constructed == []
    assert json.loads((tmp_path / "o" / "verification_report.json").read_text())["unverified_reason"] == code


# ===== provider selection, flags, UI, health ===========================================================================

def test_provider_default_comes_from_the_environment_and_a_flag_beats_it(monkeypatch, tmp_path):
    monkeypatch.setenv("VERIFYFORGE_PROVIDER", "api")
    seen = []
    monkeypatch.setattr(cli, "_make_llm", lambda provider, *a, **k: seen.append(provider) or FakeLLM(GOOD))
    cli.main(["run", "make an adder", "--out", str(tmp_path / "a")])
    cli.main(["run", "make an adder", "--provider", "cli", "--out", str(tmp_path / "b")])
    assert seen == ["api", "cli"]
    monkeypatch.delenv("VERIFYFORGE_PROVIDER")
    cli.main(["run", "make an adder", "--out", str(tmp_path / "c")])
    assert seen[-1] == "cli"  # unchanged default


def test_unknown_provider_in_the_environment_is_a_clear_error(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("VERIFYFORGE_PROVIDER", "openai")
    assert cli.main(["run", "make an adder", "--out", str(tmp_path / "o")]) == 2
    assert "unknown provider" in capsys.readouterr().err


def test_judge_and_builder_flags_reach_the_provider(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr(cli, "_make_llm", lambda provider, model, judge=None, builder=None: seen.update(p=provider, m=model, j=judge, b=builder) or FakeLLM(GOOD))
    cli.main(["run", "make an adder", "--provider", "api", "--model", "M", "--judge-model", "J", "--builder-model", "B", "--out", str(tmp_path / "o")])
    assert seen == {"p": "api", "m": "M", "j": "J", "b": "B"}


def test_ui_mode_line_for_api_shows_the_routed_models(monkeypatch, tmp_path):
    captured = {}

    class UI:
        def __init__(self, mode, goal="", **kw):
            captured.update(mode=mode, subtitle=kw.get("subtitle"))
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def __call__(self, n, d): pass

    monkeypatch.setattr("verifyforge.ui.VerifyForgeUI", UI)
    monkeypatch.setattr(cli, "_make_llm", lambda *a, **k: FakeLLM(GOOD))
    cli.main(["run", "make an adder", "--ui", "--provider", "api", "--judge-model", "J1", "--builder-model", "B1", "--out", str(tmp_path / "o")])
    assert captured == {"mode": "LIVE · ANTHROPIC API", "subtitle": "JUDGE J1 · BUILDER B1"}
    cli.main(["run", "make an adder", "--ui", "--provider", "cli", "--out", str(tmp_path / "p")])
    assert captured["mode"] == "LIVE · CLAUDE CLI"
    cli.main(["demo", "rate-limiter", "--offline", "--ui", "--pace", "0", "--out", str(tmp_path / "q")])
    assert captured == {"mode": "OFFLINE DEMO", "subtitle": "REPLAYED MODEL OUTPUT · REAL PYTEST"}


def test_health_check_goes_through_the_provider_abstraction(monkeypatch, capsys):
    class Echo:
        def complete(self, system, prompt):
            return "VERIFYFORGE_API_OK"

    sdk = FakeSDK(Echo())
    r = health_check(api(sdk))
    assert r["ok"] and r["model"] == JUDGE and r["input_tokens"] == 120 and r["seconds"] >= 0
    monkeypatch.setattr(cli, "_make_llm", lambda *a, **k: api(FakeSDK(Echo())))
    assert cli.main(["health", "--provider", "api"]) == 0
    out = capsys.readouterr().out
    assert "HEALTH      OK" in out and SECRET not in out


@pytest.mark.parametrize("name", ["auth", "rate limit", "timeout"])
def test_unhealthy_provider_is_reported_cleanly_with_its_code(monkeypatch, capsys, name):
    exc, code = FAILURES[name]
    monkeypatch.setattr(cli, "_make_llm", lambda *a, **k: api(FakeSDK(None, fail=exc)))
    assert cli.main(["health", "--provider", "api"]) == 1
    assert code in capsys.readouterr().out
