"""Repository Scan Mode: read-only, evidence-first. Deterministic fixture repositories and a scripted model throughout;
no network and no paid API."""
import io
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from rich.console import Console

from scan_fixtures import (CALC_CONTRACT, CALC_HIDDEN, FAKE_SECRET, INV_CONTRACT, INV_HIDDEN, ScanFx, broken_tests, calculator,
                           insufficient, inventory, missing_dependency, no_python, no_tests, snapshot, threaded_and_plain, with_secrets, write)
from verifyforge import cli
from verifyforge.llm import (SYS_BUILDER, SYS_REPAIR, SYS_SCAN_CONTRACT, SYS_SCAN_REVIEW, SYS_TRIAGE, SYS_VERIFIER, AnthropicLLM,
                             ProviderError)
from verifyforge.scan import scan_repository
from verifyforge.scan.contracts import MAX_CONTEXT_CHARS, build_review_prompt, validate_contract
from verifyforge.scan.discovery import api_map, discover
from verifyforge.scan.risk import priority_score, rank_deterministically
from verifyforge.scan.safety import read_text_safe, redact_secrets, resolve_inside
from verifyforge.scan.ui import ScanUI

JUDGE, BUILDER = "judge-model-X", "builder-model-Y"


def scan(tmp_path, repo, llm, **kw):
    out = tmp_path / "out"
    kw.setdefault("timeout", 120)
    return scan_repository(repo, llm, out, **kw), out


def inv_fx(**kw):
    return ScanFx([("src/inventory.py", 9.0)], {"src/inventory.py": INV_CONTRACT}, {"inventory": INV_HIDDEN}, **kw)


# ===== 1-2. CLI =========================================================================================================

def test_scan_command_parses_the_path_and_runs_end_to_end(tmp_path, monkeypatch, capsys):
    repo = calculator(tmp_path / "repo")
    fx = ScanFx([("src/calculator.py", 8.0)], {"src/calculator.py": CALC_CONTRACT}, {"calculator": CALC_HIDDEN})
    monkeypatch.setattr(cli, "_make_llm", lambda *a, **k: fx)
    rc = cli.main(["scan", str(repo), "--provider", "api", "--max-modules", "1", "--out", str(tmp_path / "o")])
    out = capsys.readouterr().out
    assert rc == 0 and "REPOSITORY SCAN COMPLETE" in out and "VERIFIED 1" in out and (tmp_path / "o" / "report.md").exists()
    assert "REPOSITORY VERIFIED" not in out


@pytest.mark.parametrize("arg,msg", [("/definitely/not/here", "does not exist"), (__file__, "not a directory")])
def test_invalid_path_is_a_clean_error_with_no_traceback(arg, msg, capsys):
    assert cli.main(["scan", arg]) == 2
    err = capsys.readouterr()
    assert msg in err.err and "Traceback" not in err.out + err.err


def test_a_repository_with_no_python_is_unsupported_not_a_crash(tmp_path):
    rep, out = scan(tmp_path, no_python(tmp_path / "r"), ScanFx([], {}, {}))
    assert rep["status"] == "UNSUPPORTED / NO PYTHON SOURCE FOUND" and rep["summary"]["python_modules"] == 0
    assert "REPOSITORY VERIFIED" not in (out / "report.md").read_text()


# ===== 3-8. deterministic discovery =====================================================================================

def test_python_files_found_and_ignored_directories_are_ignored(tmp_path):
    repo = calculator(tmp_path / "r")
    for rel in (".venv/lib/x.py", "venv/y.py", "node_modules/z.py", "build/b.py", "dist/d.py", "__pycache__/c.py", ".git/hooks/h.py",
                "htmlcov/i.py", "vendor/v.py", ".mypy_cache/m.py", "runs/old/r.py"):
        write(repo, rel, "x = 1\n")
    m = discover(repo)
    paths = {x["path"] for x in m["modules"]} | {t["path"] for t in m["tests"]}
    assert paths == {"src/calculator.py", "tests/test_calculator.py"}
    assert m["language"] == "python" and m["source_files"] == 1 and m["test_files"] == 1 and m["test_framework"] == "pytest"
    assert "pyproject.toml" in m["config_files"] and "README.md" in m["documentation_files"]


def test_public_functions_classes_methods_and_signatures_come_from_the_ast(tmp_path):
    repo = inventory(tmp_path / "r")
    write(repo, "src/inventory.py", (repo / "src/inventory.py").read_text()
          + "\n\ndef _internal(x):\n    return x\n\n\nasync def fetch(url, *, retries: int = 3) -> bytes:\n    return b''\n")
    m = discover(repo)
    mod = next(x for x in m["modules"] if x["path"] == "src/inventory.py")
    cls = next(c for c in mod["classes"] if c["name"] == "Inventory")
    assert cls["public"] and {x["name"] for x in cls["methods"]} >= {"reserve", "add_stock", "available", "__init__"}
    assert any(x["signature"] == "reserve(self, product_id, quantity)" for x in cls["methods"])
    fns = {f["name"]: f for f in mod["functions"]}
    assert fns["fetch"]["async"] and fns["fetch"]["public"] and "retries: int=3" in fns["fetch"]["signature"] and "-> bytes" in fns["fetch"]["signature"]
    assert not fns["_internal"]["public"]
    assert "Inventory" in [c["name"] for c in api_map(m)["src/inventory.py"]["classes"]]


def test_existing_tests_are_discovered_and_mapped_to_modules(tmp_path):
    m = discover(inventory(tmp_path / "r"))
    assert m["tests"][0]["tests"] == ["test_reserve_when_enough_stock"]
    mapped = m["test_mapping"]["src/inventory.py"]
    assert mapped and mapped[0]["test_file"] == "tests/test_inventory.py" and mapped[0]["confidence"] == "high"


def test_discovery_never_executes_repository_code(tmp_path):
    repo = calculator(tmp_path / "r")
    write(repo, "src/boom.py", f"import pathlib\npathlib.Path({str(tmp_path / 'EXECUTED')!r}).write_text('x')\n")
    discover(repo)
    assert not (tmp_path / "EXECUTED").exists()


# ===== 9-12. the repository's own tests =====================================================================================

def test_existing_pytest_suite_runs_and_is_recorded(tmp_path):
    rep, out = scan(tmp_path, calculator(tmp_path / "r"), ScanFx([("src/calculator.py", 5)], {}, {}))
    s = rep["existing_suite"]
    assert s["status"] == "PASS" and s["collected"] == 2 and s["passed"] == 2 and s["failed"] == 0 and s["exit_code"] == 0
    assert "shell" in s["command"] and (out / "existing_test_result.json").exists()


def test_failed_existing_suite_is_recorded_and_deep_verification_is_skipped(tmp_path):
    fx = ScanFx([("src/calculator.py", 8.0)], {"src/calculator.py": CALC_CONTRACT}, {"calculator": CALC_HIDDEN})
    rep, _ = scan(tmp_path, broken_tests(tmp_path / "r"), fx)
    assert rep["existing_suite"]["status"] == "FAIL" and rep["existing_suite"]["failed"] == 1
    assert rep["summary"]["verified"] == 0 and rep["summary"]["deeply_verified"] == 0
    assert "EXISTING_SUITE_FAILED" in rep["modules"]["src/calculator.py"]["reason"]
    assert fx.n(SYS_VERIFIER) == 0  # no hidden tests were generated while the repository's own tests fail


def test_existing_suite_timeout_is_recorded(tmp_path):
    repo = calculator(tmp_path / "r")
    write(repo, "tests/test_slow.py", "import time\n\n\ndef test_slow():\n    time.sleep(30)\n")
    rep, _ = scan(tmp_path, repo, ScanFx([("src/calculator.py", 5)], {}, {}), timeout=3)
    assert rep["existing_suite"]["status"] == "TIMEOUT" and rep["existing_suite"]["timed_out"]


def test_a_repository_with_no_tests_can_never_be_called_verified(tmp_path):
    fx = ScanFx([("src/calculator.py", 8.0)], {"src/calculator.py": CALC_CONTRACT}, {"calculator": CALC_HIDDEN})
    rep, out = scan(tmp_path, no_tests(tmp_path / "r"), fx)
    md = (out / "report.md").read_text()
    assert rep["existing_suite"]["status"] == "NO_TESTS" and "REPOSITORY VERIFIED" not in md and "REPOSITORY VERIFIED" not in rep["status"]
    assert "not a repository-wide verification" in md  # the scope disclaimer is in the report


def test_missing_third_party_dependency_is_unavailable_and_nothing_is_installed(tmp_path):
    repo = missing_dependency(tmp_path / "r")
    pipe_contract = {"requirements": [{"id": "IR1", "text": "run(x) runs x through the pipeline", "confidence": "high", "sources": [
        {"kind": "docstring", "file": "src/pipeline.py", "quote": "Run x through the pipeline."},
        {"kind": "docstring", "file": "src/pipeline.py", "quote": "Pipeline using a third-party library."}]}], "insufficient_reason": ""}
    rep, _ = scan(tmp_path, repo, ScanFx([("src/pipeline.py", 8.0)], {"src/pipeline.py": pipe_contract}, {"pipeline": CALC_HIDDEN}))
    assert rep["existing_suite"]["status"] == "UNAVAILABLE" and "DEPENDENCY_MISSING" in rep["existing_suite"]["reason"]
    assert "not_installed_pkg_xyz" in rep["existing_suite"]["reason"] and rep["summary"]["verified"] == 0
    assert rep["modules"]["src/pipeline.py"]["status"] == "UNSUPPORTED"  # the import probe reports it cleanly


# ===== 13-15. risk scoring ====================================================================================================

def test_risk_score_is_deterministic_and_labelled_not_a_vulnerability_score(tmp_path):
    m = discover(threaded_and_plain(tmp_path / "r"))
    a, b = rank_deterministically(m), rank_deterministically(m)
    assert a == b and "not a vulnerability score" in a[0]["label"]


def test_threading_and_money_raise_verification_priority(tmp_path):
    m = discover(threaded_and_plain(tmp_path / "r"))
    score = {r["path"]: r for r in rank_deterministically(m)}
    assert score["src/shared.py"]["score"] > score["src/plain.py"]["score"]
    assert {"threading", "lock", "global_state"} <= set(score["src/shared.py"]["high_signals"])
    assert "decimal" in score["src/money.py"]["high_signals"] and "money" in score["src/money.py"]["high_signals"]
    assert score["src/money.py"]["score"] > score["src/plain.py"]["score"]


# ===== 16-17. contract provenance =================================================================================================

def test_contract_provenance_is_checked_against_the_real_files_and_caps_confidence(tmp_path):
    repo = inventory(tmp_path / "r")
    m = discover(repo)
    mod = next(x for x in m["modules"] if x["path"] == "src/inventory.py")
    from verifyforge.scan.contracts import gather_evidence
    ev = gather_evidence(repo, m, mod)
    c = validate_contract(json.dumps(INV_CONTRACT), ev, "src/inventory.py")
    q = c["requirements"][0]
    assert q["confidence"] == "HIGH" and q["validated_sources"] == 2 and all(s["valid"] for s in q["sources"])
    fake = {"requirements": [{"id": "IR1", "text": "reserve is idempotent", "confidence": "high",
                              "sources": [{"kind": "readme", "file": "README.md", "quote": "reserve is always idempotent and retried safely"},
                                          {"kind": "test", "file": "tests/test_nope.py", "quote": "def test_idempotent_reserve"}]}]}
    c2 = validate_contract(json.dumps(fake), ev, "src/inventory.py")
    r = c2["requirements"][0]
    assert r["confidence"] == "INSUFFICIENT" and r["claimed_confidence"] == "HIGH" and r["downgraded"] and c2["module_confidence"] == "INSUFFICIENT"
    one = {"requirements": [{"id": "IR1", "text": "x", "confidence": "high", "sources": [INV_CONTRACT["requirements"][0]["sources"][0]]}]}
    assert validate_contract(json.dumps(one), ev, "p")["requirements"][0]["confidence"] == "MEDIUM"  # one source can never be HIGH


def test_low_confidence_contract_is_not_deeply_verified(tmp_path):
    repo = insufficient(tmp_path / "r")
    fx = ScanFx([("src/mystery.py", 9.0)], {}, {})  # the model finds no evidence for a contract
    rep, _ = scan(tmp_path, repo, fx)
    assert rep["modules"]["src/mystery.py"]["status"] == "INSUFFICIENT CONTRACT" and fx.n(SYS_VERIFIER) == 0
    assert rep["summary"]["insufficient_contract"] == 1 and rep["summary"]["verified"] == 0


def test_a_fabricated_contract_quote_makes_the_module_insufficient_not_verified(tmp_path):
    repo = inventory(tmp_path / "r")
    bad = {"requirements": [{"id": "IR1", "text": "stock never negative", "confidence": "high",
                             "sources": [{"kind": "readme", "file": "README.md", "quote": "this sentence is not in the README at all"},
                                         {"kind": "docstring", "file": "src/inventory.py", "quote": "invented docstring text that does not exist"}]}]}
    fx = ScanFx([("src/inventory.py", 9.0)], {"src/inventory.py": bad}, {"inventory": INV_HIDDEN})
    rep, _ = scan(tmp_path, repo, fx)
    assert rep["modules"]["src/inventory.py"]["status"] == "INSUFFICIENT CONTRACT" and fx.n(SYS_VERIFIER) == 0


# ===== 18, 22-27. selection, hidden verification, results ===========================================================================

def two_modules(tmp_path):
    repo = inventory(tmp_path / "r", good=True)
    write(repo, "src/calculator.py", (calculator(tmp_path / "c") / "src/calculator.py").read_text())
    write(repo, "README.md", (repo / "README.md").read_text() + "\nThe calculator adds, subtracts and divides numbers. Division by zero raises ZeroDivisionError.\n")
    return repo


def test_only_the_configured_number_of_modules_is_deeply_verified(tmp_path):
    repo = two_modules(tmp_path)
    fx = ScanFx([("src/inventory.py", 9.0), ("src/calculator.py", 8.0)], {"src/inventory.py": INV_CONTRACT, "src/calculator.py": CALC_CONTRACT},
                {"inventory": INV_HIDDEN, "calculator": CALC_HIDDEN})
    rep, _ = scan(tmp_path, repo, fx, max_modules=1)
    s = rep["summary"]
    assert s["deeply_verified"] == 1 and s["verified"] == 1 and s["not_deeply_verified"] == 1
    assert rep["modules"]["src/calculator.py"]["status"] == "NOT DEEPLY VERIFIED"
    rep2, _ = scan(tmp_path / "again", two_modules(tmp_path / "again"), fx, max_modules=2)
    assert rep2["summary"]["verified"] == 2


def test_a_race_condition_the_existing_tests_miss_makes_the_module_unverified_with_evidence(tmp_path):
    rep, out = scan(tmp_path, inventory(tmp_path / "r", good=False), inv_fx(), max_modules=1)
    m = rep["modules"]["src/inventory.py"]
    assert rep["existing_suite"]["status"] == "PASS"  # the normal tests pass...
    assert m["status"] == "UNVERIFIED" and "VALID_TEST_FAILED" in m["reason"]  # ...the independent hidden test does not
    assert "oversold" in m["failure_evidence"][0]["assertion"] and m["triage"][0]["status"] == "KEPT (valid)"
    assert rep["summary"]["unverified"] == 1 and rep["highest_risk_unresolved"] == "src/inventory.py"
    md = (out / "report.md").read_text()
    assert "oversold" in md and "REPOSITORY VERIFIED" not in md


def test_a_correct_concurrent_module_is_verified_only_after_real_tests_pass(tmp_path):
    rep, _ = scan(tmp_path, inventory(tmp_path / "r", good=True), inv_fx(), max_modules=1)
    m = rep["modules"]["src/inventory.py"]
    assert m["status"] == "VERIFIED" and m["hidden"]["passed"] and m["hidden"]["executed"] >= 1 and m["hidden"]["integrity"] == ""


def test_a_vacuous_hidden_suite_can_not_verify_a_module(tmp_path):
    skipped = "import pytest\nfrom inventory import Inventory\n\n\n@pytest.mark.skip\ndef test_ADV_x():\n    assert False\n"
    fx = ScanFx([("src/inventory.py", 9.0)], {"src/inventory.py": INV_CONTRACT}, {"inventory": skipped})
    rep, _ = scan(tmp_path, inventory(tmp_path / "r", good=False), fx, max_modules=1)
    assert rep["modules"]["src/inventory.py"]["status"] == "UNVERIFIED" and "TEST_INTEGRITY" in rep["modules"]["src/inventory.py"]["reason"]


def test_an_invalid_hidden_test_is_quarantined_and_replaced_by_triage(tmp_path):
    wrong = ("from calculator import add\n\n\ndef test_ADV_add_wrong():\n    assert add(2, 3) == 6\n\n\ndef test_ADV_add_ok():\n    assert add(1, 1) == 2\n")
    repl = "def test_ADV_replacement_1():\n    from calculator import add\n    assert add(2, 3) == 5\n"
    verdict = json.dumps({"verdict": "INVALID", "reason": "2 + 3 is 5", "spec_reference": "", "replacement_required": True})
    fx = ScanFx([("src/calculator.py", 8.0)], {"src/calculator.py": CALC_CONTRACT}, {"calculator": wrong}, triage=verdict, replacement=repl)
    rep, _ = scan(tmp_path, calculator(tmp_path / "r"), fx, max_modules=1)
    m = rep["modules"]["src/calculator.py"]
    assert m["status"] == "VERIFIED" and m["hidden"]["quarantined"] == 1 and m["triage"][0]["replacement"]


def test_ambiguous_triage_never_creates_verified(tmp_path):
    wrong = "from calculator import add\n\n\ndef test_ADV_unclear():\n    assert add(2, 3) == 6\n"
    amb = json.dumps({"verdict": "AMBIGUOUS", "reason": "the contract is silent", "spec_reference": "", "replacement_required": False})
    fx = ScanFx([("src/calculator.py", 8.0)], {"src/calculator.py": CALC_CONTRACT}, {"calculator": wrong}, triage=amb)
    rep, _ = scan(tmp_path, calculator(tmp_path / "r"), fx, max_modules=1)
    m = rep["modules"]["src/calculator.py"]
    assert m["status"] == "UNVERIFIED" and "AMBIGUOUS" in m["reason"] and rep["summary"]["verified"] == 0


def test_hidden_suite_cap_is_enforced_and_generated_vs_kept_is_disclosed(tmp_path):
    many = "from calculator import add\nimport pytest\n\n\n@pytest.mark.parametrize('v', list(range(60)))\ndef test_ADV_p(v):\n    assert add(v, 0) == v\n"
    fx = ScanFx([("src/calculator.py", 8.0)], {"src/calculator.py": CALC_CONTRACT}, {"calculator": many})
    rep, out = scan(tmp_path, calculator(tmp_path / "r"), fx, max_modules=1, max_adversarial=15)
    h = rep["modules"]["src/calculator.py"]["hidden"]
    assert h["generated"] == 60 and h["kept"] == 15 and h["executed"] == 15 and h["truncated"] and rep["modules"]["src/calculator.py"]["status"] == "VERIFIED"
    assert "60 generated / 15 kept / 15 executed" in (out / "report.md").read_text()


# ===== 19-21, 38. model routing: judge only, no Builder, no Repair =====================================================================

class FakeSDK:
    def __init__(self, backend):
        self.backend, self.calls = backend, []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, *, model, max_tokens, system, messages, **kw):
        self.calls.append((model, system))
        text = self.backend.complete(system, messages[0]["content"])
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)], stop_reason="end_turn",
                               usage=SimpleNamespace(input_tokens=100, output_tokens=40))


def test_scan_uses_the_judge_model_for_every_stage_and_never_a_builder_or_repair(tmp_path):
    sdk = FakeSDK(inv_fx())
    llm = AnthropicLLM(client=sdk, judge_model=JUDGE, builder_model=BUILDER)
    rep, out = scan(tmp_path, inventory(tmp_path / "r", good=False), llm, max_modules=1)
    systems = {s for _, s in sdk.calls}
    assert {SYS_SCAN_REVIEW, SYS_SCAN_CONTRACT, SYS_VERIFIER, SYS_TRIAGE} <= systems
    assert {m for m, _ in sdk.calls} == {JUDGE}  # one model, the judge, for every call
    assert SYS_BUILDER not in systems and SYS_REPAIR not in systems
    u = rep["model_usage"]
    assert set(u["roles"]) >= {"repo_review", "contract_inference", "verifier", "triage"} and u["total"]["input_tokens"] > 0
    assert "## Model usage" in (out / "report.md").read_text()


def test_the_cli_provider_also_works_for_scanning(tmp_path, monkeypatch):
    fx = inv_fx()

    def fake_run(cmd, **kw):
        system = cmd[cmd.index("--system-prompt") + 1]
        prompt = cmd[cmd.index("-p") + 1]
        return subprocess.CompletedProcess(cmd, 0, json.dumps({"result": fx.complete(system, prompt), "is_error": False,
                                                                "usage": {"input_tokens": 5, "output_tokens": 2}}), "")

    real = subprocess.run
    monkeypatch.setattr(subprocess, "run", lambda cmd, **kw: fake_run(cmd, **kw) if str(cmd[0]) == "claude" else real(cmd, **kw))
    llm = cli._make_llm("cli", None, "cli-judge", None)
    rep, _ = scan(tmp_path, inventory(tmp_path / "r", good=True), llm, max_modules=1)
    assert rep["modules"]["src/inventory.py"]["status"] == "VERIFIED"


# ===== 22-23. artifacts live outside the scanned repository; the repository is byte-identical ==========================================

def test_generated_tests_are_stored_outside_the_repository_and_the_repository_is_byte_identical(tmp_path):
    repo = inventory(tmp_path / "r", good=False)
    before = snapshot(repo)
    rep, out = scan(tmp_path, repo, inv_fx(), max_modules=1)
    assert snapshot(repo) == before and rep["repository_unchanged"] and rep["fingerprint_before"] == rep["fingerprint_after"]
    assert (out / "generated_tests" / "test_src__inventory_adversarial.py").exists()
    assert not list(repo.rglob("*adversarial*")) and not list(repo.rglob(".pytest_cache")) and not list(repo.rglob("__pycache__"))
    for name in ("repository_map.json", "api_map.json", "existing_tests.json", "existing_test_result.json", "risk_ranking.json", "events.jsonl",
                 "report.json", "report.md"):
        assert (out / name).exists(), name
    assert (out / "contracts" / "src__inventory.json").exists() and (out / "module_results" / "src__inventory.json").exists()


# ===== 29-31. provider failure ===================================================================================================================

def test_provider_error_during_contract_inference_fails_the_module_closed(tmp_path):
    exc = ProviderError("PROVIDER_RATE_LIMIT", "slow down")
    fx = ScanFx([("src/inventory.py", 9.0)], {"src/inventory.py": INV_CONTRACT}, {"inventory": INV_HIDDEN}, fail={SYS_SCAN_CONTRACT: (exc, 0)})
    rep, _ = scan(tmp_path, inventory(tmp_path / "r"), fx, max_modules=1)
    m = rep["modules"]["src/inventory.py"]
    assert m["status"] == "NOT COMPLETED" and "PROVIDER_RATE_LIMIT" in m["reason"] and rep["summary"]["verified"] == 0
    assert rep["status"] == "REPOSITORY SCAN INCOMPLETE — PROVIDER ERROR"


def test_provider_error_during_ranking_keeps_the_deterministic_scan_and_fabricates_nothing(tmp_path):
    exc = ProviderError("PROVIDER_AUTH_ERROR", "bad key")
    fx = ScanFx([], {}, {}, fail={SYS_SCAN_REVIEW: (exc, 0)})
    rep, out = scan(tmp_path, inventory(tmp_path / "r"), fx, max_modules=1)
    assert rep["status"] == "REPOSITORY SCAN INCOMPLETE — PROVIDER ERROR" and rep["summary"]["python_modules"] == 1
    assert rep["existing_suite"]["status"] == "PASS" and fx.n(SYS_VERIFIER) == 0
    assert rep["modules"]["src/inventory.py"]["llm_priority"] is None  # no AI ranking was invented
    assert rep["modules"]["src/inventory.py"]["status"] == "NOT COMPLETED" and (out / "repository_map.json").exists()


def test_a_later_provider_failure_keeps_the_modules_already_completed(tmp_path):
    repo = two_modules(tmp_path)
    exc = ProviderError("PROVIDER_TIMEOUT", "timed out")
    fx = ScanFx([("src/inventory.py", 9.0), ("src/calculator.py", 8.0)], {"src/inventory.py": INV_CONTRACT, "src/calculator.py": CALC_CONTRACT},
                {"inventory": INV_HIDDEN, "calculator": CALC_HIDDEN}, fail={SYS_SCAN_CONTRACT: (exc, 2)})
    rep, out = scan(tmp_path, repo, fx, max_modules=2)
    assert rep["modules"]["src/inventory.py"]["status"] == "VERIFIED"  # completed evidence is retained
    assert rep["modules"]["src/calculator.py"]["status"] == "NOT COMPLETED" and "PROVIDER_TIMEOUT" in rep["modules"]["src/calculator.py"]["reason"]
    assert (out / "module_results" / "src__inventory.json").exists() and rep["status"].startswith("REPOSITORY SCAN INCOMPLETE")


# ===== 31-32, 34-36. secrets, symlinks, traversal, binaries, bounded context ===========================================================

def test_api_key_is_redacted_everywhere(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_SECRET)
    exc = ProviderError("PROVIDER_AUTH_ERROR", f"invalid x-api-key {FAKE_SECRET}")
    fx = ScanFx([("src/inventory.py", 9.0)], {}, {}, fail={SYS_SCAN_CONTRACT: (exc, 0)})
    rep, out = scan(tmp_path, inventory(tmp_path / "r"), fx, max_modules=1)
    assert all(FAKE_SECRET not in p.read_text(errors="ignore") for p in out.rglob("*") if p.is_file())


def test_repository_secrets_are_never_sent_to_the_model_or_written_to_reports(tmp_path):
    repo = with_secrets(tmp_path / "r")
    write(repo, "README.md", f"# calc\n\nDivision by zero raises ZeroDivisionError. token = '{FAKE_SECRET}'\n")
    fx = ScanFx([("src/calculator.py", 8.0), ("config.py", 7.0)], {"src/calculator.py": CALC_CONTRACT}, {"calculator": CALC_HIDDEN})
    rep, out = scan(tmp_path, repo, fx, max_modules=1)
    assert all(FAKE_SECRET not in p for _, p in fx.calls) and all("MIIFAKE" not in p for _, p in fx.calls)  # nothing secret-like reached the model
    assert all(FAKE_SECRET not in p.read_text(errors="ignore") for p in out.rglob("*") if p.is_file())
    assert ".env" in rep["repository"]["skipped"]["secret"] and "deploy.pem" in rep["repository"]["skipped"]["secret"]  # names only
    assert redact_secrets(f'x = "{FAKE_SECRET}"') == 'x = "[REDACTED]"' or FAKE_SECRET not in redact_secrets(FAKE_SECRET)


def test_a_symlink_pointing_outside_the_root_is_ignored(tmp_path):
    outside = tmp_path / "outside.py"
    outside.write_text("def stolen():\n    return 'secret'\n")
    repo = calculator(tmp_path / "r")
    (repo / "src" / "linked.py").symlink_to(outside)
    (repo / "linkdir").symlink_to(tmp_path)
    m = discover(repo)
    assert "src/linked.py" not in {x["path"] for x in m["modules"]} and m["skipped"]["symlink_outside"] >= 1
    assert not any("linkdir" in x["path"] for x in m["modules"])
    assert read_text_safe(repo, "src/linked.py") is None


def test_path_traversal_and_absolute_paths_are_rejected(tmp_path):
    repo = calculator(tmp_path / "r")
    (tmp_path / "secret.txt").write_text("outside")
    assert resolve_inside(repo, "../secret.txt") is None and resolve_inside(repo, str(tmp_path / "secret.txt")) is None
    assert read_text_safe(repo, "../secret.txt") is None and resolve_inside(repo, "src/calculator.py") is not None


def test_binary_files_are_ignored(tmp_path):
    repo = calculator(tmp_path / "r")
    (repo / "src" / "blob.py").write_bytes(b"\x00\x01\x02def x():\n    pass\n")
    m = discover(repo)
    assert "src/blob.py" not in {x["path"] for x in m["modules"]} and m["skipped"]["binary"] == 1


def test_a_large_repository_produces_a_bounded_model_context(tmp_path):
    repo = calculator(tmp_path / "r")
    for i in range(400):
        write(repo, f"src/pkg{i}/mod{i}.py", f'"""Module {i} with a fairly long docstring to inflate the summary line for the model prompt."""\n\n\n'
              + "\n".join(f"def func_{i}_{j}(a, b):\n    return a + b\n" for j in range(6)))
    m = discover(repo)
    prompt, meta = build_review_prompt(repo, m, rank_deterministically(m))
    assert len(prompt) <= MAX_CONTEXT_CHARS and meta["truncated"] and meta["modules_in_prompt"] < meta["modules_total"]
    assert all(f"mod{i}.py" not in prompt for i in range(300, 400)) or meta["truncated"]


# ===== 39-40. nothing else changed =========================================================================================================

def test_offline_demo_and_run_command_are_unchanged(tmp_path):
    assert cli.main(["demo", "rate-limiter", "--offline", "--out", str(tmp_path / "d")]) == 0
    r = json.loads((tmp_path / "d" / "verification_report.json").read_text())
    assert r["status"] == "VERIFIED" and r["repairs"] == 1


def test_scan_ui_renders_the_flow_and_never_claims_the_repository_is_verified(tmp_path):
    ui = ScanUI("LIVE · ANTHROPIC API", JUDGE, Console(file=io.StringIO(), width=132))
    rep, _ = scan(tmp_path, inventory(tmp_path / "r", good=False), inv_fx(), max_modules=1, on_event=ui)
    c = Console(file=io.StringIO(), force_terminal=True, width=132, height=34, color_system=None)
    c.print(ui.render())
    frame = c.file.getvalue()
    assert "REPOSITORY VERIFICATION" in frame and "JUDGE: judge-model-X" in frame and "MODE: LIVE · ANTHROPIC API" in frame and "src/inventory.py" in frame
    out = Console(file=io.StringIO(), force_terminal=True, width=100, color_system=None)
    out.print(ui.takeover_panel())
    shown = out.file.getvalue()
    assert "REPOSITORY SCAN COMPLETE" in shown and "UNVERIFIED" in shown and "Verification evidence generated for selected modules." in shown
    assert "Software earned verification" not in shown and "REPOSITORY VERIFIED" not in shown and ui.ui_errors == []


def test_a_syntactically_broken_generated_suite_is_retried_once_then_reported_precisely(tmp_path):
    broken = "from calculator import add\n\n\ndef test_ADV_x():\n    assert add(1, 1) == '\n"
    fx = ScanFx([("src/calculator.py", 8.0)], {"src/calculator.py": CALC_CONTRACT}, {"calculator": broken})
    rep, out = scan(tmp_path, calculator(tmp_path / "r"), fx, max_modules=1)
    m = rep["modules"]["src/calculator.py"]
    assert fx.n(SYS_VERIFIER) == 2  # exactly one regeneration
    assert m["status"] == "UNVERIFIED" and m["reason"].startswith("HIDDEN_SUITE_INVALID") and "NOT evidence" in m["reason"]
    assert "SyntaxError" in m["hidden"]["output_tail"] and "REPOSITORY VERIFIED" not in (out / "report.md").read_text()


def test_a_broken_suite_that_is_fixed_on_the_retry_verifies(tmp_path):
    class Flaky(ScanFx):
        def complete(self, system, prompt):
            if system == SYS_VERIFIER and not prompt.startswith("Replace") and self.n(SYS_VERIFIER) == 0:
                self.calls.append((system, prompt))
                return "```python\ndef test_ADV_x(:\n```"
            return super().complete(system, prompt)

    fx = Flaky([("src/calculator.py", 8.0)], {"src/calculator.py": CALC_CONTRACT}, {"calculator": CALC_HIDDEN})
    rep, _ = scan(tmp_path, calculator(tmp_path / "r"), fx, max_modules=1)
    assert rep["modules"]["src/calculator.py"]["status"] == "VERIFIED"


def test_a_property_is_described_as_an_attribute_not_a_method(tmp_path):
    from verifyforge.scan.contracts import _module_api_text

    repo = tmp_path / "r"
    write(repo, "src/res.py", "class Result:\n    @property\n    def output(self) -> str:\n        \"\"\"Captured output.\"\"\"\n        return 'x'\n\n"
          "    @classmethod\n    def make(cls):\n        return cls()\n")
    m = discover(repo)
    mod = m["modules"][0]
    text = _module_api_text(mod)
    assert "@property output: str" in text and "WITHOUT calling" in text and "def output(self)" not in text
    assert "@classmethod def make(cls)" in text
    assert "output (property)" in api_map(m)["src/res.py"]["classes"][0]["methods"]
