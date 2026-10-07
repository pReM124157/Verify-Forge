import json
from pathlib import Path

from test_orchestrator import ADV, BAD, GOOD, SPEC, UNIT, FakeLLM
from verifyforge.orchestrator import run
from verifyforge.sandbox import failure_lines, run_pytest
from verifyforge.spec import MINIMAL_SCOPE, architect

WRONG = "\n\ndef test_ADV_wrong_arithmetic():\n    assert add(1, 1) == 3  # wrong: 1 + 1 is 2\n"
REPLACEMENT = "def test_ADV_replacement_1():\n    assert add(1, 1) == 2\n"
SNEAKY = "def add(a, b):\n    return 5 if (a, b) == (2, 3) else 1\n  # IMPL_SECRET_MARKER\n"
GENUINE = "\n\ndef test_ADV_zero_is_neutral():\n    assert add(0, 0) == 0\n"  # SNEAKY returns 1 -> real bug


def verdict(v, reason="because"):
    return json.dumps({"verdict": v, "reason": reason, "spec_reference": "add returns the sum",
                       "replacement_required": v == "INVALID"})


class Scripted(FakeLLM):
    def __init__(self, first, adv, triage_reply, repair_to=GOOD):
        super().__init__(first)
        self.adv, self.triage_reply, self.repair_to = adv, triage_reply, repair_to
        self.triage_calls = []

    def complete(self, system, prompt):
        if "test auditor" in system:
            self.prompts.append(prompt)
            self.triage_calls.append(prompt)
            return self.triage_reply
        if prompt.startswith("Replace an invalid test"):
            self.prompts.append(prompt)
            return f"```python\n{REPLACEMENT}```"
        if prompt.startswith("Fix module"):
            self.prompts.append(prompt)
            self.repairs += 1
            return f"```python\n{self.repair_to}```"
        if "adversary" in prompt:
            self.prompts.append(prompt)
            return f"```python\n{self.adv}```"
        return super().complete(system, prompt)


def test_invalid_test_is_quarantined_replaced_and_run_verifies(tmp_path: Path):
    llm = Scripted(GOOD, ADV + WRONG, verdict("INVALID", "1+1 is 2, not 3"))
    r = run(llm, SPEC, tmp_path)
    assert r.status == "VERIFIED" and r.quarantined == 1 and r.repairs == 1
    t = r.triage[0]
    assert t["status"] == "QUARANTINED" and t["replacement"] and "1+1" in t["reason"]
    assert "test_ADV_replacement_1" in (tmp_path / "adversarial_tests.py").read_text()
    md = (tmp_path / "verification_report.md").read_text()
    assert "Adversarial test triage" in md and "QUARANTINED" in md and "Replacement test generated: YES" in md
    assert json.loads((tmp_path / "quarantine.json").read_text())[0]["test"].endswith("wrong_arithmetic")


def test_triage_only_after_a_repair_has_failed_to_fix_it(tmp_path: Path):
    llm = Scripted(BAD, ADV, verdict("INVALID"))  # BAD fails Tier-1; repair fixes it; adversarial passes
    r = run(llm, SPEC, tmp_path)
    assert r.status == "VERIFIED" and llm.triage_calls == [] and r.triage == []
    events = []
    run(Scripted(GOOD, ADV + WRONG, verdict("INVALID")), SPEC, tmp_path, on_event=lambda n, d: events.append(n))
    assert events.index("REPAIR_COMPLETE") < events.index("TRIAGE_STARTED")


def test_valid_verdict_keeps_test_blocking(tmp_path: Path):
    llm = Scripted(SNEAKY, ADV + GENUINE, verdict("VALID"), repair_to=SNEAKY)
    r = run(llm, SPEC, tmp_path, max_repairs=2)
    assert r.status == "UNVERIFIED" and r.quarantined == 0
    assert r.triage[0]["status"] == "KEPT (valid)"
    assert len(llm.triage_calls) == len({t["test"] for t in r.triage}) == 2  # each failing test triaged exactly once


def test_ambiguous_and_malformed_verdicts_never_quarantine(tmp_path: Path):
    for reply in (verdict("AMBIGUOUS"), "I think it's fine", json.dumps({"verdict": "MAYBE"})):
        llm = Scripted(SNEAKY, ADV + GENUINE, reply, repair_to=SNEAKY)
        r = run(llm, SPEC, tmp_path, max_repairs=2)
        assert r.status == "UNVERIFIED" and r.quarantined == 0 and r.triage[0]["verdict"] == "AMBIGUOUS"


def test_triage_agent_never_sees_the_implementation(tmp_path: Path):
    llm = Scripted(SNEAKY, ADV + GENUINE, verdict("VALID"), repair_to=SNEAKY)
    run(llm, SPEC, tmp_path, max_repairs=2)
    assert llm.triage_calls and all("IMPL_SECRET_MARKER" not in p and "return 5 if" not in p for p in llm.triage_calls)
    assert "test_ADV_zero" in llm.triage_calls[0] and "SPECIFICATION" in llm.triage_calls[0]


def test_quarantined_test_excluded_but_suite_still_runs(tmp_path: Path):
    r = run(Scripted(GOOD, ADV + WRONG, verdict("INVALID")), SPEC, tmp_path)
    final = r.rounds[-1]
    assert final.adversarial_passed and "deselected" in final.output and final.tests_run == 3


def test_failed_test_ids_and_failure_lines_parsed():
    files = {"m.py": "", "test_x.py": "def test_ok():\n    assert 1\ndef test_bad():\n    assert 2 + 2 == 5\n"}
    r = run_pytest(files)
    assert r.failed_tests == ("test_x.py::test_bad",)
    assert "assert" in failure_lines(r.stdout, "test_x.py::test_bad")
    assert run_pytest(files, deselect=["test_x.py::test_bad"]).passed


def test_architect_prompt_enforces_minimal_scope_and_records_assumptions(tmp_path: Path):
    seen = []

    class A(FakeLLM):
        def complete(self, system, prompt):
            seen.append(prompt)
            return json.dumps({"title": "Add", "module": "adder", "specification": "add(a, b) returns the sum.",
                               "requirements": ["add returns the sum."], "assumptions": ["ints only"],
                               "tier1_tests": UNIT})

    spec_md, _ = architect(A(GOOD), "make an adder")
    assert "MINIMAL-SCOPE RULES" in seen[0] and "Do NOT" in seen[0] and "5-8" in seen[0]
    assert "## Assumptions\n- ints only" in spec_md and "R1" not in spec_md
    assert MINIMAL_SCOPE.strip() in seen[0]


def test_replacement_that_redefines_shared_helpers_is_rejected(tmp_path: Path):
    from verifyforge.adversary import replacement_test
    from verifyforge.spec import parse_spec

    spec = parse_spec(SPEC)

    def reply(code):
        class L:
            def complete(self, system, prompt):
                return f"```python\n{code}```"
        return L()

    ok = "import math\n\ndef test_ADV_replacement_1():\n    assert 1 + 1 == 2\n"
    shadow = "def fill(a, b, c):\n    pass\n\ndef test_ADV_replacement_1():\n    assert True\n"
    two = "def test_ADV_replacement_1():\n    pass\n\ndef test_other():\n    pass\n"
    wrong_name = "def test_something_else():\n    pass\n"
    assert replacement_test(reply(ok), spec, "t", "r", 1) is not None
    for bad in (shadow, two, wrong_name, "def broken(:"):
        assert replacement_test(reply(bad), spec, "t", "r", 1) is None


def test_run5_scenario_replacement_cannot_break_other_tests(tmp_path: Path):
    # Suite has a shared helper; the model's "replacement" tries to redefine it. It must be dropped (reported as NO),
    # and the remaining suite must still run against the original helper.
    adv = ADV.replace("from adder import add", "from adder import add\n\ndef helper(n):\n    return n\n") + \
        "\n\ndef test_ADV_uses_helper():\n    assert add(helper(2), 3) == 5\n" + WRONG

    class Shadowing(Scripted):
        def complete(self, system, prompt):
            if prompt.startswith("Replace an invalid test"):
                return "```python\ndef helper(a, b):\n    return 0\n\ndef test_ADV_replacement_1():\n    pass\n```"
            return super().complete(system, prompt)

    r = run(Shadowing(GOOD, adv, verdict("INVALID")), SPEC, tmp_path)
    assert r.status == "VERIFIED" and r.triage[0]["status"] == "QUARANTINED" and r.triage[0]["replacement"] is False
    assert "Replacement test generated: NO" in (tmp_path / "verification_report.md").read_text()
    assert "def helper(n)" in (tmp_path / "adversarial_tests.py").read_text()
