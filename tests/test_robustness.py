import sys
from pathlib import Path

import pytest

from test_orchestrator import ADV, BAD, GOOD, SPEC, UNIT, FakeLLM
from verifyforge.orchestrator import run, run_request
from verifyforge.sandbox import run_pytest
from verifyforge.spec import parse_spec


def test_runner_timeout():
    r = run_pytest({"test_x.py": "import time\ndef test_a():\n    time.sleep(10)\n"}, timeout=1)
    assert not r.passed and "TIMEOUT" in r.output


def test_syntax_error_in_generated_code_fails_not_crashes(tmp_path: Path):
    r = run(FakeLLM("def add(a, b) return a + b"), SPEC, tmp_path, max_rounds=1)
    assert r.status == "UNVERIFIED" and not r.rounds[0].unit_passed


def test_prose_instead_of_code_is_unverified(tmp_path: Path):
    r = run(FakeLLM("Sorry, I cannot help with that."), SPEC, tmp_path, max_rounds=1)
    assert r.status == "UNVERIFIED"


def test_api_failure_recorded_never_verified(tmp_path: Path):
    class Down:
        def complete(self, system, prompt):
            raise ConnectionError("API unreachable")

    r = run(Down(), SPEC, tmp_path)
    assert r.status == "UNVERIFIED" and "ConnectionError" in r.error
    assert "API unreachable" in (tmp_path / "verification_report.md").read_text()


def test_api_failure_mid_repair(tmp_path: Path):
    class DiesOnRepair(FakeLLM):
        def complete(self, system, prompt):
            if prompt.startswith("Fix module"):
                raise TimeoutError("boom")
            return super().complete(system, prompt)

    r = run(DiesOnRepair(BAD), SPEC, tmp_path)
    assert r.status == "UNVERIFIED" and "TimeoutError" in r.error


def test_adversarial_failure_alone_blocks_verified(tmp_path: Path):
    # Implementation passes the unit test but is wrong for the adversarial case.
    sneaky = "def add(a, b):\n    return 5 if (a, b) == (2, 3) else 1\n"
    class Stuck(FakeLLM):
        def complete(self, system, prompt):
            if prompt.startswith("Fix module"):
                return f"```python\n{sneaky}```"
            return super().complete(system, prompt)

    r = run(Stuck(sneaky), SPEC, tmp_path, max_rounds=2)
    assert r.rounds[0].unit_passed and not r.rounds[0].adversarial_passed
    assert r.status == "UNVERIFIED"


def test_max_repair_count_respected(tmp_path: Path):
    class Counting(FakeLLM):
        def complete(self, system, prompt):
            if prompt.startswith("Fix module"):
                self.repairs += 1
                return f"```python\n{BAD}```"
            return super().complete(system, prompt)

    llm = Counting(BAD)
    r = run(llm, SPEC, tmp_path, max_rounds=3)
    assert len(r.rounds) == 3 and llm.repairs == 2


def test_diff_recorded_and_in_report(tmp_path: Path):
    r = run(FakeLLM(BAD), SPEC, tmp_path)
    assert "-    return a - b" in r.rounds[1].diff and "+    return a + b" in r.rounds[1].diff
    assert "```diff" in (tmp_path / "verification_report.md").read_text()


def test_repair_regression_flagged(tmp_path: Path):
    # Round 1 passes unit but fails adversarial; the "repair" breaks unit too.
    half = "def add(a, b):\n    return 5 if (a, b) == (2, 3) else 1\n"
    worse = "def add(a, b):\n    return 0\n"

    class Regress(FakeLLM):
        def complete(self, system, prompt):
            if prompt.startswith("Fix module"):
                return f"```python\n{worse}```"
            return super().complete(system, prompt)

    r = run(Regress(half), SPEC, tmp_path, max_rounds=2)
    assert r.rounds[1].regression and r.status == "UNVERIFIED"


def test_earlier_tests_rerun_after_repair(tmp_path: Path):
    r = run(FakeLLM(BAD), SPEC, tmp_path)
    assert r.rounds[1].unit_passed and r.rounds[1].adversarial_passed  # both suites reran on the repaired code


def test_request_drafts_spec_then_runs(tmp_path: Path):
    class NL(FakeLLM):
        def complete(self, system, prompt):
            if "requirements engineer" in system:
                return SPEC
            return super().complete(system, prompt)

    r = run_request(NL(GOOD), "make an adder", tmp_path)
    assert r.status == "VERIFIED" and r.request == "make an adder"
    assert (tmp_path / "spec.md").exists()


def test_malformed_drafted_spec_rejected(tmp_path: Path):
    class Bad(FakeLLM):
        def complete(self, system, prompt):
            return "no requirements here"

    with pytest.raises(ValueError):
        run_request(Bad(GOOD), "x", tmp_path)


def test_rate_limiter_spec_parses():
    s = parse_spec(Path(__file__).parent.parent.joinpath("examples/spec_rate_limiter.md").read_text())
    assert s.module == "rate_limiter" and len(s.requirements) == 8
