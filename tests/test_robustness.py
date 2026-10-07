import sys
from pathlib import Path

import pytest

from test_orchestrator import ADV, BAD, GOOD, SPEC, UNIT, FakeLLM
from verifyforge.orchestrator import run, run_request
from verifyforge.demos import RATE_LIMITER_REQUEST, rate_limiter_replay
from verifyforge.sandbox import run_pytest
from verifyforge.spec import parse_spec


def test_runner_timeout():
    r = run_pytest({"test_x.py": "import time\ndef test_a():\n    time.sleep(10)\n"}, timeout=1)
    assert not r.passed and "TIMEOUT" in r.output


def test_syntax_error_in_generated_code_fails_not_crashes(tmp_path: Path):
    r = run(FakeLLM("def add(a, b) return a + b"), SPEC, tmp_path, max_repairs=0)
    assert r.status == "UNVERIFIED" and not r.rounds[0].tier1_passed


def test_prose_instead_of_code_is_unverified(tmp_path: Path):
    r = run(FakeLLM("Sorry, I cannot help with that."), SPEC, tmp_path, max_repairs=0)
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

    r = run(Stuck(sneaky), SPEC, tmp_path, max_repairs=1)
    assert r.rounds[0].tier1_passed and r.rounds[0].adversarial_passed is False
    assert r.status == "UNVERIFIED"


def test_max_repair_count_respected(tmp_path: Path):
    class Counting(FakeLLM):
        def complete(self, system, prompt):
            if prompt.startswith("Fix module"):
                self.repairs += 1
                return f"```python\n{BAD}```"
            return super().complete(system, prompt)

    llm = Counting(BAD)
    r = run(llm, SPEC, tmp_path, max_repairs=2)
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

    r = run(Regress(half), SPEC, tmp_path, max_repairs=1)
    assert r.rounds[1].regression and r.status == "UNVERIFIED"


def test_earlier_tests_rerun_after_repair(tmp_path: Path):
    r = run(FakeLLM(BAD), SPEC, tmp_path)
    assert r.rounds[1].tier1_passed and r.rounds[1].adversarial_passed  # both suites reran on the repaired code


def test_request_drafts_spec_then_runs(tmp_path: Path):
    r = run_request(FakeLLM(GOOD), "make an adder", tmp_path)
    assert r.status == "VERIFIED" and r.request == "make an adder"
    assert (tmp_path / "specification.md").exists()


def test_malformed_drafted_spec_rejected(tmp_path: Path):
    class Bad(FakeLLM):
        def complete(self, system, prompt):
            return "no requirements here"

    with pytest.raises(ValueError, match="malformed JSON"):
        run_request(Bad(GOOD), "x", tmp_path)


def test_rate_limiter_spec_parses():
    s = parse_spec(Path(__file__).parent.parent.joinpath("examples/spec_rate_limiter.md").read_text())
    assert s.module == "rate_limiter" and len(s.requirements) == 8


def test_tests_never_see_the_implementation(tmp_path: Path):
    marker = "# IMPL_SECRET_MARKER"
    llm = FakeLLM(GOOD + marker + "\n")
    r = run_request(llm, "make an adder", tmp_path)
    assert r.status == "VERIFIED"
    seen_by_builder = [p for p in llm.prompts if p.startswith("Write module")]
    assert seen_by_builder and all(marker not in p for p in llm.prompts if not p.startswith("Fix module"))


def test_adversarial_generated_only_after_tier1_passes(tmp_path: Path):
    events = []
    run(FakeLLM(BAD), SPEC, tmp_path, on_event=lambda n, d: events.append(n))
    # BAD fails Tier-1 first; the adversarial suite is not generated until Tier-1 passes after repair.
    assert events.index("REPAIR_COMPLETE") < events.index("ADVERSARIAL_GENERATED")
    r = run(FakeLLM(BAD), SPEC, tmp_path)
    assert r.rounds[0].tier1_passed is False and r.rounds[0].adversarial_passed is None


def test_events_emitted_in_order(tmp_path: Path):
    events = []
    run(FakeLLM(GOOD), SPEC, tmp_path, on_event=lambda n, d: events.append(n))
    assert events[0] == "SPEC_GENERATED" and events[-1] == "VERIFIED"
    for a, b in [("BUILD_COMPLETE", "TIER1_RESULT"), ("TIER1_RESULT", "ADVERSARIAL_GENERATED"),
                 ("ADVERSARIAL_RESULT", "VERIFIED")]:
        assert events.index(a) < events.index(b)


def test_audit_artifacts_written(tmp_path: Path):
    run(FakeLLM(BAD), SPEC, tmp_path)
    for f in ["specification.md", "tier1_tests.py", "adversarial_tests.py", "solution_v1.py", "solution_v2.py",
              "repair_1.patch", "verification_report.json", "events.jsonl", "adder.py"]:
        assert (tmp_path / f).exists(), f


def test_runner_reports_real_numbers():
    r = run_pytest({"test_x.py": "def test_a():\n    assert 1\ndef test_b():\n    assert 1\n"})
    assert r.passed and r.exit_code == 0 and r.tests_run == 2 and r.duration > 0 and not r.timed_out


def test_architect_missing_key_rejected(tmp_path: Path):
    class Partial(FakeLLM):
        def complete(self, system, prompt):
            return '{"title": "x", "module": "x"}'

    with pytest.raises(ValueError, match="missing key"):
        run_request(Partial(GOOD), "x", tmp_path)


def test_offline_demo_runs_real_pytest_fail_then_repair(tmp_path: Path):
    events = []
    r = run_request(rate_limiter_replay(), RATE_LIMITER_REQUEST, tmp_path, on_event=lambda n, d: events.append((n, d)))
    assert r.status == "VERIFIED" and r.repairs == 1
    assert r.rounds[0].tier1_passed and r.rounds[0].adversarial_passed is False
    assert "observed" in r.rounds[0].output and "<= 70" in r.rounds[0].output
    assert "with self._lock" in (tmp_path / "rate_limiter.py").read_text()
    assert r.rounds[1].tier1_passed and r.rounds[1].adversarial_passed
