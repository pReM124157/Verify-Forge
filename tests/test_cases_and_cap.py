import json
from pathlib import Path

import pytest

from test_orchestrator import FakeLLM
from test_triage import Scripted, verdict
from verifyforge.adversary import MAX_ADVERSARIAL_TESTS, _round_robin, bounded_adversarial
from verifyforge.orchestrator import run
from verifyforge.sandbox import collect_tests, failure_lines, run_pytest
from verifyforge.spec import parse_spec
from verifyforge.triage import extract_test_source, resolve_case

PAL_SPEC = ("# Palindrome\n\nModule: palindrome\n\nis_palindrome(s: str) -> bool returns True iff s equals its exact "
            "reverse (direct string reversal; punctuation, spaces and case are all significant).\n\n"
            "## Requirements\n- is_palindrome(s) is True iff s equals its reverse.\n")
PAL_IMPL = "def is_palindrome(s):\n    return s == s[::-1]\n\n# IMPL_SECRET_MARKER\n"
PAL_TIER1 = ("from palindrome import is_palindrome\n\ndef test_R1_basic():\n    assert is_palindrome('racecar')\n"
             "    assert not is_palindrome('abc')\n")
PAL_ADV = '''import pytest
from palindrome import is_palindrome

@pytest.mark.parametrize("s,expected", [("racecar", True), ("a.b,a", True), ("hello world", False), ("", True)])
def test_ADV_whitespace_punctuation_significant(s, expected):
    assert is_palindrome(s) is expected
'''
PAL_REPLACEMENT = ("def test_ADV_replacement_1():\n    from palindrome import is_palindrome\n"
                   "    assert is_palindrome('a.b,a') is False  # direct reversal: 'a,b.a' != 'a.b,a'\n")


def many(n_funcs, per_func):
    out = "import pytest\nfrom palindrome import is_palindrome\n\n"
    for f in range(n_funcs):
        cases = ", ".join(f"('x{f}{i}', False)" for i in range(per_func))
        out += f'@pytest.mark.parametrize("s,expected", [{cases}])\ndef test_ADV_f{f}(s, expected):\n    assert is_palindrome(s) is expected\n\n'
    return out


# ---- Fix 1: triage sees the exact parametrized case ---------------------------------------------------------

def test_failure_lines_survive_dots_in_case_id():
    r = run_pytest({"palindrome.py": PAL_IMPL, "test_adversarial.py": PAL_ADV})
    assert r.failed_tests == ("test_adversarial.py::test_ADV_whitespace_punctuation_significant[a.b,a-True]",)
    fl = failure_lines(r.stdout, r.failed_tests[0])
    assert "False is True" in fl and "is_palindrome('a.b,a')" in fl


def test_failed_ids_with_spaces_are_not_truncated():
    adv = PAL_ADV.replace('("hello world", False)', '("hello world", True)')
    r = run_pytest({"palindrome.py": PAL_IMPL, "test_adversarial.py": adv})
    assert any(t.endswith("[hello world-True]") for t in r.failed_tests)


@pytest.mark.parametrize("node,expected", [
    ("t.py::test_x[a.b,a-True]", {"s": "'a.b,a'", "expected": "True"}),
    ("t.py::test_x[hello world-False]", {"s": "'hello world'", "expected": "False"}),
    ("t.py::test_x[-True]", {"s": "''", "expected": "True"}),
    ("t.py::test_x[nope-True]", None),
    ("t.py::test_x", None),
])
def test_resolve_case(node, expected):
    assert resolve_case(PAL_ADV.replace("test_ADV_whitespace_punctuation_significant", "test_x"), node) == expected


def test_resolve_case_pytest_param_ids_and_stacked_decorators():
    src = ('import pytest\n@pytest.mark.parametrize("a", [1, 2])\n@pytest.mark.parametrize("b", ["x", "y"])\n'
           'def test_s(a, b): pass\n'
           '@pytest.mark.parametrize("v", [pytest.param(5, id="five"), pytest.param(6, id="six")])\n'
           'def test_p(v): pass\n')
    assert resolve_case(src, "t.py::test_s[y-2]") == {"b": "'y'", "a": "2"}  # bottom decorator id comes first
    assert resolve_case(src, "t.py::test_p[six]") == {"v": "6"}


def test_extracted_source_includes_the_parametrize_decorator():
    src, _ = extract_test_source(PAL_ADV, "t.py::test_ADV_whitespace_punctuation_significant[a.b,a-True]")
    assert src.startswith("@pytest.mark.parametrize") and '("a.b,a", True)' in src


def test_triage_prompt_has_exact_case_expected_actual_and_spec_but_no_implementation(tmp_path: Path):
    llm = Scripted(PAL_IMPL, PAL_ADV, verdict("AMBIGUOUS"), repair_to=PAL_IMPL)
    run(llm, PAL_SPEC, tmp_path, tier1=PAL_TIER1, max_repairs=1)
    p = llm.triage_calls[0]
    assert "test_adversarial.py::test_ADV_whitespace_punctuation_significant[a.b,a-True]" in p
    assert "s = 'a.b,a'" in p and "expected = True" in p
    assert "False is True" in p  # actual vs expected from the assertion
    assert "direct string reversal" in p  # the specification
    assert "IMPL_SECRET_MARKER" not in p and "return s == s[::-1]" not in p


def test_palindrome_scenario_invalid_case_is_quarantined_alone(tmp_path: Path):
    class P(Scripted):
        def complete(self, system, prompt):
            if prompt.startswith("Replace an invalid test"):
                return f"```python\n{PAL_REPLACEMENT}```"
            return super().complete(system, prompt)

    llm = P(PAL_IMPL, PAL_ADV, verdict("INVALID", "The spec says direct reversal; punctuation is NOT ignored."),
            repair_to=PAL_IMPL)
    r = run(llm, PAL_SPEC, tmp_path, tier1=PAL_TIER1)
    assert r.status == "VERIFIED" and r.quarantined == 1 and len(llm.triage_calls) == 1
    assert r.triage[0]["test"].endswith("[a.b,a-True]") and r.triage[0]["replacement"]
    assert r.triage[0]["case"] == {"s": "'a.b,a'", "expected": "True"}
    final = r.rounds[-1]
    assert final.adversarial_passed and "deselected" in final.output
    assert "3 passed" in final.output or "4 passed" in final.output  # sibling cases kept running


def test_deselect_is_exact_not_prefix():
    adv = ("def test_a():\n    assert False\n\ndef test_a_per_key():\n    assert True\n")
    r = run_pytest({"test_x.py": adv}, deselect=["test_x.py::test_a"])
    assert r.passed and "1 passed" in r.stdout and "1 deselected" in r.stdout  # test_a_per_key still ran


# ---- Fix 2: adversarial suite is hard-capped in code --------------------------------------------------------

class Seq(FakeLLM):
    """FakeLLM whose successive adversary replies come from a list; records adversary prompts."""
    def __init__(self, first, advs):
        super().__init__(first)
        self.advs, self.adv_prompts = list(advs), []

    def complete(self, system, prompt):
        if "adversary" in prompt:
            self.adv_prompts.append(prompt)
            return f"```python\n{self.advs.pop(0) if len(self.advs) > 1 else self.advs[0]}```"
        return super().complete(system, prompt)


FILES = {"palindrome.py": PAL_IMPL}


def test_collect_counts_parametrized_cases():
    assert len(collect_tests({**FILES, "test_adversarial.py": many(3, 7)})) == 21


def test_within_cap_is_untouched_and_not_regenerated():
    llm = Seq(PAL_IMPL, [many(1, 10)])
    code, drop, info = bounded_adversarial(llm, parse_spec(PAL_SPEC), FILES)
    assert drop == [] and len(llm.adv_prompts) == 1 and not info["regenerated"] and info["generated"] == 10


def test_over_cap_regenerates_with_count_feedback_then_accepts_smaller_suite():
    llm = Seq(PAL_IMPL, [many(4, 30), many(2, 6)])
    code, drop, info = bounded_adversarial(llm, parse_spec(PAL_SPEC), FILES)
    assert len(llm.adv_prompts) == 2 and "collected 120 tests" in llm.adv_prompts[1]
    assert drop == [] and info["regenerated"] and not info["truncated"] and info["generated"] == 12


def test_still_over_cap_is_truncated_round_robin_and_recorded(tmp_path: Path):
    events = []
    llm = Seq(PAL_IMPL, [many(4, 30)])  # model ignores the limit both times
    r = run(llm, PAL_SPEC, tmp_path, tier1=PAL_TIER1, on_event=lambda n, d: events.append((n, d)))
    c = r.adversarial_cap
    assert c["generated"] == 120 and c["kept"] == MAX_ADVERSARIAL_TESTS and c["truncated"] and c["regenerated"]
    assert r.rounds[0].tests_run == 1 + MAX_ADVERSARIAL_TESTS  # tier-1 + at most cap hidden tests actually ran
    assert "ADVERSARIAL_CAPPED" in [n for n, _ in events]
    md = (tmp_path / "verification_report.md").read_text()
    assert "Adversarial suite size" in md and "generated 120" in md and "kept 15" in md


def test_round_robin_keeps_every_function_represented():
    ids = [f"t.py::test_f{f}[c{i}]" for f in range(3) for i in range(20)]
    keep = _round_robin(ids, 15)
    assert len(keep) == 15 and {k.split("[")[0] for k in keep} == {f"t.py::test_f{f}" for f in range(3)}


def test_collection_failure_does_not_crash_or_cap():
    llm = Seq(PAL_IMPL, ["def test_broken(:\n"])
    code, drop, info = bounded_adversarial(llm, parse_spec(PAL_SPEC), FILES)
    assert drop == [] and info["generated"] is None and len(llm.adv_prompts) == 1


def test_cap_is_configurable_and_prompt_states_it(tmp_path: Path):
    llm = Seq(PAL_IMPL, [many(1, 12)])
    r = run(llm, PAL_SPEC, tmp_path, tier1=PAL_TIER1, max_adversarial=8)
    assert r.adversarial_cap["kept"] == 8 and "at most 8 collected tests" in llm.adv_prompts[0]
