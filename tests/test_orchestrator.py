from pathlib import Path

from verifyforge.orchestrator import run
from verifyforge.spec import parse_spec

SPEC = "# Add\n\nModule: adder\n\n## Requirements\n- add(a, b) returns the sum.\n"
GOOD = "def add(a, b):\n    return a + b\n"
BAD = "def add(a, b):\n    return a - b\n"
UNIT = "from adder import add\n\ndef test_R1_sum():\n    assert add(2, 3) == 5\n"
ADV = "from adder import add\n\ndef test_ADV_zero():\n    assert add(0, 0) == 0\n    assert add(-1, 1) == 0\n"


class FakeLLM:
    """Routes on prompt content; first impl is `first`, repairs return GOOD. Records every prompt."""

    def __init__(self, first):
        self.first = first
        self.repairs = 0
        self.prompts = []

    def complete(self, system, prompt):
        self.prompts.append(prompt)
        if "requirements auditor" in system:  # independent preservation audit: everything preserved
            import json
            return json.dumps({"requirements": [{"user_requirement": "the user's request", "mapped": ["R1"],
                                                 "status": "PRESERVED", "evidence": "R1 states it."}]})
        if "software architect" in system:
            import json
            return json.dumps({"title": "Add", "module": "adder", "specification": "add(a, b) returns the sum.",
                               "requirements": ["add(a, b) returns the sum."], "tier1_tests": UNIT})
        if prompt.startswith("Fix module"):
            self.repairs += 1
            return f"```python\n{GOOD}```"
        if "adversary" in prompt:
            return f"```python\n{ADV}```"
        if prompt.startswith("Write pytest"):
            return f"```python\n{UNIT}```"
        return f"```python\n{self.first}```"


def test_parse_spec():
    s = parse_spec(SPEC)
    assert s.module == "adder" and s.requirements[0].id == "R1"


def test_pass_first_round(tmp_path: Path):
    r = run(FakeLLM(GOOD), SPEC, tmp_path)
    assert r.status == "VERIFIED" and len(r.rounds) == 1 and r.repairs == 0
    assert (tmp_path / "verification_report.md").exists()


def test_repair_then_pass(tmp_path: Path):
    llm = FakeLLM(BAD)
    r = run(llm, SPEC, tmp_path)
    assert r.status == "VERIFIED" and len(r.rounds) == 2 and llm.repairs == 1


def test_give_up(tmp_path: Path):
    class Stubborn(FakeLLM):
        def complete(self, system, prompt):
            if prompt.startswith("Fix module"):
                return f"```python\n{BAD}```"
            return super().complete(system, prompt)

    r = run(Stubborn(BAD), SPEC, tmp_path, max_repairs=1)
    assert r.status == "UNVERIFIED" and len(r.rounds) == 2
