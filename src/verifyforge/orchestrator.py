from __future__ import annotations

import difflib
from pathlib import Path

from .adversary import adversarial_tests
from .generate import generate_impl, generate_tests
from .llm import LLM
from .repair import repair_impl
from .report import Report, Round
from .sandbox import run_pytest
from .spec import draft_spec, parse_spec


def _diff(old: str, new: str) -> str:
    return "".join(difflib.unified_diff(old.splitlines(True), new.splitlines(True), "before", "after"))


def run(llm: LLM, spec_text: str, out: Path, max_rounds: int = 4, timeout: int = 60,
        request: str = "") -> Report:
    """Run the loop. `spec_text` is a markdown spec; `request` (if given) is the NL prompt it was drafted from."""
    spec = parse_spec(spec_text)
    out.mkdir(parents=True, exist_ok=True)
    report = Report(spec.title, spec.module, [r.__dict__ for r in spec.requirements], request=request)
    impl = unit = adv = ""
    diff = ""
    try:
        impl = generate_impl(llm, spec)
        unit = generate_tests(llm, spec, impl)
        adv = adversarial_tests(llm, spec, impl)
        for n in range(1, max_rounds + 1):
            base = {f"{spec.module}.py": impl}
            # Both suites are rerun in full every round, so a repair can't silently regress earlier passes.
            u = run_pytest({**base, "test_unit.py": unit}, timeout)
            a = run_pytest({**base, "test_adversarial.py": adv}, timeout)
            prev = report.rounds[-1] if report.rounds else None
            regression = bool(prev and ((prev.unit_passed and not u.passed) or (prev.adversarial_passed and not a.passed)))
            report.rounds.append(Round(n, u.passed, a.passed, (u.output + "\n" + a.output).strip(), diff, regression))
            if u.passed and a.passed:
                break
            if n < max_rounds:
                failure = ("UNIT:\n" + u.output if not u.passed else "") + ("\nADVERSARIAL:\n" + a.output if not a.passed else "")
                new = repair_impl(llm, spec, impl, failure)
                diff = _diff(impl, new)
                impl = new
    except Exception as e:  # API/network failure etc: record honestly, never claim VERIFIED
        report.error = f"{type(e).__name__}: {e}"

    (out / f"{spec.module}.py").write_text(impl)
    (out / "test_unit.py").write_text(unit)
    (out / "test_adversarial.py").write_text(adv)
    report.write(out)
    return report


def run_request(llm: LLM, request: str, out: Path, **kw) -> Report:
    """Natural-language entry point: VerifyForge drafts the spec itself, saves it, then runs the loop."""
    spec_text = draft_spec(llm, request)
    out.mkdir(parents=True, exist_ok=True)
    (out / "spec.md").write_text(spec_text)
    return run(llm, spec_text, out, request=request, **kw)
