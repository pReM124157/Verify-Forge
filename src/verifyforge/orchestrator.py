from __future__ import annotations

import difflib
import json
from pathlib import Path
from typing import Callable

from .adversary import adversarial_tests, replacement_test
from .generate import generate_impl
from .llm import LLM
from .repair import repair_impl
from .report import Report, Round
from .sandbox import run_pytest
from .spec import architect, architect_tests, parse_spec
from .triage import extract_test_source, triage_test

MAX_QUARANTINES = 3

EventHook = Callable[[str, dict], None]


def _diff(old: str, new: str) -> str:
    return "".join(difflib.unified_diff(old.splitlines(True), new.splitlines(True), "before", "after"))


def run(llm: LLM, spec_text: str, out: Path, max_repairs: int = 3, timeout: int = 60,
        request: str = "", tier1: str | None = None, on_event: EventHook | None = None) -> Report:
    """Spec -> Tier-1 tests -> build -> Tier-1 -> (only if it passes) hidden adversarial suite -> repair loop.

    Neither the Tier-1 tests nor the adversarial tests are ever shown the implementation."""
    events: list[dict] = []

    def emit(name: str, **data) -> None:
        events.append({"event": name, **{k: v for k, v in data.items() if k != "text"}})
        if on_event:
            on_event(name, data)

    spec = parse_spec(spec_text)
    out.mkdir(parents=True, exist_ok=True)
    report = Report(spec.title, spec.module, [r.__dict__ for r in spec.requirements], request=request)
    versions: list[str] = []
    patches: list[str] = []
    adv = ""
    quarantined: list[str] = []  # pytest node ids excluded from the adversarial run, each with a recorded reason
    triaged: set[str] = set()
    prev_failed: set[str] = set()
    pending_diff = ""
    try:
        emit("SPEC_GENERATED", text=spec_text, module=spec.module, requirements=len(spec.requirements))
        if tier1 is None:
            tier1 = architect_tests(llm, spec)
        emit("TIER1_TESTS_READY", text=tier1)

        emit("BUILD_STARTED")
        impl = generate_impl(llm, spec, tier1)
        versions.append(impl)
        emit("BUILD_COMPLETE", text=impl, version=1)

        n = 0
        while True:
            n += 1
            files = {f"{spec.module}.py": impl}
            emit("TIER1_RUNNING", round=n)
            t = run_pytest({**files, "test_tier1.py": tier1}, timeout)
            emit("TIER1_RESULT", round=n, passed=t.passed, tests=t.tests_run, exit_code=t.exit_code,
                 seconds=t.duration, text=t.output)
            a = None
            if t.passed:
                if not adv:
                    emit("ADVERSARIAL_GENERATING")
                    adv = adversarial_tests(llm, spec)
                    emit("ADVERSARIAL_GENERATED", text=adv)
                emit("ADVERSARIAL_RUNNING", round=n)
                a = run_pytest({**files, "test_adversarial.py": adv}, timeout, deselect=quarantined)
                emit("ADVERSARIAL_RESULT", round=n, passed=a.passed, tests=a.tests_run, exit_code=a.exit_code,
                     seconds=a.duration, text=a.output)

            prev = report.rounds[-1] if report.rounds else None
            regression = bool(prev and ((prev.tier1_passed and not t.passed)
                                        or (prev.adversarial_passed is True and a is not None and not a.passed)))
            outputs = [t.output] + ([a.output] if a else [])
            report.rounds.append(Round(
                n, t.passed, None if a is None else a.passed, "\n".join(outputs).strip(),
                pending_diff, regression,
                t.tests_run + (a.tests_run if a else 0), t.duration + (a.duration if a else 0.0)))
            pending_diff = ""
            if regression:
                emit("REGRESSION_DETECTED", round=n)

            # Triage: a test that still fails after a repair gets audited by a fresh agent that never sees the code.
            if a is not None and not a.passed:
                changed = False
                for node in [f for f in a.failed_tests if f in prev_failed and f not in triaged]:
                    if len(quarantined) >= MAX_QUARANTINES:
                        break
                    triaged.add(node)
                    emit("TRIAGE_STARTED", test=node)
                    entry = triage_test(llm, spec, adv, node, a.stdout)
                    status = {"VALID": "KEPT (valid)", "AMBIGUOUS": "KEPT (ambiguous)"}.get(entry["verdict"], "QUARANTINED")
                    replacement = None
                    if status == "QUARANTINED":
                        quarantined.append(node)
                        changed = True
                        try:
                            bad, helpers = extract_test_source(adv, node)
                            replacement = replacement_test(llm, spec, bad, entry["reason"], len(quarantined), helpers)
                        except Exception:
                            replacement = None
                        if replacement:
                            adv = adv.rstrip() + "\n\n\n" + replacement
                    report.triage.append({**entry, "status": status, "replacement": replacement is not None})
                    emit("TRIAGE_RESULT", test=entry["test"], status=status, verdict=entry["verdict"],
                         reason=entry["reason"], replacement=replacement is not None)
                prev_failed = set(a.failed_tests)
                if changed:
                    continue  # rerun the (amended) suite on the same code; this does not consume a repair
            elif a is not None:
                prev_failed = set()

            if t.passed and a is not None and a.passed:
                emit("VERIFIED", repairs=report.repairs)
                break
            if report.repairs >= max_repairs:
                emit("UNVERIFIED", repairs=report.repairs)
                break

            failure = ("TIER-1:\n" + t.output) if not t.passed else ("ADVERSARIAL:\n" + a.output)
            emit("REPAIR_STARTED", repair=report.repairs + 1, of=max_repairs)
            new = repair_impl(llm, spec, impl, failure)
            patches.append(_diff(impl, new))
            pending_diff = patches[-1]
            report.repairs += 1
            impl = new
            versions.append(impl)
            emit("REPAIR_COMPLETE", repair=report.repairs, diff=patches[-1], text=impl)
            emit("FINAL_VERIFY" if report.repairs >= max_repairs else "REVERIFY", round=n + 1)
    except Exception as e:  # API/network failure etc: record honestly, never claim VERIFIED
        report.error = f"{type(e).__name__}: {e}"
        emit("ERROR", error=report.error)

    _write_artifacts(out, spec_text, spec.module, tier1 or "", adv, versions, patches, events, report)
    return report


def _write_artifacts(out: Path, spec_text: str, module: str, tier1: str, adv: str,
                     versions: list[str], patches: list[str], events: list[dict], report: Report) -> None:
    (out / "specification.md").write_text(spec_text)
    (out / "tier1_tests.py").write_text(tier1)
    (out / "adversarial_tests.py").write_text(adv)
    if report.quarantined:
        (out / "quarantine.json").write_text(json.dumps([t for t in report.triage if t["status"] == "QUARANTINED"], indent=2))
    for i, v in enumerate(versions, 1):
        (out / f"solution_v{i}.py").write_text(v)
    for i, p in enumerate(patches, 1):
        (out / f"repair_{i}.patch").write_text(p)
    if versions:
        (out / f"{module}.py").write_text(versions[-1])
    (out / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    report.write(out)


def run_request(llm: LLM, request: str, out: Path, **kw) -> Report:
    """Natural-language entry point: the Architect writes the spec AND Tier-1 tests before any code exists."""
    on_event = kw.get("on_event")
    spec_text, tier1 = architect(llm, request)
    if on_event:
        on_event("ARCHITECT_DONE", {"request": request})
    return run(llm, spec_text, out, request=request, tier1=tier1, **kw)
