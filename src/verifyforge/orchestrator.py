from __future__ import annotations

import difflib
import json
from pathlib import Path
from typing import Callable

from .adversary import MAX_ADVERSARIAL_TESTS, bounded_adversarial, replacement_test
from .generate import generate_impl
from .llm import LLM, SYS_TIER1
from .repair import repair_impl
from .report import Report, Round
from .sandbox import run_pytest
from .preservation import REASON_AUDIT_UNAVAILABLE, REASON_NOT_PRESERVED, check_preservation
from .spec import architect_full, architect_tests, parse_spec
from .triage import extract_test_source, triage_test

MAX_QUARANTINES = 3

EventHook = Callable[[str, dict], None]


def _diff(old: str, new: str) -> str:
    return "".join(difflib.unified_diff(old.splitlines(True), new.splitlines(True), "before", "after"))


def run(llm: LLM, spec_text: str, out: Path, max_repairs: int = 3, timeout: int = 60,
        request: str = "", tier1: str | None = None, on_event: EventHook | None = None,
        max_adversarial: int = MAX_ADVERSARIAL_TESTS, preflight: dict | None = None) -> Report:
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
    if preflight:  # results of the requirement-preservation gate, which ran before this point
        report.traceability, report.preservation = preflight["traceability"], preflight["preservation"]
        events[:0] = preflight["events"]
    versions: list[str] = []
    patches: list[str] = []
    adv = ""
    capped_out: list[str] = []  # node ids dropped by the size cap (exact deselect), recorded in the report
    quarantined: list[str] = []  # pytest node ids excluded from the adversarial run, each with a recorded reason
    triaged: set[str] = set()
    prev_failed: set[str] = set()
    t1_quarantined: list[str] = []  # same, for the Tier-1 suite
    t1_triaged: set[str] = set()
    prev_t1_failed: set[str] = set()
    pending_diff = ""

    def audit(suite: str, src: str, result, prev_set: set[str], seen: set[str], quarantine: list[str]) -> tuple[str, bool]:
        """A test that still fails after a repair is audited by a fresh agent that never sees the code.
        INVALID -> quarantine + one replacement; VALID / AMBIGUOUS / unusable -> the test keeps blocking."""
        changed = False
        label = "Tier-1" if suite == "tier1" else "adversarial"
        for node in [f for f in result.failed_tests if f in prev_set and f not in seen]:
            if len(quarantined) + len(t1_quarantined) >= MAX_QUARANTINES:
                break
            seen.add(node)
            emit("TRIAGE_STARTED", test=node, suite=suite)
            entry = triage_test(llm, spec, src, node, result.stdout)
            status = {"VALID": "KEPT (valid)", "AMBIGUOUS": "KEPT (ambiguous)"}.get(entry["verdict"], "QUARANTINED")
            replacement = None
            if status == "QUARANTINED":
                quarantine.append(node)
                changed = True
                try:
                    bad, helpers = extract_test_source(src, node)
                    kw = ({"system": SYS_TIER1, "prefix": "test_R_replacement_"} if suite == "tier1" else {})
                    replacement = replacement_test(llm, spec, bad, entry["reason"], len(quarantine), helpers, **kw)
                except Exception:
                    replacement = None
                if replacement:
                    src = src.rstrip() + "\n\n\n" + replacement
            report.triage.append({**entry, "status": status, "replacement": replacement is not None, "suite": label})
            emit("TRIAGE_RESULT", test=entry["test"], status=status, verdict=entry["verdict"], suite=suite,
                 reason=entry["reason"], replacement=replacement is not None)
        return src, changed
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
            t = run_pytest({**files, "test_tier1.py": tier1}, timeout, deselect=t1_quarantined)
            emit("TIER1_RESULT", round=n, passed=t.passed, tests=t.tests_run, exit_code=t.exit_code,
                 seconds=t.duration, text=t.output)
            a = None
            if t.passed:
                if not adv:
                    emit("ADVERSARIAL_GENERATING")
                    adv, capped_out, cap_info = bounded_adversarial(llm, spec, files, max_adversarial, timeout)
                    report.adversarial_cap = cap_info
                    if cap_info["regenerated"] or cap_info["truncated"]:
                        emit("ADVERSARIAL_CAPPED", **cap_info)
                    emit("ADVERSARIAL_GENERATED", text=adv)
                emit("ADVERSARIAL_RUNNING", round=n)
                a = run_pytest({**files, "test_adversarial.py": adv}, timeout, deselect=quarantined + capped_out)
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

            # Triage (both suites): a test that still fails after a repair is audited before more code is bent to it.
            if not t.passed:
                tier1, changed = audit("tier1", tier1, t, prev_t1_failed, t1_triaged, t1_quarantined)
                prev_t1_failed = set(t.failed_tests)
                if changed:
                    continue  # rerun the amended Tier-1 on the same code; this does not consume a repair
            else:
                prev_t1_failed = set()
            if a is not None and not a.passed:
                adv, changed = audit("adversarial", adv, a, prev_failed, triaged, quarantined)
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


def _feedback(missing: list[str]) -> str:
    return ("PREVIOUS SPECIFICATION REJECTED: it did not preserve the user's explicit requirements. Missing or changed:\n"
            + "\n".join(f"- {m}" for m in missing)
            + "\nRegenerate the full JSON. Keep every explicit user requirement exactly as the user wrote it.")


def run_request(llm: LLM, request: str, out: Path, **kw) -> Report:
    """Natural-language entry point: the Architect writes the spec AND Tier-1 tests before any code exists.

    Requirement-preservation gate: before the Builder runs, an independent audit must confirm the spec keeps every
    explicit user requirement. One regeneration is allowed; otherwise the run ends UNVERIFIED with no code written."""
    # Invariant: an empty request must never reach the Architect (no model call, event, run dir or spec).
    if not isinstance(request, str) or not request.strip():
        raise ValueError("request must not be empty")
    on_event = kw.get("on_event")
    events: list[dict] = []

    def emit(name: str, **data) -> None:
        events.append({"event": name, **{k: v for k, v in data.items() if k != "text"}})
        if on_event:
            on_event(name, data)

    rejected: list[dict] = []
    specs: list[str] = []
    feedback = ""
    for attempt in (1, 2):
        arch = architect_full(llm, request, feedback)
        specs.append(arch.spec_md)
        emit("PRESERVATION_CHECK_STARTED", attempt=attempt)
        pres = check_preservation(llm, request, arch.spec_md, arch.traceability)
        if pres.passed:
            emit("PRESERVATION_PASSED", attempt=attempt, preserved=pres.preserved, total=pres.total)
            break
        rej = {"attempt": attempt, "preserved": pres.preserved, "total": pres.total, "missing": pres.missing,
               "audit_error": pres.audit_error}
        rejected.append(rej)
        emit("SPEC_REJECTED", **rej)
        if not pres.audit_ok:
            break  # regenerating cannot help when the audit itself is unavailable
        if attempt == 1:
            feedback = _feedback(pres.missing)
            emit("SPEC_REGENERATING", attempt=2)
    else:
        pres = pres  # second attempt also rejected

    info = {"attempts": len(specs), "rejected": rejected, "passed": pres.passed}
    if pres.passed:
        emit("ARCHITECT_DONE", request=request)
        for i, rej in enumerate(rejected, 1):
            out.mkdir(parents=True, exist_ok=True)
            (out / f"specification_rejected_{i}.md").write_text(specs[i - 1])
        return run(llm, arch.spec_md, out, request=request, tier1=arch.tier1,
                   preflight={"events": events, "traceability": pres.items, "preservation": info}, **kw)

    # Rejected: no Builder, no tests executed, no code. Record the evidence and end UNVERIFIED.
    spec = parse_spec(specs[-1])
    reason = REASON_NOT_PRESERVED if pres.audit_ok else REASON_AUDIT_UNAVAILABLE
    report = Report(spec.title, spec.module, [r.__dict__ for r in spec.requirements], request=request,
                    traceability=pres.items, preservation=info, unverified_reason=reason,
                    error=pres.audit_error if not pres.audit_ok else "")
    emit("UNVERIFIED", repairs=0, reason=reason, preserved=pres.preserved, total=pres.total)
    out.mkdir(parents=True, exist_ok=True)
    for i, text in enumerate(specs, 1):
        (out / f"specification_rejected_{i}.md").write_text(text)
    (out / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    report.write(out)
    return report
