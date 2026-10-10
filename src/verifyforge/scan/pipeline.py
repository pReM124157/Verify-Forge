"""Repository Scan Mode: read-only, evidence-first. Sits BESIDE the build pipeline and reuses its pieces (providers, hidden
verifier with cap, triage, replacement rules, pytest integrity guard). It never modifies the scanned repository, never
repairs, never calls a Builder, and never claims the repository as a whole is VERIFIED."""
from __future__ import annotations

import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Callable

from ..adversary import MAX_ADVERSARIAL_TESTS, bounded_adversarial, replacement_test
from ..llm import LLM, redact
from ..sandbox import failure_lines, run_pytest
from ..triage import extract_test_source, triage_test
from .contracts import (build_contract_prompt, build_review_prompt, contract_to_spec, gather_evidence, parse_review,
                        validate_contract)
from .discovery import api_map, discover
from .execution import StageError, import_probe, import_roots, run_existing_suite, stage_repo
from .risk import rank_deterministically
from .safety import is_ignored_dir, redact_secrets

EventHook = Callable[[str, dict], None]
HIDDEN_TIMEOUT = 120
MAX_TRIAGE_PER_ROUND, MAX_QUARANTINES, MAX_TRIAGE_ROUNDS = 5, 3, 2

VERIFIED, UNVERIFIED = "VERIFIED", "UNVERIFIED"
INSUFFICIENT, NOT_DEEP, UNSUPPORTED, NOT_COMPLETED = "INSUFFICIENT CONTRACT", "NOT DEEPLY VERIFIED", "UNSUPPORTED", "NOT COMPLETED"


def slug(path: str) -> str:
    return path.replace("/", "__").removesuffix(".py")


def fingerprint(root: Path) -> str:
    """sha256 over every regular file in the repository tree (ignored directories excluded): proves it was not modified."""
    import os

    h = hashlib.sha256()
    root = root.resolve()
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = sorted(d for d in dirnames if not is_ignored_dir(d))
        for fn in sorted(filenames):
            p = Path(dirpath) / fn
            if p.is_symlink():
                continue
            try:
                h.update(str(p.relative_to(root)).encode() + b"\0" + hashlib.sha256(p.read_bytes()).digest())
            except OSError:
                continue
    return h.hexdigest()


def _code(exc: Exception) -> str:
    return getattr(exc, "code", "") or "PROVIDER_ERROR"


def scan_repository(repo: Path, llm: LLM, out: Path, *, max_modules: int = 3, max_adversarial: int = MAX_ADVERSARIAL_TESTS,
                    timeout: int = 300, on_event: EventHook | None = None, provider: str = "", models: dict | None = None) -> dict:
    events: list[dict] = []
    hook_errors: list[str] = []

    def emit(name: str, **data) -> None:
        events.append({"event": name, **data})
        if on_event:
            try:
                on_event(name, data)
            except Exception as e:  # a subscriber can never alter or abort a scan
                hook_errors.append(f"{name}: {type(e).__name__}: {e}")

    t0 = time.monotonic()
    root = repo.resolve()
    out.mkdir(parents=True, exist_ok=True)
    before = fingerprint(root)
    report: dict = {"scan": "repository", "root": str(root), "name": root.name, "provider": provider, "models": models or {},
                    "limits": {"max_modules": max_modules, "max_adversarial": max_adversarial, "suite_timeout": timeout},
                    "status": "REPOSITORY SCAN COMPLETE", "errors": [], "modules": {}, "fingerprint_before": before}
    emit("SCAN_STARTED", root=str(root), repo=root.name)

    # ---------------- 1. deterministic discovery (no model, no execution) ----------------
    repo_map = discover(root)
    report["repository"] = {k: repo_map[k] for k in ("language", "python_files", "source_files", "test_files", "test_framework", "import_root",
                                                        "config_files", "documentation_files", "packages", "skipped")}
    (out / "repository_map.json").write_text(redact_secrets(json.dumps(repo_map, indent=2)))
    if repo_map["language"] is None:
        report["status"] = "UNSUPPORTED / NO PYTHON SOURCE FOUND"
        emit("REPO_MAPPED", modules=0, tests=0, python_files=0, unsupported=True)
        return _finish(report, out, root, events, hook_errors, llm, t0, repo_map, [], {}, emit)
    apis = api_map(repo_map)
    (out / "api_map.json").write_text(redact_secrets(json.dumps(apis, indent=2)))
    (out / "existing_tests.json").write_text(json.dumps({"tests": repo_map["tests"], "module_to_tests": repo_map["test_mapping"]}, indent=2))
    det = rank_deterministically(repo_map)
    (out / "risk_ranking.json").write_text(json.dumps({"deterministic": det, "label": "verification priority score (not a vulnerability score)"}, indent=2))
    emit("REPO_MAPPED", modules=repo_map["source_files"], tests=sum(t["count"] for t in repo_map["tests"]), python_files=repo_map["python_files"],
         test_files=repo_map["test_files"], secret_like_files=len(repo_map["secret_like_files"]))
    modules = {m["path"]: m for m in repo_map["modules"]}
    candidates = [r for r in det if r["public_api"] > 0 and r["importable"] and not modules[r["path"]]["parse_error"]]
    tops = {(m["dotted"] or "").split(".")[0] for m in modules.values()} | {Path(p).stem for p in modules}
    # ---------------- 2. existing suite, in a staged copy ----------------
    staged = None
    try:
        emit("EXISTING_TESTS_RUNNING")
        staged = stage_repo(root)
        suite = run_existing_suite(staged, repo_map["import_root"], tops, timeout)
    except StageError as e:
        suite = {"status": "UNAVAILABLE", "reason": str(e), "collected": 0, "passed": 0, "failed": 0, "skipped": 0, "xfailed": 0}
    report["existing_suite"] = suite
    (out / "existing_test_result.json").write_text(redact(redact_secrets(json.dumps(suite, indent=2))))
    emit("EXISTING_TESTS_RESULT", status=suite["status"], collected=suite["collected"], passed=suite["passed"], failed=suite["failed"],
         skipped=suite["skipped"], reason=suite.get("reason", ""))
    suite_blocks_deep = suite["status"] in ("FAIL", "TIMEOUT", "ERROR")

    # ---------------- 3. AI risk review (judge model); failure is isolated ----------------
    llm_rank: dict[str, dict] = {}
    review_failed = ""
    emit("RANKING_STARTED")
    try:
        prompt, meta = build_review_prompt(root, repo_map, det)
        report["review_context"] = meta
        for r in parse_review(llm.complete(_sys("review"), prompt), {r["path"] for r in candidates}):
            llm_rank[r["path"]] = r
    except Exception as e:
        review_failed = _code(e) if hasattr(e, "code") else f"{type(e).__name__}"
        report["errors"].append({"stage": "repo_review", "code": review_failed, "message": redact(str(e))[:300]})
        emit("RANKING_FAILED", code=review_failed)
    combined = []
    for r in candidates:
        lr = llm_rank.get(r["path"])
        combined.append({**r, "llm_priority": lr["priority"] if lr else None, "llm_reasons": lr["reasons"] if lr else [],
                         "llm_contract_confidence": lr["contract_confidence"] if lr else None,
                         "combined": round(0.5 * r["score"] + 0.5 * (lr["priority"] if lr else 0.0), 2), "selected": False})
    combined.sort(key=lambda r: (-r["combined"], r["path"]))
    emit("RANKING_DONE", ranked=[{"path": r["path"], "combined": r["combined"], "det": r["score"], "llm": r["llm_priority"]} for r in combined[:10]],
         ai=bool(llm_rank), failed=review_failed)

    results: dict[str, dict] = {m: {"path": m, "status": NOT_DEEP, "reason": "not selected for deep verification"} for m in modules}
    for r in combined:
        results[r["path"]].update(priority=r["combined"], det_priority=r["score"], llm_priority=r["llm_priority"])

    # ---------------- 4. contract inference + hidden verification, module by module ----------------
    if suite_blocks_deep:
        for r in combined[:max_modules]:
            results[r["path"]].update(reason="EXISTING_SUITE_FAILED: deep verification skipped while the repository's own tests do not pass")
    elif review_failed:
        for r in combined[:max_modules]:
            results[r["path"]].update(status=NOT_COMPLETED, reason=f"{review_failed}: the AI ranking was unavailable; no deep verification was attempted")
    elif staged is None:
        pass
    else:
        done = evaluated = 0
        evaluated_paths: set[str] = set()
        budget = min(len(combined), max_modules + 2, 6)
        provider_stop = ""
        for r in combined:
            if done >= max_modules or evaluated >= budget:
                break
            path, m = r["path"], modules[r["path"]]
            evaluated += 1
            evaluated_paths.add(path)
            res = results[path]
            try:
                emit("CONTRACT_STARTED", module=path)
                ev = gather_evidence(root, repo_map, m)
                contract = validate_contract(llm.complete(_sys("contract"), build_contract_prompt(m, ev, m["dotted"])), ev, path)
                (out / "contracts").mkdir(exist_ok=True)
                (out / "contracts" / f"{slug(path)}.json").write_text(redact(json.dumps(contract, indent=2)))
                res["contract"] = {"module_confidence": contract["module_confidence"], "requirements": contract["requirements"],
                                   "insufficient_reason": contract["insufficient_reason"]}
                emit("CONTRACT_INFERRED", module=path, confidence=contract["module_confidence"], requirements=len(contract["requirements"]),
                     eligible=len(contract["eligible_requirements"]))
                if contract["module_confidence"] not in ("HIGH", "MEDIUM"):
                    res.update(status=INSUFFICIENT, reason=contract["insufficient_reason"] or "not enough validated evidence to establish intended behaviour")
                    emit("MODULE_RESULT", module=path, status=INSUFFICIENT, reason=res["reason"])
                    continue
                r["selected"] = True
                done += 1
                emit("MODULE_STARTED", module=path, index=done, total=max_modules)
                _verify_module(llm, root, staged, repo_map, m, contract, res, out, max_adversarial, emit)
            except Exception as e:  # provider failure: keep completed evidence, stop, mark the rest honestly
                code = _code(e)
                res.update(status=NOT_COMPLETED, reason=f"{code}: {redact(str(e))[:200]}")
                report["errors"].append({"stage": "module", "module": path, "code": code, "message": redact(str(e))[:300]})
                emit("MODULE_RESULT", module=path, status=NOT_COMPLETED, reason=res["reason"])
                provider_stop = code
                break
        if provider_stop:
            remaining = max(0, max_modules - done - 1)  # slots the scan would still have filled
            for r in combined:
                if remaining and results[r["path"]]["status"] == NOT_DEEP and r["path"] not in evaluated_paths:
                    results[r["path"]].update(status=NOT_COMPLETED, reason=f"{provider_stop}: not completed because the provider failed earlier in the scan")
                    remaining -= 1
            report["status"] = "REPOSITORY SCAN INCOMPLETE — PROVIDER ERROR"
    if review_failed:
        report["status"] = "REPOSITORY SCAN INCOMPLETE — PROVIDER ERROR"
    for r in combined:
        r["selected"] = results[r["path"]]["status"] in (VERIFIED, UNVERIFIED)
    (out / "risk_ranking.json").write_text(json.dumps({"deterministic": det, "combined": combined,
                                                      "label": "verification priority score (not a vulnerability score)"}, indent=2))
    if staged is not None:
        shutil.rmtree(staged.parent, ignore_errors=True)
    return _finish(report, out, root, events, hook_errors, llm, t0, repo_map, combined, results, emit)


def _first_assertion(result) -> str:
    if not result.failed_tests:
        return ""
    lines = [l[1:].strip() for l in failure_lines(result.stdout, result.failed_tests[0]).splitlines() if not l.startswith("E        +")]
    return " | ".join(lines[:2])[:200]


def _sys(kind: str) -> str:
    from ..llm import SYS_SCAN_CONTRACT, SYS_SCAN_REVIEW

    return SYS_SCAN_REVIEW if kind == "review" else SYS_SCAN_CONTRACT


def _verify_module(llm, root, staged, repo_map, m, contract, res, out, cap, emit) -> None:
    path, dotted = m["path"], m["dotted"]
    roots = import_roots(staged, repo_map["import_root"])
    tops = {(x["dotted"] or "").split(".")[0] for x in repo_map["modules"]} | {Path(x["path"]).stem for x in repo_map["modules"]}
    ok, why = import_probe(staged, repo_map["import_root"], dotted, tops)
    if not ok:
        res.update(status=UNSUPPORTED, reason=why)
        emit("MODULE_RESULT", module=path, status=UNSUPPORTED, reason=why)
        return
    spec = contract_to_spec(contract, m, dotted)
    code, deselect, cap_info = bounded_adversarial(llm, spec, {}, cap, HIDDEN_TIMEOUT, extra_paths=roots)  # judge role, spec only
    if cap_info["generated"] is None:  # the generated file could not even be collected (syntax/import error): one fresh attempt
        emit("HIDDEN_RETRY", module=path, reason="the generated hidden suite could not be collected")
        code, deselect, cap_info = bounded_adversarial(llm, spec, {}, cap, HIDDEN_TIMEOUT, extra_paths=roots)
    emit("HIDDEN_GENERATED", module=path, generated=cap_info.get("first_attempt", cap_info["generated"]), kept=cap_info["kept"],
         truncated=cap_info["truncated"], collected=cap_info["generated"] is not None)
    gen_dir = out / "generated_tests"
    gen_dir.mkdir(exist_ok=True)
    fname = f"test_{slug(path)}_adversarial.py"
    quarantined: list[str] = list(deselect)
    triage_log: list[dict] = []
    triaged: set[str] = set()
    qcount = 0
    result = run_pytest({"test_adversarial.py": code}, HIDDEN_TIMEOUT, deselect=quarantined, extra_paths=roots)
    emit("HIDDEN_RESULT", module=path, passed=result.passed, tests=result.tests_run, failed=len(result.failed_tests),
         assertion=_first_assertion(result))
    for _ in range(MAX_TRIAGE_ROUNDS):
        todo = [n for n in result.failed_tests if n not in triaged][:MAX_TRIAGE_PER_ROUND]
        if result.passed or not todo or qcount >= MAX_QUARANTINES:
            break
        changed = False
        for node in todo:
            if qcount >= MAX_QUARANTINES:
                break
            triaged.add(node)
            entry = triage_test(llm, spec, code, node, result.stdout)  # sees the contract and the test, NEVER the repository code
            status = {"VALID": "KEPT (valid)", "AMBIGUOUS": "KEPT (ambiguous)"}.get(entry["verdict"], "QUARANTINED")
            repl = None
            if status == "QUARANTINED":
                qcount += 1
                quarantined.append(node)
                changed = True
                try:
                    bad, helpers = extract_test_source(code, node)
                    repl = replacement_test(llm, spec, bad, entry["reason"], qcount, helpers)
                except Exception:
                    repl = None
                if repl:
                    code = code.rstrip() + "\n\n\n" + repl
            triage_log.append({**entry, "status": status, "replacement": repl is not None})
            emit("TRIAGE_RESULT", module=path, test=entry["test"], verdict=entry["verdict"], status=status)
        if not changed:
            break
        result = run_pytest({"test_adversarial.py": code}, HIDDEN_TIMEOUT, deselect=quarantined, extra_paths=roots)
        emit("HIDDEN_RESULT", module=path, passed=result.passed, tests=result.tests_run, failed=len(result.failed_tests),
             assertion=_first_assertion(result))
    (gen_dir / fname).write_text(code)
    kept_valid = [t for t in triage_log if t["status"] == "KEPT (valid)"]
    kept_amb = [t for t in triage_log if t["status"] == "KEPT (ambiguous)"]
    untriaged = [n for n in result.failed_tests if n not in triaged]
    evidence = [{"test": n.split("::", 1)[-1], "assertion": failure_lines(result.stdout, n)} for n in result.failed_tests[:5]]
    if result.passed and not kept_valid and not kept_amb:
        status, reason = VERIFIED, ""
    elif kept_valid or (result.failed_tests and not kept_amb and not result.timed_out):
        status, reason = UNVERIFIED, "VALID_TEST_FAILED: an independently generated, triage-confirmed test failed against the real module"
    elif kept_amb:
        status, reason = UNVERIFIED, "AMBIGUOUS_TEST_UNRESOLVED: a failing test could not be judged valid or invalid from the contract"
    elif result.timed_out:
        status, reason = UNVERIFIED, "TIMEOUT: the hidden suite did not finish"
    elif cap_info["generated"] is None or (result.exit_code == 2 and not result.failed_tests):
        status, reason = UNVERIFIED, ("HIDDEN_SUITE_INVALID: the generated hidden suite could not be collected or run (syntax/import error in the "
                                      "generated test file, after one regeneration). This is NOT evidence about the repository either way.")
    else:
        status, reason = UNVERIFIED, f"TEST_INTEGRITY: {result.integrity_error or 'the hidden suite did not produce a valid pass'}"
    res.update(status=status, reason=reason, hidden={"generated": cap_info.get("first_attempt", cap_info["generated"]), "kept": cap_info["kept"],
                                                    "executed": result.tests_run, "cap": cap, "truncated": cap_info["truncated"],
                                                    "regenerated": cap_info["regenerated"], "passed": result.passed, "exit_code": result.exit_code,
                                                    "integrity": result.integrity_error, "duration": round(result.duration, 2),
                                                    "output_tail": "" if result.passed or result.failed_tests else redact_secrets(result.output[-900:]),
                                                    "quarantined": len([t for t in triage_log if t["status"] == "QUARANTINED"])},
               triage=triage_log, failure_evidence=evidence, untriaged_failures=untriaged[:5],
               existing_tests=[t["test_file"] for t in repo_map["test_mapping"].get(path, []) if t["confidence"] in ("high", "medium")])
    (out / "module_results").mkdir(exist_ok=True)
    (out / "module_results" / f"{slug(path)}.json").write_text(redact(json.dumps(res, indent=2)))
    if triage_log:
        (out / "triage").mkdir(exist_ok=True)
        (out / "triage" / f"{slug(path)}.json").write_text(redact(json.dumps(triage_log, indent=2)))
    emit("MODULE_RESULT", module=path, status=status, reason=reason)


# ---------------------------------------------------------------------------------------------------------------------
def _finish(report, out, root, events, hook_errors, llm, t0, repo_map, combined, results, emit) -> dict:
    after = fingerprint(root)
    report["fingerprint_after"] = after
    report["repository_unchanged"] = report["fingerprint_before"] == after
    counts = {VERIFIED: 0, UNVERIFIED: 0, INSUFFICIENT: 0, NOT_DEEP: 0, UNSUPPORTED: 0, NOT_COMPLETED: 0}
    for r in results.values():
        counts[r["status"]] += 1
    report["modules"] = results
    report["summary"] = {"python_modules": repo_map["source_files"], "existing_test_files": repo_map["test_files"],
                         "existing_tests_collected": (report.get("existing_suite") or {}).get("collected", 0),
                         "existing_suite": (report.get("existing_suite") or {}).get("status", "NOT RUN"),
                         "deeply_verified": counts[VERIFIED] + counts[UNVERIFIED], **{k.lower().replace(" ", "_"): v for k, v in counts.items()}}
    ranked = [r for r in combined if results[r["path"]]["status"] == UNVERIFIED] or [r for r in combined if results[r["path"]]["status"] in (NOT_COMPLETED, NOT_DEEP, INSUFFICIENT)]
    report["highest_risk_unresolved"] = ranked[0]["path"] if ranked else ""
    report["model_usage"] = getattr(llm, "usage_summary", lambda: {})() or {}
    report["duration"] = round(time.monotonic() - t0, 1)
    report["subscriber_errors"] = hook_errors
    from .report import render_markdown

    emit("SCAN_COMPLETE", summary=report["summary"], status=report["status"], unchanged=report["repository_unchanged"],
         highest_risk=report["highest_risk_unresolved"], duration=report["duration"])
    report["subscriber_errors"] = hook_errors
    (out / "events.jsonl").write_text(redact(redact_secrets("".join(json.dumps(e) + "\n" for e in events))))
    (out / "report.json").write_text(redact(redact_secrets(json.dumps(report, indent=2))))
    (out / "report.md").write_text(redact(redact_secrets(render_markdown(report, combined))))
    return report
