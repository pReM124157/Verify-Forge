"""Running repository code safely-ish: a byte-identical STAGED COPY is what executes, never the scanned repository.

Not a sandbox: a repository's conftest.py, pytest hooks or import side effects run as ordinary Python. Mitigations only:
subprocess (no shell), timeout, credential-looking env vars scrubbed, no dependency installation, no setup.py execution,
bytecode writing disabled, and the original tree is never the working directory."""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from ..sandbox import _GUARD, _clean_env
from .safety import walk_repo

MAX_STAGE_BYTES = 150_000_000


class StageError(RuntimeError):
    pass


def stage_repo(root: Path) -> Path:
    """Copy the scannable files of the repository into a fresh temp dir. Secrets, symlinks leaving the root, huge files
    and ignored directories are NOT copied. Returns the staged root (caller removes its parent)."""
    parent = Path(tempfile.mkdtemp(prefix="vf_scan_stage_"))
    staged = parent / (root.name or "repo")
    staged.mkdir()
    total = 0
    for rel, abs_, kind in walk_repo(root):
        if kind not in ("ok", "binary"):
            continue
        total += abs_.stat().st_size
        if total > MAX_STAGE_BYTES:
            shutil.rmtree(parent, ignore_errors=True)
            raise StageError("REPOSITORY_TOO_LARGE_TO_STAGE")
        dest = staged / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(abs_, dest)
    return staged


def import_roots(staged: Path, import_root: str) -> list[str]:
    roots = []
    if import_root and import_root != "." and (staged / import_root).is_dir():
        roots.append(str(staged / import_root))
    roots.append(str(staged))
    return roots


def _env(staged: Path, import_root: str, extra: dict[str, str] | None = None) -> dict[str, str]:
    env = _clean_env()
    env["PYTHONPATH"] = os.pathsep.join(import_roots(staged, import_root))
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.update(extra or {})
    return env


_MISSING = re.compile(r"No module named ['\"]([A-Za-z0-9_\.]+)['\"]")


def _external_missing(output: str, repo_tops: set[str]) -> list[str]:
    names = sorted({m.group(1).split(".")[0] for m in _MISSING.finditer(output)})
    return [n for n in names if n not in repo_tops]


def run_existing_suite(staged: Path, import_root: str, repo_tops: set[str], timeout: int = 300) -> dict:
    """Run the repository's own pytest suite (in the staged copy) and classify the outcome honestly."""
    cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-p", "_vf_guard"]
    (staged / "_vf_guard.py").write_text(_GUARD)
    rd = Path(tempfile.mkdtemp(prefix="vf_scan_result_"))
    nonce = os.urandom(12).hex()
    res_path = rd / "result.json"
    env = _env(staged, import_root, {"VF_RESULT": str(res_path), "VF_NONCE": nonce, "VF_DESELECT": "[]"})
    out = {"command": "python -m pytest -q -p no:cacheprovider (run in a staged copy of the repository; no shell)",
           "status": "ERROR", "reason": "", "exit_code": None, "duration": 0.0, "timed_out": False, "collected": 0, "passed": 0,
           "failed": 0, "skipped": 0, "xfailed": 0, "xpassed": 0, "errors": 0, "failed_tests": [], "stdout_tail": "", "stderr_tail": ""}
    t0 = time.monotonic()
    try:
        p = subprocess.run(cmd, cwd=staged, capture_output=True, text=True, timeout=timeout, env=env)
    except subprocess.TimeoutExpired:
        out.update(status="TIMEOUT", reason=f"the existing suite did not finish within {timeout}s", timed_out=True,
                   duration=round(time.monotonic() - t0, 2))
        shutil.rmtree(rd, ignore_errors=True)
        return out
    out["duration"] = round(time.monotonic() - t0, 2)
    out["exit_code"] = p.returncode
    out["stdout_tail"], out["stderr_tail"] = p.stdout[-3000:], p.stderr[-1500:]
    text = p.stdout + p.stderr
    m = re.search(r"(\d+) errors?\b", p.stdout)
    out["errors"] = int(m.group(1)) if m else 0
    out["failed_tests"] = list(dict.fromkeys(re.findall(r"^FAILED (\S+?::.+?)(?: - .*)?$", p.stdout, re.M)))[:30]
    record = None
    try:
        record = json.loads(res_path.read_text())
        if record.get("nonce") != nonce:
            record = None
    except Exception:
        record = None
    shutil.rmtree(rd, ignore_errors=True)
    if record:
        coll = list(record.get("collected", []))
        called, bad, skipped, xf, xp = set(), set(), set(), set(), set()
        for nodeid, when, outcome, wasxfail in record.get("reports", []):
            if outcome == "skipped":
                (xf if wasxfail else skipped).add(nodeid)
            elif wasxfail:
                xp.add(nodeid)
            elif outcome != "passed":
                bad.add(nodeid)
            elif when == "call":
                called.add(nodeid)
        out.update(collected=len(coll), passed=len(called - bad), failed=len(bad), skipped=len(skipped), xfailed=len(xf), xpassed=len(xp))
    ext = _external_missing(text, repo_tops)
    if p.returncode == 5 or (record and not record.get("collected") and p.returncode == 0):
        out.update(status="NO_TESTS", reason="pytest collected no tests")
    elif ext and (p.returncode != 0) and not out["passed"]:
        out.update(status="UNAVAILABLE", reason=f"DEPENDENCY_MISSING: {', '.join(ext)} (not installed in VerifyForge's environment; nothing was installed)")
    elif "unrecognized arguments" in text or "INTERNALERROR" in text or "usage: " in p.stderr[:200]:
        out.update(status="UNAVAILABLE", reason="CONFIG_ERROR: the repository's pytest configuration could not be used here")
    elif p.returncode == 0:
        if not record:
            out.update(status="ERROR", reason="INTEGRITY: exit code 0 but no result record from pytest (the process may have ended early)")
        elif out["failed"] or out["errors"]:
            out.update(status="FAIL", reason="failures recorded")
        else:
            out.update(status="PASS", reason="")
    else:
        out.update(status="FAIL", reason=f"pytest exit code {p.returncode}: {out['failed']} failed, {out['errors']} collection/other errors")
    return out


def import_probe(staged: Path, import_root: str, dotted: str, repo_tops: set[str], timeout: int = 60) -> tuple[bool, str]:
    """Can the real module be imported in the staged repository? (This executes the module's top-level code.)"""
    with tempfile.TemporaryDirectory(prefix="vf_probe_") as cwd:
        try:
            p = subprocess.run([sys.executable, "-c", "import importlib,sys; importlib.import_module(sys.argv[1])", dotted],
                               cwd=cwd, capture_output=True, text=True, timeout=timeout, env=_env(staged, import_root))
        except subprocess.TimeoutExpired:
            return False, "IMPORT_TIMEOUT: importing the module did not finish"
    if p.returncode == 0:
        return True, ""
    ext = _external_missing(p.stderr, repo_tops)
    if ext:
        return False, f"DEPENDENCY_MISSING: {', '.join(ext)}"
    last = (p.stderr.strip().splitlines() or ["import failed"])[-1][:200]
    return False, f"IMPORT_FAILED: {last}"
