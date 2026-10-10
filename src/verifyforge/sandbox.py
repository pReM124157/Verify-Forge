from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path


# Guard plugin, loaded INSIDE the pytest process. It (1) deselects exact node ids (pytest's own --deselect is a prefix
# match) and (2) records what was really collected and how each test finished into a result file. The parent trusts that
# file, not the process exit code, which code under test can forge (os._exit(0), atexit, ...).
_GUARD = """\
import json, os
import pytest

_collected = []
_reports = []


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config, items):
    drop = set(json.loads(os.environ.get("VF_DESELECT", "[]")))
    if drop:
        gone = [i for i in items if i.nodeid in drop]
        if gone:
            config.hook.pytest_deselected(items=gone)
            items[:] = [i for i in items if i.nodeid not in drop]
    _collected[:] = [i.nodeid for i in items]


def pytest_runtest_logreport(report):
    _reports.append([report.nodeid, report.when, report.outcome, hasattr(report, "wasxfail")])


def pytest_sessionfinish(session, exitstatus):
    path = os.environ.get("VF_RESULT")
    if path:
        with open(path, "w") as f:
            json.dump({"nonce": os.environ.get("VF_NONCE"), "exitstatus": int(exitstatus),
                       "collected": _collected, "reports": _reports}, f)
"""

_SECRET_HINTS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "PASSWD", "CREDENTIAL", "ANTHROPIC", "OPENAI", "AWS_", "GITHUB",
                 "GH_", "SSH_", "NPM_", "COOKIE", "SESSION")


def _clean_env() -> dict[str, str]:
    """The code under test inherits nothing that looks like a credential. (Not a sandbox: it can still read files.)"""
    return {k: v for k, v in os.environ.items() if not any(h in k.upper() for h in _SECRET_HINTS)}


@dataclass
class TestResult:
    __test__ = False  # not a pytest class

    passed: bool
    exit_code: int
    stdout: str
    stderr: str
    duration: float
    timed_out: bool = False
    tests_run: int = 0
    failed_tests: tuple[str, ...] = ()
    skipped: int = 0  # skipped / xfailed / xpassed tests: never counted as a pass
    collected: int = 0
    integrity_error: str = ""  # why a zero exit code was NOT accepted as a pass

    @property
    def output(self) -> str:
        return (self.stdout + self.stderr)[-4000:]


def _name(nodeid: str) -> str:
    return nodeid.split("::")[-1].split("[")[0]


def _expected_test_names(files: dict[str, str]) -> set[str]:
    """Names of every test function/method defined in the test files (what must have been collected)."""
    import ast

    names: set[str] = set()
    for fname, src in files.items():
        base = fname.rsplit("/", 1)[-1]
        if not (base.startswith("test_") or base.endswith("_test.py")):
            continue
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test"):
                names.add(node.name)
            elif isinstance(node, ast.ClassDef) and (node.name.startswith("Test") or any(
                    getattr(b, "attr", getattr(b, "id", "")) == "TestCase" for b in node.bases)):
                names |= {n.name for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                          and n.name.startswith("test")}
    return names


def _judge(result_path: Path, nonce: str, exit_code: int, files: dict[str, str], deselect: list[str]):
    """-> (passed, integrity_error, executed_passed, skipped, collected). Pass requires pytest's OWN record that every
    collected test passed, that nothing was skipped/xfailed, and that every defined test was collected."""
    if exit_code != 0:
        return False, "", 0, 0, 0
    try:
        data = json.loads(result_path.read_text())
    except Exception:
        return False, "no result record from the pytest process (it exited before finishing, e.g. os._exit)", 0, 0, 0
    if data.get("nonce") != nonce:
        return False, "result record failed the nonce check", 0, 0, 0
    collected = list(data.get("collected", []))
    called_ok: set[str] = set()
    bad: set[str] = set()
    skipped_ids: set[str] = set()
    for nodeid, when, out, xfail in data.get("reports", []):
        if out == "skipped" or xfail:  # a skip, an xfail ("skipped" + wasxfail) or an unexpected pass of an xfail
            skipped_ids.add(nodeid)
        elif out != "passed":
            bad.add(nodeid)
        elif when == "call":
            called_ok.add(nodeid)
    skipped = len(skipped_ids)
    ran = sum(1 for n in collected if n in called_ok and n not in bad and n not in skipped_ids)
    if not collected:
        return False, "no tests were collected", 0, skipped, 0
    if skipped:
        return False, f"{skipped} test(s) were skipped or xfailed; skipped tests are not evidence", ran, skipped, len(collected)
    if ran != len(collected):
        return False, f"only {ran} of {len(collected)} collected tests ran to a pass", ran, skipped, len(collected)
    fully_deselected = {_name(d) for d in deselect}
    missing = sorted(n for n in _expected_test_names(files) - {_name(c) for c in collected} if n not in fully_deselected)
    if missing:
        return False, f"defined tests were not collected: {', '.join(missing[:5])}", ran, skipped, len(collected)
    return True, "", ran, skipped, len(collected)


def run_pytest(files: dict[str, str], timeout: int = 60, deselect: list[str] | None = None) -> TestResult:
    """Write files to a temp dir and run pytest there. Not a hardened sandbox: it only raises the bar against a
    result being faked by accident or by simple tampering (exit codes, atexit, conftest, collection patching)."""
    deselect = list(deselect or [])
    with tempfile.TemporaryDirectory(prefix="verifyforge_") as d, tempfile.TemporaryDirectory(prefix="vf_result_") as rd:
        for name, content in files.items():
            (Path(d) / name).write_text(content)
        (Path(d) / "_vf_guard.py").write_text(_GUARD)
        nonce = os.urandom(12).hex()
        result_path = Path(rd) / "result.json"
        env = {**_clean_env(), "VF_DESELECT": json.dumps(deselect), "VF_NONCE": nonce, "VF_RESULT": str(result_path)}
        start = time.monotonic()
        try:
            p = subprocess.run(
                [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-p", "_vf_guard"],
                cwd=d, capture_output=True, text=True, timeout=timeout, env=env,
            )
        except subprocess.TimeoutExpired:
            return TestResult(False, -1, f"TIMEOUT after {timeout}s", "", time.monotonic() - start, True,
                              integrity_error="timed out")
        n = sum(int(c) for c, _ in re.findall(r"(\d+) (passed|failed|errors?)\b", p.stdout))
        # Node ids may contain spaces and dots (parametrized cases); the optional " - message" tail is stripped.
        failed = tuple(dict.fromkeys(re.findall(r"^FAILED (\S+?::.+?)(?: - .*)?$", p.stdout, re.M)))
        ok, why, ran, skipped, collected = _judge(result_path, nonce, p.returncode, files, deselect)
        return TestResult(ok, p.returncode, p.stdout, p.stderr, time.monotonic() - start, False, ran if ok else n, failed,
                          skipped, collected, why)


def failure_lines(stdout: str, node_id: str) -> str:
    """Only the assertion ('E ') lines of one test's failure block. Deliberately omits traceback source
    so implementation code never leaks to the triage model."""
    name = node_id.split("::")[-1]
    out: list[str] = []
    inside = False
    for line in stdout.splitlines():
        m = re.match(r"^_{3,} (.+?) _{3,}$", line)
        if m:
            # exact match on the whole header: case ids such as "a.b,a-True" contain dots
            inside = m.group(1) == name or m.group(1).endswith("." + name)
        elif inside and line.startswith("E "):
            out.append(line)
    return "\n".join(out[:12])


def collect_tests(files: dict[str, str], timeout: int = 60) -> list[str] | None:
    """Node ids pytest would run (parametrized cases counted individually), or None if collection fails."""
    with tempfile.TemporaryDirectory(prefix="verifyforge_") as d:
        for name, content in files.items():
            (Path(d) / name).write_text(content)
        try:
            p = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider"],
                               cwd=d, capture_output=True, text=True, timeout=timeout, env=_clean_env())
        except subprocess.TimeoutExpired:
            return None
        if p.returncode != 0:
            return None
        return [l for l in p.stdout.splitlines() if "::" in l and not l.startswith(("=", " ", "E "))]
