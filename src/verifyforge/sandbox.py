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


# pytest's own --deselect is a *prefix* match: deselecting "t.py::test_a" also drops "t.py::test_a_per_key".
_EXACT_DESELECT = """\
import json, os


def pytest_collection_modifyitems(config, items):
    drop = set(json.loads(os.environ.get("VF_DESELECT", "[]")))
    if not drop:
        return
    gone = [i for i in items if i.nodeid in drop]
    if gone:
        config.hook.pytest_deselected(items=gone)
        items[:] = [i for i in items if i.nodeid not in drop]
"""


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

    @property
    def output(self) -> str:
        return (self.stdout + self.stderr)[-4000:]


def run_pytest(files: dict[str, str], timeout: int = 60, deselect: list[str] | None = None) -> TestResult:
    """Write files to a temp dir and run pytest there. Not a hardened sandbox."""
    with tempfile.TemporaryDirectory(prefix="verifyforge_") as d:
        for name, content in files.items():
            (Path(d) / name).write_text(content)
        (Path(d) / "_vf_deselect.py").write_text(_EXACT_DESELECT)
        env = {**os.environ, "VF_DESELECT": json.dumps(list(deselect or []))}
        start = time.monotonic()
        try:
            p = subprocess.run(
                [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-p", "_vf_deselect"],
                cwd=d, capture_output=True, text=True, timeout=timeout, env=env,
            )
        except subprocess.TimeoutExpired:
            return TestResult(False, -1, f"TIMEOUT after {timeout}s", "", time.monotonic() - start, True)
        n = sum(int(c) for c, _ in re.findall(r"(\d+) (passed|failed|errors?)\b", p.stdout))
        # Node ids may contain spaces and dots (parametrized cases); the optional " - message" tail is stripped.
        failed = tuple(dict.fromkeys(re.findall(r"^FAILED (\S+?::.+?)(?: - .*)?$", p.stdout, re.M)))
        return TestResult(p.returncode == 0, p.returncode, p.stdout, p.stderr, time.monotonic() - start, False, n, failed)


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
                               cwd=d, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return None
        if p.returncode != 0:
            return None
        return [l for l in p.stdout.splitlines() if "::" in l and not l.startswith(("=", " ", "E "))]
