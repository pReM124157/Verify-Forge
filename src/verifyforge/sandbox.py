from __future__ import annotations

import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path


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
        start = time.monotonic()
        try:
            p = subprocess.run(
                [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                 *[a for d in (deselect or []) for a in ("--deselect", d)]],
                cwd=d, capture_output=True, text=True, timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return TestResult(False, -1, f"TIMEOUT after {timeout}s", "", time.monotonic() - start, True)
        n = sum(int(c) for c, _ in re.findall(r"(\d+) (passed|failed|errors?)\b", p.stdout))
        failed = tuple(dict.fromkeys(re.findall(r"^FAILED (\S+)", p.stdout, re.M)))
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
            inside = m.group(1).split(".")[-1] == name
        elif inside and line.startswith("E "):
            out.append(line)
    return "\n".join(out[:12])
