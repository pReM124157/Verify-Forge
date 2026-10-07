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

    @property
    def output(self) -> str:
        return (self.stdout + self.stderr)[-4000:]


def run_pytest(files: dict[str, str], timeout: int = 60) -> TestResult:
    """Write files to a temp dir and run pytest there. Not a hardened sandbox."""
    with tempfile.TemporaryDirectory(prefix="verifyforge_") as d:
        for name, content in files.items():
            (Path(d) / name).write_text(content)
        start = time.monotonic()
        try:
            p = subprocess.run(
                [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"],
                cwd=d, capture_output=True, text=True, timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return TestResult(False, -1, f"TIMEOUT after {timeout}s", "", time.monotonic() - start, True)
        n = sum(int(c) for c, _ in re.findall(r"(\d+) (passed|failed|errors?)\b", p.stdout))
        return TestResult(p.returncode == 0, p.returncode, p.stdout, p.stderr, time.monotonic() - start, False, n)
