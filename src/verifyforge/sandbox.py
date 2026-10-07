from __future__ import annotations

import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TestResult:
    passed: bool
    output: str


def run_pytest(files: dict[str, str], timeout: int = 60) -> TestResult:
    """Write files to a temp dir and run pytest there. Not a hardened sandbox."""
    with tempfile.TemporaryDirectory(prefix="verifyforge_") as d:
        for name, content in files.items():
            (Path(d) / name).write_text(content)
        try:
            p = subprocess.run(
                [sys.executable, "-m", "pytest", "-x", "-q", "-p", "no:cacheprovider"],
                cwd=d, capture_output=True, text=True, timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return TestResult(False, f"TIMEOUT after {timeout}s")
        return TestResult(p.returncode == 0, (p.stdout + p.stderr)[-4000:])
