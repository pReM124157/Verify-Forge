from __future__ import annotations

import json
from pathlib import Path

from ..llm import SYS_ARCHITECT, SYS_BUILDER, SYS_REPAIR, SYS_TIER1, SYS_VERIFIER, ReplayLLM

_DIR = Path(__file__).parent / "rate_limiter"

RATE_LIMITER_REQUEST = (
    "Build a thread-safe in-memory sliding-window rate limiter in Python: 60 requests per rolling 60 seconds "
    "per client key plus a burst of 10 (never more than 70 in one window), isolated per key, expired entries "
    "removed, rejected requests consume nothing, safe under concurrent calls, with injectable time for tests."
)


def _code(name: str) -> str:
    return f"```python\n{(_DIR / name).read_text()}```"


def rate_limiter_replay() -> ReplayLLM:
    """Saved model outputs for the rate-limiter demo. The first build is the classic check-then-append race."""
    arch = json.loads((_DIR / "architect.json").read_text())
    arch["tier1_tests"] = (_DIR / "tier1_tests.py").read_text()
    return ReplayLLM({
        SYS_ARCHITECT: [json.dumps(arch)],
        SYS_BUILDER: [_code("solution_v1.py")],
        SYS_VERIFIER: [_code("adversarial_tests.py")],
        SYS_REPAIR: [_code("solution_v2.py")],
        SYS_TIER1: [],
    })
