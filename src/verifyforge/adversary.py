from __future__ import annotations

from .generate import SYS, _reqs
from .llm import LLM, extract_python
from .spec import Spec


def adversarial_tests(llm: LLM, spec: Spec, impl: str) -> str:
    """A separate pass whose only goal is to break the implementation."""
    return extract_python(llm.complete(
        SYS,
        f"You are an adversary. Write pytest tests for module `{spec.module}` that try to BREAK this implementation: "
        "edge cases, unusual inputs (empty, unicode, huge, whitespace-only), boundary values. "
        "Only assert behaviour the requirements actually demand; do not invent requirements. "
        f"Name tests test_ADV_...\n{_reqs(spec)}\n\nImplementation:\n{impl}",
    ))
