from __future__ import annotations

from .llm import LLM, extract_python
from .spec import Spec

SYS = "You are a meticulous Python engineer. Reply with exactly one fenced python code block and nothing else."


def _reqs(spec: Spec) -> str:
    return "\n".join(f"{r.id}: {r.text}" for r in spec.requirements)


def generate_impl(llm: LLM, spec: Spec) -> str:
    return extract_python(llm.complete(SYS, f"Write module `{spec.module}` (a single file) satisfying:\n{_reqs(spec)}"))


def generate_tests(llm: LLM, spec: Spec, impl: str) -> str:
    return extract_python(llm.complete(
        SYS,
        f"Write pytest tests (file imports `{spec.module}`) covering every requirement. "
        f"Name each test with its requirement id, e.g. test_R1_...\n{_reqs(spec)}\n\nImplementation:\n{impl}",
    ))
