from __future__ import annotations

from .llm import LLM, SYS_VERIFIER, extract_python
from .spec import Spec


def adversarial_tests(llm: LLM, spec: Spec) -> str:
    """Hidden verifier: sees only the specification (which states the public API), never the implementation."""
    return extract_python(llm.complete(
        SYS_VERIFIER,
        f"You are an adversary. Write pytest tests for module `{spec.module}` that try to BREAK any implementation "
        "of this specification: edge cases, unusual inputs, boundary values, concurrency and timing hazards where "
        "relevant. Only assert behaviour the specification actually demands; do not invent requirements. "
        f"Name tests test_ADV_...\n\n{spec.raw}",
    ))


def replacement_test(llm: LLM, spec: Spec, bad_test: str, reason: str, k: int) -> str | None:
    """Ask for exactly ONE replacement for a test the auditor found to contradict the spec."""
    import ast

    code = extract_python(llm.complete(
        SYS_VERIFIER,
        f"Replace an invalid test for module `{spec.module}` with exactly ONE corrected pytest test named "
        f"test_ADV_replacement_{k}. It must follow strictly from the specification; derive expected values "
        f"arithmetically in comments. Include any imports/helpers it needs.\n\nInvalid test:\n{bad_test}\n\n"
        f"Why it is invalid: {reason}\n\n{spec.raw}",
    ))
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    ok = any(isinstance(n, ast.FunctionDef) and n.name == f"test_ADV_replacement_{k}" for n in tree.body)
    return code if ok else None
