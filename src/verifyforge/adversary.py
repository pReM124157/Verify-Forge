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


def replacement_test(llm: LLM, spec: Spec, bad_test: str, reason: str, k: int, helpers: str = "") -> str | None:
    """Ask for exactly ONE replacement for a test the auditor found to contradict the spec.

    The replacement is appended to the suite, so it must not touch shared module-level names: anything other than
    imports and the single test function is rejected (a redefined helper would silently break other tests)."""
    import ast

    name = f"test_ADV_replacement_{k}"
    code = extract_python(llm.complete(
        SYS_VERIFIER,
        f"Replace an invalid test for module `{spec.module}` with exactly ONE corrected pytest test named {name}. "
        "It must follow strictly from the specification; derive expected values arithmetically in comments. "
        "These module-level helpers already exist in the suite: use them as they are, and NEVER redefine or shadow "
        "them. Output only imports plus the one test function; define anything else inside the function.\n\n"
        f"Existing helpers:\n{helpers}\n\nInvalid test:\n{bad_test}\n\nWhy it is invalid: {reason}\n\n{spec.raw}",
    ))
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    others = [n for n in tree.body if not isinstance(n, (ast.Import, ast.ImportFrom, ast.FunctionDef))]
    if others or len(funcs) != 1 or funcs[0].name != name:
        return None
    return code
