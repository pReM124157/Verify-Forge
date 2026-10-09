from __future__ import annotations

from .llm import LLM, SYS_VERIFIER, extract_python
from .spec import Spec


MAX_ADVERSARIAL_TESTS = 15  # hard ceiling on collected hidden tests (each parametrized case counts as one)


def adversarial_tests(llm: LLM, spec: Spec, cap: int = MAX_ADVERSARIAL_TESTS, previous_count: int | None = None) -> str:
    """Hidden verifier: sees only the specification (which states the public API), never the implementation."""
    retry = (f"Your previous suite collected {previous_count} tests, which is over the limit. Write a smaller, "
             "sharper suite. " if previous_count else "")
    return extract_python(llm.complete(
        SYS_VERIFIER,
        f"You are an adversary. Write pytest tests for module `{spec.module}` that try to BREAK any implementation "
        "of this specification: edge cases, unusual inputs, boundary values, concurrency and timing hazards where "
        "relevant. Only assert behaviour the specification actually demands; do not invent requirements. "
        "Derive every expected value arithmetically from the specification before asserting it. "
        f"HARD LIMIT: at most {cap} collected tests in total, and a parametrized case counts as one test; "
        f"prefer a few high-value tests over many near-duplicates. {retry}"
        f"Name tests test_ADV_...\n\n{spec.raw}",
    ))


def _round_robin(ids: list[str], cap: int) -> list[str]:
    """Pick `cap` ids spreading across test functions (first case of each function first), so breadth survives."""
    groups: dict[str, list[str]] = {}
    for i in ids:
        groups.setdefault(i.split("[")[0], []).append(i)
    keep: set[str] = set()
    for depth in range(max(len(g) for g in groups.values())):
        for g in groups.values():
            if depth < len(g) and len(keep) < cap:
                keep.add(g[depth])
    return [i for i in ids if i in keep]


def bounded_adversarial(llm: LLM, spec: Spec, base_files: dict[str, str], cap: int = MAX_ADVERSARIAL_TESTS,
                        timeout: int = 60) -> tuple[str, list[str], dict]:
    """Generate the hidden suite and enforce the size cap in code, not just in the prompt.

    Over the cap: regenerate once with the count fed back; if still over, keep a bounded round-robin subset by
    deselecting the rest (exact node ids). Returns (code, deselected ids, info for the audit report)."""
    from .sandbox import collect_tests

    code = adversarial_tests(llm, spec, cap)
    ids = collect_tests({**base_files, "test_adversarial.py": code}, timeout)
    info = {"cap": cap, "generated": None if ids is None else len(ids), "regenerated": False, "truncated": False,
            "kept": None if ids is None else min(len(ids), cap)}
    if ids is None or len(ids) <= cap:
        return code, [], info
    code2 = adversarial_tests(llm, spec, cap, previous_count=len(ids))
    ids2 = collect_tests({**base_files, "test_adversarial.py": code2}, timeout)
    info["regenerated"], info["first_attempt"] = True, len(ids)
    if ids2 is not None:
        code, ids = code2, ids2
    info["generated"] = len(ids)
    if len(ids) <= cap:
        info["kept"] = len(ids)
        return code, [], info
    keep = _round_robin(ids, cap)
    info.update(truncated=True, kept=len(keep))
    return code, [i for i in ids if i not in keep], info


def replacement_test(llm: LLM, spec: Spec, bad_test: str, reason: str, k: int, helpers: str = "",
                     system: str = SYS_VERIFIER, prefix: str = "test_ADV_replacement_") -> str | None:
    """Ask for exactly ONE replacement for a test the auditor found to contradict the spec.

    The replacement is appended to the suite, so it must not touch shared module-level names: anything other than
    imports and the single test function is rejected (a redefined helper would silently break other tests)."""
    import ast

    name = f"{prefix}{k}"
    code = extract_python(llm.complete(
        system,
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
