from __future__ import annotations

import ast

from .llm import LLM, SYS_TRIAGE, extract_json
from .sandbox import failure_lines
from .spec import Spec

VERDICTS = {"VALID", "INVALID", "AMBIGUOUS"}


def extract_test_source(adv_src: str, node_id: str) -> tuple[str, str]:
    """(source of the failing test, module-level helpers/imports it may rely on). Never the implementation."""
    name = node_id.split("::")[-1].split("[")[0]
    tree = ast.parse(adv_src)
    helpers, target = [], None
    for node in tree.body:
        is_test = (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test")) or (
            isinstance(node, ast.ClassDef) and node.name.startswith("Test"))
        if not is_test:
            helpers.append(ast.get_source_segment(adv_src, node) or "")
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            target = ast.get_source_segment(adv_src, node)
    if target is None:
        raise ValueError(f"test {name} not found in adversarial suite")
    return target, "\n".join(h for h in helpers if h)


def triage_test(llm: LLM, spec: Spec, adv_src: str, node_id: str, stdout: str) -> dict:
    """Fresh agent: does this test logically follow from the spec? Sees spec + test + assertion failure only."""
    failure = failure_lines(stdout, node_id)
    base = {"test": node_id.split("::", 1)[-1], "failure": failure, "spec_reference": "", "replacement_required": False}
    try:
        src, helpers = extract_test_source(adv_src, node_id)
        data = extract_json(llm.complete(
            SYS_TRIAGE,
            "A hidden test keeps failing against an implementation that was repaired. Decide ONLY whether the TEST "
            "follows logically from the SPECIFICATION. You are not shown the implementation. Work the arithmetic "
            "through explicitly. Verdicts: VALID (the spec demands what the test asserts), INVALID (the test "
            "contradicts the spec or its own arithmetic is wrong), AMBIGUOUS (the spec does not settle it). "
            "When unsure, answer AMBIGUOUS.\n"
            'Return JSON: {"verdict": "VALID|INVALID|AMBIGUOUS", "reason": "...", "spec_reference": "quote from '
            'the spec", "replacement_required": true|false}\n\n'
            f"SPECIFICATION:\n{spec.raw}\n\nTEST HELPERS:\n{helpers}\n\nFAILING TEST:\n{src}\n\n"
            f"ASSERTION FAILURE:\n{failure}",
        ))
        verdict = str(data.get("verdict", "")).upper()
        if verdict not in VERDICTS:
            raise ValueError(f"unknown verdict {data.get('verdict')!r}")
        return base | {"verdict": verdict, "reason": str(data.get("reason", "")),
                       "spec_reference": str(data.get("spec_reference", "")),
                       "replacement_required": bool(data.get("replacement_required", verdict == "INVALID"))}
    except Exception as e:  # triage must never quarantine on failure: fall back to AMBIGUOUS (keeps test blocking)
        return base | {"verdict": "AMBIGUOUS", "reason": f"Triage unavailable ({type(e).__name__}: {e}); test kept blocking."}
