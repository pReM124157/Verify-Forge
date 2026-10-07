from __future__ import annotations

from .llm import LLM, SYS_BUILDER, extract_python
from .spec import Spec


def generate_impl(llm: LLM, spec: Spec, tier1_tests: str) -> str:
    """Builder: sees the contract and the Tier-1 tests, never the other way round."""
    return extract_python(llm.complete(
        SYS_BUILDER,
        f"Write module `{spec.module}` (a single file) satisfying this specification and passing these tests.\n\n"
        f"{spec.raw}\n\nTier-1 tests:\n{tier1_tests}",
    ))
