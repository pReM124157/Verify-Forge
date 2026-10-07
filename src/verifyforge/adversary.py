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
