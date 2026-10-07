from __future__ import annotations

from .llm import LLM, SYS_REPAIR, extract_python
from .spec import Spec


def repair_impl(llm: LLM, spec: Spec, impl: str, failure: str) -> str:
    return extract_python(llm.complete(
        SYS_REPAIR,
        f"Fix module `{spec.module}` so all tests pass. Return the full corrected file. "
        f"Do not weaken the specification.\n\n{spec.raw}\n\nCurrent code:\n{impl}\n\nFailing output:\n{failure}",
    ))
