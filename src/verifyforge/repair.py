from __future__ import annotations

from .generate import SYS, _reqs
from .llm import LLM, extract_python
from .spec import Spec


def repair_impl(llm: LLM, spec: Spec, impl: str, failure: str) -> str:
    return extract_python(llm.complete(
        SYS,
        f"Fix module `{spec.module}` so all tests pass. Return the full corrected file. "
        f"Do not weaken requirements.\n{_reqs(spec)}\n\nCurrent code:\n{impl}\n\nFailing output:\n{failure}",
    ))
