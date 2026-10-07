from __future__ import annotations

import re
from dataclasses import dataclass

from .llm import LLM, SYS_ARCHITECT, SYS_TIER1, extract_json, extract_python


@dataclass
class Requirement:
    id: str
    text: str


@dataclass
class Spec:
    title: str
    module: str
    requirements: list[Requirement]
    raw: str


def parse_spec(raw: str) -> Spec:
    title = "Untitled"
    module = ""
    reqs: list[Requirement] = []
    in_reqs = False
    for line in raw.splitlines():
        s = line.strip()
        if s.startswith("# "):
            title = s[2:].strip()
        elif m := re.match(r"module\s*:\s*([A-Za-z_][A-Za-z0-9_]*)", s, re.I):
            module = m.group(1)
        elif s.startswith("## "):
            in_reqs = s[3:].strip().lower().startswith("requirement")
        elif in_reqs and s.startswith(("- ", "* ")):
            reqs.append(Requirement(f"R{len(reqs) + 1}", s[2:].strip()))
    if not module:
        module = re.sub(r"\W+", "_", title.lower()).strip("_") or "solution"
    if not reqs:
        raise ValueError("Spec has no requirements under '## Requirements'.")
    return Spec(title, module, reqs, raw)


ARCHITECT_PROMPT = (
    "Turn this request into a contract. Return a JSON object with keys:\n"
    '  "title": short title,\n'
    '  "module": snake_case python module name,\n'
    '  "specification": prose that states the EXACT public API (class/function names, signatures, constructor args),\n'
    '  "requirements": list of one-line testable requirements,\n'
    '  "tier1_tests": a complete pytest file (imports the module by name) covering every requirement\n'
    "with deterministic, single-threaded baseline checks.\n\nRequest: "
)


def architect(llm: LLM, request: str) -> tuple[str, str]:
    """Natural-language request -> (markdown spec, tier-1 tests). Written before any implementation exists."""
    data = extract_json(llm.complete(SYS_ARCHITECT, ARCHITECT_PROMPT + request))
    try:
        title, module = str(data["title"]), str(data["module"])
        specification, tier1 = str(data["specification"]), str(data["tier1_tests"])
        reqs = [str(r) for r in data["requirements"]]
    except KeyError as e:
        raise ValueError(f"Architect JSON is missing key {e}.") from e
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", module):
        raise ValueError(f"Architect returned an invalid module name: {module!r}")
    spec_md = f"# {title}\n\nModule: {module}\n\n{specification.strip()}\n\n## Requirements\n"
    spec_md += "".join(f"- {r}\n" for r in reqs)
    parse_spec(spec_md)  # fail early on an empty requirement list
    return spec_md, extract_python(tier1)


def architect_tests(llm: LLM, spec: Spec) -> str:
    """Tier-1 tests from the spec alone, for hand-written spec files."""
    return extract_python(llm.complete(
        SYS_TIER1,
        f"Write pytest tests (the file imports `{spec.module}`) covering every requirement with deterministic, "
        f"single-threaded baseline checks. Name each test with its requirement id, e.g. test_R1_...\n\n{spec.raw}",
    ))
