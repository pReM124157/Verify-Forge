from __future__ import annotations

import re
from dataclasses import dataclass, field

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


MINIMAL_SCOPE = """MINIMAL-SCOPE RULES

Translate the user's request into the smallest complete, testable specification that satisfies exactly what was asked.

Do NOT:
- invent convenience APIs;
- add getters, reset methods, cleanup methods, statistics, retry helpers, observability methods, or configuration
  options unless explicitly required;
- expand scope merely because a feature may be useful;
- turn implementation details into public requirements.

Prefer 5-8 precise behavioral requirements. Every requirement must be directly traceable to the user's request or
strictly necessary to make that request unambiguous. If an assumption is needed, record it under "assumptions"
instead of creating a new product feature.

Generate a compact Tier-1 suite covering the contract without duplicating equivalent cases: roughly 5-10 focused
tests for a small utility.
"""

PRESERVATION_RULES = """REQUIREMENT PRESERVATION RULES (these override the minimal-scope rules above)

Minimal scope limits what you may ADD. It never allows you to remove, rename, weaken or move anything the user
explicitly asked for. Every explicit user requirement must survive into the specification exactly.

You may clarify ambiguity. You may NOT silently:
- rename an explicitly requested public function, class or method;
- remove an explicit argument or move a per-operation parameter to constructor/global scope;
- remove or change a numeric limit or capacity;
- remove a concurrency / thread-safety requirement;
- remove an eviction, ordering or expiry policy;
- weaken a MUST/NEVER into a MAY;
- invent substitute behavior.

If the user wrote `put(key, value, ttl_seconds)`, the specification must contain exactly that signature.

Also return "requirement_traceability": one entry per explicit user requirement, as
{"user_requirement": "<quoted from the request>", "mapped_requirements": ["R1", ...], "status": "preserved"}.
An independent auditor will check this; do not claim "preserved" for anything you changed.
"""

ARCHITECT_PROMPT = (
    "Turn this request into a contract. Return a JSON object with keys:\n"
    '  "title": short title,\n'
    '  "module": snake_case python module name,\n'
    '  "specification": prose that states the EXACT public API (class/function names, signatures, constructor args),\n'
    '  "requirements": list of one-line testable requirements,\n'
    '  "assumptions": list of assumptions you had to make (may be empty),\n'
    '  "requirement_traceability": list of {user_requirement, mapped_requirements, status} (see rules below),\n'
    '  "tier1_tests": a complete pytest file (imports the module by name) covering every requirement\n'
    "with deterministic, single-threaded baseline checks.\n\n"
    + MINIMAL_SCOPE + "\n" + PRESERVATION_RULES + "\nRequest: "
)


@dataclass
class ArchitectResult:
    spec_md: str
    tier1: str
    traceability: list[dict] = field(default_factory=list)  # the Architect's OWN claims; never trusted for VERIFIED


def architect_full(llm: LLM, request: str, feedback: str = "") -> ArchitectResult:
    """Natural-language request -> spec, Tier-1 tests and the Architect's traceability claims.

    `feedback` is appended after the (untouched) request when a previous spec was rejected by the preservation gate."""
    if not isinstance(request, str) or not request.strip():
        raise ValueError("request must not be empty")
    prompt = ARCHITECT_PROMPT + request
    if feedback:
        prompt += "\n\n" + feedback
    data = extract_json(llm.complete(SYS_ARCHITECT, prompt))
    try:
        title, module = str(data["title"]), str(data["module"])
        specification, tier1 = str(data["specification"]), str(data["tier1_tests"])
        reqs = [str(r) for r in data["requirements"]]
    except KeyError as e:
        raise ValueError(f"Architect JSON is missing key {e}.") from e
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", module):
        raise ValueError(f"Architect returned an invalid module name: {module!r}")
    assumptions = [str(a) for a in data.get("assumptions") or []]
    spec_md = f"# {title}\n\nModule: {module}\n\n{specification.strip()}\n\n"
    if assumptions:
        spec_md += "## Assumptions\n" + "".join(f"- {a}\n" for a in assumptions) + "\n"
    spec_md += "## Requirements\n"
    spec_md += "".join(f"- {r}\n" for r in reqs)
    parse_spec(spec_md)  # fail early on an empty requirement list
    claims = [c for c in (data.get("requirement_traceability") or []) if isinstance(c, dict)]
    return ArchitectResult(spec_md, extract_python(tier1), claims)


def architect(llm: LLM, request: str) -> tuple[str, str]:
    """Natural-language request -> (markdown spec, tier-1 tests). Written before any implementation exists."""
    if not isinstance(request, str) or not request.strip():
        raise ValueError("request must not be empty")
    r = architect_full(llm, request)
    return r.spec_md, r.tier1


def architect_tests(llm: LLM, spec: Spec) -> str:
    """Tier-1 tests from the spec alone, for hand-written spec files."""
    return extract_python(llm.complete(
        SYS_TIER1,
        f"Write pytest tests (the file imports `{spec.module}`) covering every requirement with deterministic, "
        f"single-threaded baseline checks, compact (roughly 5-10 focused tests, no duplicates). "
        f"Name each test with its requirement id, e.g. test_R1_...\n\n{spec.raw}",
    ))
