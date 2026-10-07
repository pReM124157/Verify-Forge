from __future__ import annotations

import re
from dataclasses import dataclass


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


SPEC_SYS = "You are a requirements engineer. Reply with only a markdown spec, no code fences."


def draft_spec(llm, request: str) -> str:
    """Turn a natural-language request into the markdown spec format parse_spec expects."""
    text = llm.complete(
        SPEC_SYS,
        "Turn this request into a spec in exactly this format:\n"
        "# <Title>\n\nModule: <snake_case_name>\n\n## Requirements\n- <one testable requirement per bullet>\n\n"
        f"Request: {request}",
    )
    text = text.strip().strip("`").removeprefix("markdown").strip()
    parse_spec(text)  # fail early on malformed output
    return text + "\n"
