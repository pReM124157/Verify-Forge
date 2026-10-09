from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Round:
    n: int
    tier1_passed: bool
    adversarial_passed: bool | None  # None: not run (Tier-1 failed first)
    output: str
    diff: str = ""
    regression: bool = False
    tests_run: int = 0
    duration: float = 0.0


@dataclass
class Report:
    title: str
    module: str
    requirements: list[dict]
    rounds: list[Round] = field(default_factory=list)
    request: str = ""
    error: str = ""
    repairs: int = 0
    triage: list[dict] = field(default_factory=list)
    adversarial_cap: dict = field(default_factory=dict)
    traceability: list[dict] = field(default_factory=list)  # auditor-verified user requirement -> spec mapping
    preservation: dict = field(default_factory=dict)  # attempts / rejections of the requirement-preservation gate
    unverified_reason: str = ""

    @property
    def quarantined(self) -> int:
        return sum(t["status"] == "QUARANTINED" for t in self.triage)

    @property
    def status(self) -> str:
        r = self.rounds[-1] if self.rounds else None
        if self.error or self.unverified_reason:
            return "UNVERIFIED"
        # VERIFIED is forbidden unless every explicit user requirement was PRESERVED in the specification.
        if any(t["status"] != "PRESERVED" for t in self.traceability):
            return "UNVERIFIED"
        return "VERIFIED" if r and r.tier1_passed and r.adversarial_passed else "UNVERIFIED"

    def write(self, out: Path) -> None:
        data = asdict(self) | {"status": self.status, "quarantined": self.quarantined}
        (out / "verification_report.json").write_text(json.dumps(data, indent=2))

        def cell(v):
            return "not run" if v is None else ("pass" if v else "FAIL")

        lines = [f"# Verification report: {self.title}", "", f"**Status: {self.status}**", f"Repairs: {self.repairs}", f"Quarantined adversarial tests: {self.quarantined}", "",
                 *(["## User request", "```text", self.request, "```", ""] if self.request else []),
                 "## Requirements", *[f"- {r['id']}: {r['text']}" for r in self.requirements], "",
                 "## Rounds", "| Round | Tier-1 | Adversarial | Tests | Seconds |", "|---|---|---|---|---|"]
        lines += [f"| {r.n} | {cell(r.tier1_passed)} | {cell(r.adversarial_passed)} | {r.tests_run} | {r.duration:.2f} |"
                  for r in self.rounds]
        if self.unverified_reason:
            lines += ["", f"**Reason: {self.unverified_reason}**"]
        if self.traceability:
            ok = sum(t["status"] == "PRESERVED" for t in self.traceability)
            lines += ["", "## User requirement traceability", f"{ok} / {len(self.traceability)} user requirements preserved", "",
                      "| User requirement | Spec mapping | Status |", "|---|---|---|"]
            lines += [f"| {t['user_requirement'].replace('|', '/')} | {', '.join(t['mapped']) or '-'} | {t['status']} |"
                      for t in self.traceability]
            bad = [t for t in self.traceability if t["status"] != "PRESERVED"]
            if bad:
                lines += ["", "Not preserved:"] + [f"- {t['user_requirement']}: {t['evidence']}" for t in bad]
        for rej in self.preservation.get("rejected", []):
            lines += ["", f"## Specification rejected (attempt {rej['attempt']})",
                      f"{rej['preserved']} / {rej['total']} user requirements preserved"]
            lines += [f"- {m}" for m in rej["missing"]]
        c = self.adversarial_cap
        if c:
            how = ("regenerated, then truncated" if c.get("truncated") else "regenerated" if c.get("regenerated") else "within cap")
            lines += ["", "## Adversarial suite size",
                      f"Cap {c['cap']} | generated {c['generated']} | kept {c['kept']} | {how}"
                      + (f" | first attempt {c['first_attempt']}" if c.get("first_attempt") else "")]
        if self.triage:
            lines += ["", "## Test triage (Tier-1 and adversarial)"]
            for t in self.triage:
                lines += ["", f"### {t['test']}", f"Suite: {t.get('suite', 'adversarial')}",
                          f"Status: **{t['status']}**  (verdict: {t['verdict']})",
                          f"Reason: {t['reason']}", f"Spec reference: {t['spec_reference'] or 'n/a'}",
                          "Observed failure:", "```", t["failure"], "```"]
                if t["status"] == "QUARANTINED":
                    lines.append(f"Replacement test generated: {'YES' if t['replacement'] else 'NO'}")
        if self.error:
            lines += ["", "## Error", self.error]
        for r in self.rounds:
            if r.diff:
                lines += ["", f"## Repair diff before round {r.n}" + (" (REGRESSION)" if r.regression else ""),
                          "```diff", r.diff, "```"]
        (out / "verification_report.md").write_text("\n".join(lines) + "\n")
