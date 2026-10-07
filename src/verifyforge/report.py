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

    @property
    def repairs(self) -> int:
        return max(len(self.rounds) - 1, 0)

    @property
    def status(self) -> str:
        r = self.rounds[-1] if self.rounds else None
        if self.error:
            return "UNVERIFIED"
        return "VERIFIED" if r and r.tier1_passed and r.adversarial_passed else "UNVERIFIED"

    def write(self, out: Path) -> None:
        data = asdict(self) | {"status": self.status, "repairs": self.repairs}
        (out / "verification_report.json").write_text(json.dumps(data, indent=2))

        def cell(v):
            return "not run" if v is None else ("pass" if v else "FAIL")

        lines = [f"# Verification report: {self.title}", "", f"**Status: {self.status}**", f"Repairs: {self.repairs}", "",
                 "## Requirements", *[f"- {r['id']}: {r['text']}" for r in self.requirements], "",
                 "## Rounds", "| Round | Tier-1 | Adversarial | Tests | Seconds |", "|---|---|---|---|---|"]
        lines += [f"| {r.n} | {cell(r.tier1_passed)} | {cell(r.adversarial_passed)} | {r.tests_run} | {r.duration:.2f} |"
                  for r in self.rounds]
        if self.error:
            lines += ["", "## Error", self.error]
        for r in self.rounds:
            if r.diff:
                lines += ["", f"## Repair diff before round {r.n}" + (" (REGRESSION)" if r.regression else ""),
                          "```diff", r.diff, "```"]
        (out / "verification_report.md").write_text("\n".join(lines) + "\n")
