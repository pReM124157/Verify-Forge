from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Round:
    n: int
    unit_passed: bool
    adversarial_passed: bool
    output: str
    diff: str = ""
    regression: bool = False


@dataclass
class Report:
    title: str
    module: str
    requirements: list[dict]
    rounds: list[Round] = field(default_factory=list)
    request: str = ""
    error: str = ""

    @property
    def status(self) -> str:
        r = self.rounds[-1] if self.rounds else None
        if self.error:
            return "UNVERIFIED"
        return "VERIFIED" if r and r.unit_passed and r.adversarial_passed else "UNVERIFIED"

    def write(self, out: Path) -> None:
        data = asdict(self) | {"status": self.status}
        (out / "verification_report.json").write_text(json.dumps(data, indent=2))
        lines = [f"# Verification report: {self.title}", "", f"**Status: {self.status}**", "",
                 "## Requirements", *[f"- {r['id']}: {r['text']}" for r in self.requirements], "",
                 "## Rounds", "| Round | Unit | Adversarial |", "|---|---|---|"]
        lines += [f"| {r.n} | {'pass' if r.unit_passed else 'FAIL'} | {'pass' if r.adversarial_passed else 'FAIL'} |"
                  for r in self.rounds]
        if self.error:
            lines += ["", f"## Error", self.error]
        for r in self.rounds:
            if r.diff:
                lines += ["", f"## Repair diff before round {r.n}" + (" (REGRESSION)" if r.regression else ""), "```diff", r.diff, "```"]
        (out / "verification_report.md").write_text("\n".join(lines) + "\n")
