"""Rich three-pane presentation. A pure subscriber to orchestrator events.

It never calls the orchestrator, models, pytest or the filesystem, and every handler is wrapped so a UI bug can
never change a verification result. Remove it (drop --ui) and the run is identical.
"""
from __future__ import annotations

import re
import threading
import time
from typing import Any

from rich import box
from rich.align import Align
from rich.console import Console, Group, RenderableType
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

from .spec import parse_spec

GREEN, RED, AMBER, DIM, CYAN = "bold green", "bold red", "bold yellow", "dim", "bold cyan"

# Seconds (x pace) to linger after an event so an audience can read it. Presentation only.
HOLD = {"SPEC_GENERATED": 1.2, "TIER1_TESTS_READY": 0.8, "BUILD_STARTED": 0.8, "BUILD_COMPLETE": 1.6,
        "TIER1_RESULT": 1.3, "ADVERSARIAL_GENERATED": 1.0, "ADVERSARIAL_RESULT": 2.2, "TRIAGE_STARTED": 1.2,
        "TRIAGE_RESULT": 3.0, "REPAIR_STARTED": 1.6, "REPAIR_COMPLETE": 2.4, "REVERIFY": 0.8, "FINAL_VERIFY": 0.8,
        "REGRESSION_DETECTED": 1.5}


def _fails(text: str) -> int:
    m = re.search(r"(\d+) failed", text)
    return int(m.group(1)) if m else 0


def _failing_tests(text: str) -> list[str]:
    return list(dict.fromkeys(re.findall(r"^FAILED \S+?::(\S+)", text, re.M)))


def _assertion_lines(text: str, limit: int = 2) -> list[str]:
    out = []
    for line in text.splitlines():
        if line.startswith("E ") and not re.match(r"E\s+\+", line):
            out.append(re.sub(r"^E\s+(AssertionError: )?", "", line).strip())
    return list(dict.fromkeys(out))[:limit]


def _meaningful_patch(diff: str) -> list[str]:
    """+/- lines only, minus docstring noise and pure re-indentation (same text on both sides)."""
    rows = [l for l in diff.splitlines() if l[:1] in "+-" and not l.startswith(("+++", "---"))]
    added = {l[1:].strip() for l in rows if l[0] == "+"}
    removed = {l[1:].strip() for l in rows if l[0] == "-"}
    keep = []
    for l in rows:
        body = l[1:].strip()
        if not body or '"""' in body or body.startswith("#"):
            continue
        if body in (added if l[0] == "-" else removed):
            continue
        keep.append(l[0] + " " + body)
    return keep


def _outline(code: str, limit: int = 12) -> list[str]:
    keep = [l.rstrip() for l in code.splitlines() if re.match(r"\s*(class |def |async def )", l)]
    return keep[:limit]


class VerifyForgeUI:
    def __init__(self, mode: str, goal: str = "", max_repairs: int = 3, pace: float = 0.0,
                 console: Console | None = None, subtitle: str = ""):
        self.console = console or Console()
        self.mode, self.subtitle, self.goal = mode, subtitle, goal
        self.max_repairs, self.pace = max_repairs, pace
        self._lock = threading.RLock()
        self._live: Live | None = None
        self.started = time.monotonic()
        self.ended: float | None = None
        self.ui_errors: list[str] = []
        # state
        self.state = "ARCHITECTING"
        self.requirements: list[str] = []
        self.tier1_ready = False
        self.code, self.version = "", 0
        self.build_state = "waiting"
        self.patch: list[str] = []
        self.repair_n = 0
        self.repair_state = ""  # "", "running", "applied"
        self.tier1: Any = None  # None | "running" | dict
        self.adv: Any = None  # None | "generating" | "running" | dict
        self.triage: list[dict] = []
        self.triaging: str = ""
        self.regression = ""  # "", "PASS", "FAIL"
        self.final = ""  # "", "VERIFIED", "UNVERIFIED", "ERROR"
        self.error = ""
        self.repairs_used = 0
        self.takeover = False

    # ----- lifecycle -------------------------------------------------------------------------------------
    def __enter__(self) -> "VerifyForgeUI":
        self.started = time.monotonic()
        self._live = Live(get_renderable=self.render, console=self.console, screen=self.console.is_terminal,
                          refresh_per_second=10, transient=False)
        self._live.__enter__()
        return self

    def __exit__(self, *exc) -> None:
        if self.final:
            self.takeover = True
            time.sleep(min(self.pace * 3, 6) if self.pace else 1.5)
        if self._live:
            self._live.__exit__(*exc)
        if self.final and self.console.is_terminal:  # alt-screen is gone now; leave the result in scrollback
            self.console.print(self.takeover_panel())

    # ----- event subscriber ------------------------------------------------------------------------------
    def __call__(self, name: str, data: dict) -> None:
        try:
            with self._lock:
                self._handle(name, data)
        except Exception as e:  # presentation must never affect verification
            self.ui_errors.append(f"{name}: {type(e).__name__}: {e}")
        if self.pace:
            time.sleep(HOLD.get(name, 0) * self.pace)

    def _handle(self, name: str, d: dict) -> None:
        if name == "ARCHITECT_DONE":
            self.goal = d.get("request") or self.goal
        elif name == "SPEC_GENERATED":
            self.requirements = [r.text for r in parse_spec(d["text"]).requirements]
            self.state = "SPEC LOCKED"
        elif name == "TIER1_TESTS_READY":
            self.tier1_ready = True
        elif name == "BUILD_STARTED":
            self.state, self.build_state = "BUILDING", "writing"
        elif name == "BUILD_COMPLETE":
            self.code, self.version, self.build_state = d["text"], d["version"], "done"
        elif name == "TIER1_RUNNING":
            self.state, self.tier1 = "TIER-1 TESTING" if not self.repair_n else "RE-VERIFYING", "running"
            if self.repair_n:
                self.adv = None
        elif name == "TIER1_RESULT":
            self.tier1 = {"passed": d["passed"], "tests": d["tests"], "seconds": d["seconds"],
                          "failed": _fails(d["text"]), "assertions": _assertion_lines(d["text"])}
        elif name == "ADVERSARIAL_GENERATING":
            self.state, self.adv = "VERIFIER WRITING HIDDEN TESTS", "generating"
        elif name == "ADVERSARIAL_RUNNING":
            self.state, self.adv = ("ADVERSARIAL TESTING" if not self.repair_n else "RE-VERIFYING"), "running"
        elif name == "ADVERSARIAL_RESULT":
            self.adv = {"passed": d["passed"], "tests": d["tests"], "seconds": d["seconds"],
                        "failed": _fails(d["text"]), "names": _failing_tests(d["text"]),
                        "assertions": _assertion_lines(d["text"])}
            self.state = "ADVERSARIAL FAILED" if not d["passed"] else self.state
        elif name == "TRIAGE_STARTED":
            self.state, self.triaging = "TRIAGING HIDDEN TEST", d["test"].split("::")[-1]
        elif name == "TRIAGE_RESULT":
            self.triage.append(d)
            self.triaging = ""
        elif name == "REPAIR_STARTED":
            self.repair_n, self.repair_state, self.patch, self.state = d["repair"], "running", [], "REPAIRING"
        elif name == "REPAIR_COMPLETE":
            self.repair_state, self.state = "applied", "PATCH APPLIED"
            self.patch = _meaningful_patch(d["diff"])
        elif name in ("REVERIFY", "FINAL_VERIFY"):
            self.state, self.tier1, self.adv = "RE-VERIFYING", "running", None
        elif name == "REGRESSION_DETECTED":
            self.regression = "FAIL"
        elif name == "VERIFIED":
            self.final, self.state, self.repairs_used = "VERIFIED", "VERIFIED", d["repairs"]
            if d["repairs"] and self.regression != "FAIL":
                self.regression = "PASS"
            self.ended = time.monotonic()
        elif name == "UNVERIFIED":
            self.final, self.state, self.repairs_used = "UNVERIFIED", "UNVERIFIED", d["repairs"]
            self.ended = time.monotonic()
        elif name == "ERROR":
            self.final, self.state, self.error = "ERROR", "ERROR", d["error"]
            self.ended = time.monotonic()

    # ----- rendering -------------------------------------------------------------------------------------
    def elapsed(self) -> str:
        s = int((self.ended or time.monotonic()) - self.started)
        return f"{s // 60:02d}:{s % 60:02d}"

    def render(self) -> RenderableType:
        with self._lock:
            if self.takeover and self.final:
                return Align.center(self.takeover_panel(), vertical="middle")
            root = Layout()
            root.split_column(Layout(self.header(), size=4), Layout(name="body"), Layout(self.footer(), size=3))
            root["body"].split_row(Layout(self.architect_panel()), Layout(self.builder_panel()),
                                   Layout(self.verifier_panel()))
            return root

    def header(self) -> RenderableType:
        offline = "OFFLINE" in self.mode
        t = Table.grid(expand=True)
        t.add_column(ratio=1)
        t.add_column(justify="right")
        t.add_row(Text("VERIFYFORGE", style="bold white"),
                  Text(f"MODE: {self.mode}", style=AMBER if offline else GREEN))
        t.add_row(Text("AI PROPOSES · EXECUTION VERIFIES", style=CYAN),
                  Text(self.subtitle, style=DIM))
        return Group(t, Text("─" * 400, style="dim", no_wrap=True, overflow="crop"))

    def footer(self) -> RenderableType:
        color = {"VERIFIED": GREEN, "UNVERIFIED": RED, "ERROR": RED}.get(self.state, AMBER)
        t = Table.grid(expand=True)
        for _ in range(3):
            t.add_column(ratio=1, justify="center")
        t.add_row(Text.assemble(("STATE: ", DIM), (self.state, color)),
                  Text.assemble(("REPAIR ", DIM), (f"{self.repair_n} / {self.max_repairs}", "bold white")),
                  Text.assemble(("ELAPSED: ", DIM), (self.elapsed(), "bold white")))
        return Panel(t, box=box.HEAVY, border_style="grey50")

    @staticmethod
    def _pane(title: str, body: RenderableType, border: str = "cyan") -> Panel:
        return Panel(body, title=f"[bold]{title}[/bold]", title_align="left", border_style=border, box=box.ROUNDED,
                     padding=(1, 2))

    @staticmethod
    def _busy(label: str) -> RenderableType:
        return Spinner("dots", text=Text(label, style=AMBER))

    def architect_panel(self) -> Panel:
        parts: list[RenderableType] = []
        if self.repair_n and self.requirements:  # collapsed once a patch lands, like the finished layout
            parts += [Text("SPEC LOCKED ✓", style=GREEN), Text(f"{len(self.requirements)} requirement{'' if len(self.requirements) == 1 else 's'}", style=DIM)]
            if self.tier1_ready:
                parts.append(Text("Tier-1 tests written before code ✓", style=DIM))
            return self._pane("ARCHITECT", Group(*parts), "green")
        parts += [Text("USER GOAL", style=CYAN), Text(_clip(self.goal or "…", 190)), Text("")]
        if not self.requirements:
            parts.append(self._busy("drafting specification + Tier-1 tests"))
        else:
            parts.append(Text("INVARIANTS", style=CYAN))
            for r in self.requirements[:8]:
                parts.append(Text.assemble(("✓ ", GREEN), _clip(r, 96)))
            if len(self.requirements) > 8:
                parts.append(Text(f"+ {len(self.requirements) - 8} more", style=DIM))
            if self.tier1_ready:
                parts += [Text(""), Text("Tier-1 tests written before any code ✓", style=GREEN)]
        return self._pane("ARCHITECT", Group(*parts), "green" if self.requirements else "cyan")

    def builder_panel(self) -> Panel:
        parts: list[RenderableType] = []
        if self.build_state == "waiting":
            parts.append(Text("waiting for specification", style=DIM))
        elif self.build_state == "writing":
            parts.append(self._busy("writing solution.py"))
        else:
            n = len(self.code.splitlines())
            parts.append(Text(f"solution.py · v{self.version} · {n} lines", style=CYAN))
            parts += [Text(l, style="white") for l in _outline(self.code)] if not self.patch else []
        if self.repair_state:
            parts.append(Text(""))
            parts.append(Text(f"REPAIR #{self.repair_n}", style=AMBER))
            if self.repair_state == "running":
                parts.append(self._busy("patching from failing assertion"))
            else:
                parts.append(Text("PATCH APPLIED ✓", style=GREEN))
                shown = self.patch[:12]
                for l in shown:
                    parts.append(Text(_clip(l, 60), style="green" if l.startswith("+") else "red"))
                if len(self.patch) > len(shown):
                    parts.append(Text(f"… {len(self.patch) - len(shown)} more changed lines", style=DIM))
        border = "green" if self.repair_state == "applied" or self.build_state == "done" else "cyan"
        return self._pane("BUILDER", Group(*parts), border)

    def verifier_panel(self) -> Panel:
        parts: list[RenderableType] = []

        def suite(label: str, v: Any, waiting: str) -> None:
            parts.append(Text(label, style=CYAN))
            if v is None:
                parts.append(Text(waiting, style=DIM))
            elif v in ("running", "generating"):
                parts.append(self._busy("writing hidden tests (spec only)" if v == "generating" else "running real pytest"))
            elif v["passed"]:
                parts.append(Text.assemble((f"{v['tests']} / {v['tests']} PASS ✓", GREEN), (f"   {v['seconds']:.2f}s", DIM)))
            else:
                parts.append(Text.assemble((f"{max(v['failed'], 1)} TEST{'S' if v['failed'] > 1 else ''} FAILED ✗", RED),
                                           (f"   of {v['tests']}   {v['seconds']:.2f}s", DIM)))
                for name in v.get("names", [])[:2]:
                    parts.append(Text(_clip(name, 52), style="red"))
                for a in v.get("assertions", [])[:2]:
                    parts.append(Text(_clip(a, 60), style="white"))
            parts.append(Text(""))

        suite("TIER-1", self.tier1, "waiting for build")
        suite("ADVERSARIAL", self.adv, "runs after Tier-1 passes")
        if self.triaging:
            parts += [Text("ADVERSARIAL TRIAGE", style=AMBER), Text(self.triaging, style="white"),
                      self._busy("auditing test against spec only"), Text("")]
        for t in self.triage[-1:]:
            quarantined = t["status"] == "QUARANTINED"
            parts += [Text("ADVERSARIAL TRIAGE", style=AMBER), Text(_clip(t["test"].split("::")[-1], 52), style="white"),
                      Text("INVALID TEST ✗" if quarantined else t["status"].upper(), style=RED if quarantined else AMBER),
                      Text(_clip(t["reason"], 230), style="white")]
            if quarantined:
                parts += [Text("QUARANTINED", style=AMBER),
                          Text(f"Replacement test generated {'✓' if t['replacement'] else '✗'}",
                               style=GREEN if t["replacement"] else RED)]
            parts.append(Text(""))
        if self.regression == "FAIL":
            parts.append(Text("REGRESSION ✗", style=RED))
        elif self.regression == "PASS":
            parts.append(Text("REGRESSION PASS ✓", style=GREEN))
        if self.final == "VERIFIED" and self.tier1 and self.adv:
            total = self.tier1["tests"] + self.adv["tests"]
            parts.append(Text(f"{total} / {total} tests passed", style=GREEN))
        border = "green" if self.final == "VERIFIED" else "red" if self.final else "cyan"
        return self._pane("VERIFIER", Group(*parts), border)

    def takeover_panel(self) -> Panel:
        ok = self.final == "VERIFIED"
        color = "green" if ok else "red"
        word = "V E R I F I E D  ✓" if ok else ("U N V E R I F I E D  ✗" if self.final == "UNVERIFIED" else "E R R O R  ✗")
        tier1_ok = bool(isinstance(self.tier1, dict) and self.tier1["passed"])
        adv_ok = bool(isinstance(self.adv, dict) and self.adv["passed"])
        rows = Table.grid(padding=(0, 6))
        rows.add_column(justify="left")
        rows.add_column(justify="left")

        def row(k: str, v: str, good: bool | None = None) -> None:
            style = GREEN if good else RED if good is False else "bold white"
            rows.add_row(Text(k, style=DIM), Text(v, style=style))

        if self.final == "ERROR":
            row("ERROR", _clip(self.error, 60), False)
        else:
            row("TIER-1", "PASS" if tier1_ok else "FAIL", tier1_ok)
            row("ADVERSARIAL", "PASS" if adv_ok else "FAIL", adv_ok)
            row("REGRESSION", self.regression or "NOT NEEDED", None if self.regression != "FAIL" else False)
            row("REPAIRS", str(self.repairs_used))
            row("TRIAGE", str(len(self.triage)))
        row("EXECUTION", "REAL PYTEST")
        row("MODEL", "REPLAYED (offline)" if "OFFLINE" in self.mode else self.mode.replace("LIVE · ", "LIVE "))
        row("ELAPSED", self.elapsed())
        tail = "Software earned verification." if ok else "Not verified. See the report for the audit trail."
        body = Group(Text(""), Align.center(Text(word, style=f"bold {color}")), Text(""), Align.center(rows), Text(""),
                     Align.center(Text(tail, style=f"italic {color}")), Text(""))
        return Panel(body, box=box.DOUBLE, border_style=color, padding=(1, 8), width=76)


def _clip(s: str, n: int) -> str:
    s = " ".join(s.split())
    return s if len(s) <= n else s[: n - 1] + "…"
