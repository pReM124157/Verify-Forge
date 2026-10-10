"""Rich UI for Repository Scan Mode: a pure event subscriber (it never calls the scanner, models or pytest)."""
from __future__ import annotations

import threading
import time

from rich import box
from rich.align import Align
from rich.console import Console, Group, RenderableType
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

from ..ui import AMBER, CYAN, DIM, GREEN, RED, _clip

STATUS_STYLE = {"VERIFIED": GREEN, "UNVERIFIED": RED, "INSUFFICIENT CONTRACT": AMBER, "NOT DEEPLY VERIFIED": DIM,
                "UNSUPPORTED": AMBER, "NOT COMPLETED": RED}


class ScanUI:
    def __init__(self, mode: str, judge: str = "", console: Console | None = None, subtitle: str = ""):
        self.console, self.mode, self.judge, self.subtitle = console or Console(), mode, judge, subtitle
        self._lock = threading.RLock()
        self._live: Live | None = None
        self.started, self.ended = time.monotonic(), None
        self.ui_errors: list[str] = []
        self.state, self.name = "SCANNING REPOSITORY", ""
        self.modules = self.tests = 0
        self.suite: dict = {}
        self.ranked: list[dict] = []
        self.contracts: dict[str, str] = {}
        self.results: dict[str, str] = {}
        self.current, self.index, self.total = "", 0, 0
        self.hidden: dict = {}
        self.triage: list[str] = []
        self.assertion = ""
        self.final: dict | None = None
        self.takeover = False
        self.ai_failed = ""

    def __enter__(self):
        self.started = time.monotonic()
        self._live = Live(get_renderable=self.render, console=self.console, screen=self.console.is_terminal, refresh_per_second=8)
        self._live.__enter__()
        return self

    def __exit__(self, *exc):
        if self.final and self.console.is_terminal:
            self.takeover = True
            time.sleep(1.5)
        if self._live:
            self._live.__exit__(*exc)
        if self.final and self.console.is_terminal:
            self.console.print(self.takeover_panel())

    def __call__(self, name: str, d: dict) -> None:
        try:
            with self._lock:
                self._handle(name, d)
        except Exception as e:
            self.ui_errors.append(f"{name}: {type(e).__name__}: {e}")

    def _handle(self, n: str, d: dict) -> None:
        if n == "SCAN_STARTED":
            self.name, self.state = d["repo"], "SCANNING REPOSITORY"
        elif n == "REPO_MAPPED":
            self.modules, self.tests, self.state = d["modules"], d["tests"], "MAPPING PYTHON MODULES"
        elif n == "EXISTING_TESTS_RUNNING":
            self.state = "RUNNING EXISTING TESTS"
        elif n == "EXISTING_TESTS_RESULT":
            self.suite = d
        elif n == "RANKING_STARTED":
            self.state = "RANKING VERIFICATION PRIORITY"
        elif n == "RANKING_DONE":
            self.ranked, self.ai_failed = d["ranked"], d.get("failed", "")
        elif n == "RANKING_FAILED":
            self.ai_failed = d["code"]
        elif n == "CONTRACT_STARTED":
            self.state, self.current = "INFERRING CONTRACTS", d["module"]
        elif n == "CONTRACT_INFERRED":
            self.contracts[d["module"]] = d["confidence"]
        elif n == "MODULE_STARTED":
            self.current, self.index, self.total = d["module"], d["index"], d["total"]
            self.state, self.hidden, self.triage, self.assertion = f"VERIFYING MODULE {d['index']} / {d['total']}", {}, [], ""
        elif n == "HIDDEN_RETRY":
            self.hidden = {"retry": True}
        elif n == "HIDDEN_GENERATED":
            self.hidden.update(generated=d["generated"], kept=d["kept"], collected=d.get("collected", True), retry=False)
        elif n == "HIDDEN_RESULT":
            self.hidden.update(passed=d["passed"], tests=d["tests"], failed=d["failed"])
            self.assertion = d.get("assertion", "")
        elif n == "TRIAGE_RESULT":
            self.triage.append(f"{d['verdict']} -> {d['status']}: {d['test'][:30]}")
        elif n == "MODULE_RESULT":
            self.results[d["module"]] = d["status"]
        elif n == "SCAN_COMPLETE":
            self.final, self.state, self.ended = d, "SCAN COMPLETE", time.monotonic()

    def elapsed(self) -> str:
        s = int((self.ended or time.monotonic()) - self.started)
        return f"{s // 60:02d}:{s % 60:02d}"

    def render(self) -> RenderableType:
        with self._lock:
            if self.takeover and self.final:
                return Align.center(self.takeover_panel(), vertical="middle")
            root = Layout()
            hdr = Table.grid(expand=True)
            hdr.add_column(ratio=1)
            hdr.add_column(justify="right")
            hdr.add_row(Text("VERIFYFORGE", style="bold white"), Text(f"MODE: {self.mode}", style=GREEN if "LIVE" in self.mode else AMBER))
            hdr.add_row(Text("REPOSITORY VERIFICATION · READ-ONLY", style=CYAN), Text(f"JUDGE: {self.judge}" if self.judge else self.subtitle, style=DIM))
            foot = Table.grid(expand=True)
            for _ in range(3):
                foot.add_column(ratio=1, justify="center")
            foot.add_row(Text.assemble(("STATE: ", DIM), (self.state, AMBER)),
                         Text.assemble(("MODULE ", DIM), (f"{self.index} / {self.total}" if self.total else "-", "bold white")),
                         Text.assemble(("ELAPSED: ", DIM), (self.elapsed(), "bold white")))
            root.split_column(Layout(Group(hdr, Text("─" * 400, style="dim", no_wrap=True, overflow="crop")), size=3),
                              Layout(name="body"), Layout(Panel(foot, box=box.HEAVY, border_style="grey50"), size=3))
            root["body"].split_row(Layout(self._repo_pane()), Layout(self._risk_pane(), ratio=2), Layout(self._verify_pane(), ratio=2))
            return root

    @staticmethod
    def _pane(title: str, body: RenderableType, border: str = "cyan") -> Panel:
        return Panel(body, title=f"[bold]{title}[/bold]", title_align="left", border_style=border, box=box.ROUNDED, padding=(1, 2))

    def _repo_pane(self) -> Panel:
        p: list[RenderableType] = [Text(self.name or "…", style="bold white"), Text("language: python", style=DIM), Text(""),
                                   Text.assemble(("modules  ", DIM), (str(self.modules), "bold white")),
                                   Text.assemble(("tests    ", DIM), (str(self.tests), "bold white")), Text("")]
        st = self.suite.get("status")
        if st:
            color = GREEN if st == "PASS" else AMBER if st in ("NO_TESTS", "UNAVAILABLE") else RED
            p += [Text("EXISTING SUITE", style=CYAN), Text(st, style=color),
                  Text(f"{self.suite.get('passed', 0)} passed / {self.suite.get('failed', 0)} failed / {self.suite.get('collected', 0)} collected", style=DIM)]
            if self.suite.get("reason"):
                p.append(Text(_clip(self.suite["reason"], 70), style=DIM))
        elif self.state == "RUNNING EXISTING TESTS":
            p.append(Spinner("dots", text=Text("running existing tests (staged copy)", style=AMBER)))
        return self._pane("REPOSITORY", Group(*p))

    def _risk_pane(self) -> Panel:
        p: list[RenderableType] = [Text("VERIFICATION PRIORITY (not a vulnerability score)", style=DIM)]
        if not self.ranked:
            p.append(Spinner("dots", text=Text("ranking…", style=AMBER)) if "RANKING" in self.state or self.modules else Text(""))
        if self.ai_failed:
            p.append(Text(f"AI ranking unavailable: {self.ai_failed}", style=RED))
        for r in self.ranked[:9]:
            res = self.results.get(r["path"])
            conf = self.contracts.get(r["path"], "")
            mark = "▶" if r["path"] == self.current and not res else ("✓" if res == "VERIFIED" else "✗" if res == "UNVERIFIED" else " ")
            p.append(Text.assemble((f"{mark} ", STATUS_STYLE.get(res or "", "white")), (f"{r['combined']:>4}  ", "bold white"),
                                   (_clip(r["path"], 34), "white"), (f"  {conf}", DIM)))
        return self._pane("RISK / MODULES", Group(*p))

    def _verify_pane(self) -> Panel:
        p: list[RenderableType] = []
        if not self.current:
            p.append(Text("waiting for the first module", style=DIM))
        else:
            p += [Text(_clip(self.current, 44), style="bold white"),
                  Text(f"contract: {self.contracts.get(self.current, 'inferring…')}", style=CYAN), Text("")]
            h = self.hidden
            if h.get("retry"):
                p.append(Text("generated suite was invalid: regenerating", style=AMBER))
            elif "generated" in h and h.get("collected", True):
                p.append(Text(f"hidden suite: {h['generated']} generated / {h['kept']} kept", style=DIM))
            elif "generated" in h:
                p.append(Text("hidden suite could not be collected", style=RED))
            if "passed" in h:
                p.append(Text(f"{h['tests']} / {h['tests']} PASS ✓" if h["passed"] else f"{h['failed']} FAILED ✗ of {h['tests']}", style=GREEN if h["passed"] else RED))
                if self.assertion:
                    p.append(Text(_clip(self.assertion, 110), style="white"))
            elif self.current in self.contracts and self.state.startswith("VERIFYING"):
                p.append(Spinner("dots", text=Text("generating + running hidden tests", style=AMBER)))
            for t in self.triage[-3:]:
                p.append(Text("TRIAGE " + t, style=AMBER))
            res = self.results.get(self.current)
            if res:
                p += [Text(""), Text(res, style=STATUS_STYLE.get(res, "white"))]
        return self._pane("VERIFICATION", Group(*p), "green" if self.results.get(self.current) == "VERIFIED" else "cyan")

    def takeover_panel(self) -> Panel:
        f = self.final or {}
        s = f.get("summary", {})
        rows = Table.grid(padding=(0, 6))
        rows.add_column()
        rows.add_column()

        def row(k, v, style="bold white"):
            rows.add_row(Text(k, style=DIM), Text(str(v), style=style))

        row("Existing suite", s.get("existing_suite", "-"), GREEN if s.get("existing_suite") == "PASS" else AMBER)
        row("Modules discovered", s.get("python_modules", 0))
        row("Deeply verified", s.get("deeply_verified", 0))
        row("VERIFIED", s.get("verified", 0), GREEN)
        row("UNVERIFIED", s.get("unverified", 0), RED if s.get("unverified") else "bold white")
        if s.get("insufficient_contract"):
            row("INSUFFICIENT CONTRACT", s["insufficient_contract"], AMBER)
        if s.get("not_completed"):
            row("NOT COMPLETED", s["not_completed"], RED)
        row("NOT DEEPLY VERIFIED", s.get("not_deeply_verified", 0), DIM)
        row("Execution", "REAL PYTEST")
        row("Repository unchanged", "yes" if f.get("unchanged") else "NO", GREEN if f.get("unchanged") else RED)
        row("Scope", f"{s.get('deeply_verified', 0)} / {s.get('python_modules', 0)} modules deeply verified")
        if f.get("highest_risk"):
            row("Highest-risk unresolved", f["highest_risk"], AMBER)
        color = "red" if s.get("unverified") or "INCOMPLETE" in f.get("status", "") else "green"
        title = f.get("status", "REPOSITORY SCAN COMPLETE").replace(" — ", "\n")
        body = Group(Text(""), Align.center(Text(title, style=f"bold {color}")), Text(""), Align.center(rows), Text(""),
                     Align.center(Text("Verification evidence generated for selected modules.", style=f"italic {color}")), Text(""))
        return Panel(body, box=box.DOUBLE, border_style=color, padding=(1, 6), width=78)
