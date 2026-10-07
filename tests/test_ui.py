import io
import json
import tempfile
from pathlib import Path

from rich.console import Console

from test_orchestrator import ADV, GOOD, SPEC, FakeLLM
from test_triage import WRONG, Scripted, verdict
from verifyforge import cli
from verifyforge.demos import RATE_LIMITER_REQUEST, rate_limiter_replay
from verifyforge.orchestrator import run, run_request
from verifyforge.ui import VerifyForgeUI, _meaningful_patch


def make_ui(mode="OFFLINE DEMO", goal="goal"):
    con = Console(file=io.StringIO(), force_terminal=True, width=132, height=34, color_system=None)
    return VerifyForgeUI(mode, goal, 3, 0, con, "x"), con


def text_of(ui):
    c = Console(file=io.StringIO(), force_terminal=True, width=132, height=34, color_system=None)
    c.print(ui.render())
    return c.file.getvalue()


def signature(report):
    return (report.status, report.repairs, [(r.tier1_passed, r.adversarial_passed, r.tests_run) for r in report.rounds])


def test_results_identical_with_and_without_ui(tmp_path: Path):
    bare = run_request(rate_limiter_replay(), RATE_LIMITER_REQUEST, tmp_path / "a")
    ui, _ = make_ui()
    shown = run_request(rate_limiter_replay(), RATE_LIMITER_REQUEST, tmp_path / "b", on_event=ui)
    assert signature(bare) == signature(shown) and ui.ui_errors == []
    assert (tmp_path / "a" / "rate_limiter.py").read_text() == (tmp_path / "b" / "rate_limiter.py").read_text()


def test_offline_run_renders_story_and_labels(tmp_path: Path):
    ui, _ = make_ui()
    frames = []
    run_request(rate_limiter_replay(), RATE_LIMITER_REQUEST, tmp_path,
                on_event=lambda n, d: (ui(n, d), frames.append((n, text_of(ui)))))
    by = {}
    for n, f in frames:
        by.setdefault(n, f)  # first occurrence: the failing adversarial round
    assert "MODE: OFFLINE DEMO" in by["VERIFIED"]
    assert "AI PROPOSES · EXECUTION VERIFIES" in by["BUILD_COMPLETE"]
    assert "observed 100" in by["ADVERSARIAL_RESULT"] and "FAILED ✗" in by["ADVERSARIAL_RESULT"]
    assert "REPAIR #1" in by["REPAIR_COMPLETE"] and "+ with self._lock:" in by["REPAIR_COMPLETE"]
    assert ui.final == "VERIFIED"
    out = Console(file=io.StringIO(), force_terminal=True, width=100, color_system=None)
    out.print(ui.takeover_panel())
    shown = out.file.getvalue()
    assert "V E R I F I E D" in shown and "REAL PYTEST" in shown and "REPLAYED (offline)" in shown


def test_live_mode_label_never_says_offline():
    ui, _ = make_ui("LIVE · CLAUDE CLI")
    out = text_of(ui)
    assert "MODE: LIVE · CLAUDE CLI" in out and "OFFLINE" not in out


def test_triage_shown_explicitly(tmp_path: Path):
    ui, _ = make_ui("LIVE · CLAUDE CLI")
    seen = []
    run(Scripted(GOOD, ADV + WRONG, verdict("INVALID", "1 + 1 is 2, the test asserts 3")), SPEC, tmp_path,
        on_event=lambda n, d: (ui(n, d), n == "TRIAGE_RESULT" and seen.append(text_of(ui))))
    assert "ADVERSARIAL TRIAGE" in seen[0] and "INVALID TEST ✗" in seen[0] and "QUARANTINED" in seen[0]
    assert "Replacement test generated ✓" in seen[0]


def test_ui_handler_swallows_garbage_events():
    ui, _ = make_ui()
    for name, data in [("SPEC_GENERATED", {}), ("TIER1_RESULT", {"oops": 1}), ("REPAIR_COMPLETE", None), ("???", {})]:
        ui(name, data or {})  # must not raise
    assert ui.ui_errors  # recorded, not raised


def test_unverified_takeover_is_red_and_honest(tmp_path: Path):
    ui, _ = make_ui("LIVE · CLAUDE CLI")
    run(Scripted("def add(a, b):\n    return 0\n", ADV, verdict("VALID"), repair_to="def add(a, b):\n    return 0\n"),
        SPEC, tmp_path, max_repairs=1, on_event=ui)
    assert ui.final == "UNVERIFIED"
    out = Console(file=io.StringIO(), force_terminal=True, width=100, color_system=None)
    out.print(ui.takeover_panel())
    assert "U N V E R I F I E D" in out.file.getvalue() and "Software earned" not in out.file.getvalue()


def test_patch_view_drops_reindentation_and_docstrings():
    diff = ('--- before\n+++ after\n-    x = 1\n+        x = 1\n-    """old doc"""\n+    """new doc"""\n'
            '-    return a\n+    with lock:\n+        return a\n')
    assert _meaningful_patch(diff) == ["+ with lock:"]


def test_cli_ui_flag_runs_same_engine(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert cli.main(["demo", "rate-limiter", "--offline", "--ui", "--pace", "0", "--out", str(tmp_path / "ui")]) == 0
    assert cli.main(["demo", "rate-limiter", "--offline", "--out", str(tmp_path / "plain")]) == 0
    a = json.loads((tmp_path / "ui" / "verification_report.json").read_text())
    b = json.loads((tmp_path / "plain" / "verification_report.json").read_text())
    key = lambda r: (r["status"], r["repairs"], [(x["tier1_passed"], x["adversarial_passed"]) for x in r["rounds"]])
    assert key(a) == key(b)
