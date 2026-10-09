"""Interactive request collection: one line or a pasted multi-line spec; user text is DATA, never shell syntax."""
import builtins
import json
import os
import select
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from verifyforge import cli
from verifyforge.llm import SYS_ARCHITECT, SYS_BUILDER, SYS_PRESERVATION
from verifyforge.preservation import _signatures, deterministic_checks

REQUEST = "Build a thread-safe in-memory TTL cache with a maximum capacity of 100 items."

# The exact input from the bug report. Note the BLANK LINE inside it and the apostrophe in "product's".
INVENTORY = """Build a thread-safe in-memory inventory reservation service in Python.

Requirements:
- add_stock(product_id, quantity) increases available stock
- reserve(product_id, quantity) succeeds only when enough stock exists
- stock must never become negative
- cancel(product_id, quantity) restores previously reserved stock
- multiple threads may reserve the same product concurrently
- operations for one product must not corrupt another product's inventory
- quantities must be positive integers
- keep the public API minimal"""


class Harness:
    """Fakes the terminal (tty flag, pending-paste detection, input()), the model factory, the orchestrator entry point
    and the UI class, and records everything."""

    def __init__(self, monkeypatch, inputs=(), tty=True, pasted=False):
        self.order, self.requests, self.ui_goals, self.prompts, self.drains = [], [], [], [], []
        self.inputs = list(inputs)
        self.pasted = pasted
        h = self

        def fake_input(prompt=""):
            h.order.append("input")
            h.prompts.append(prompt)
            if not h.inputs:
                raise AssertionError("input() called but no input was scripted")
            v = h.inputs.pop(0)
            if isinstance(v, BaseException):
                raise v
            return v

        def pending(timeout=0):
            # a paste burst: everything is already queued, so input is "pending" while lines remain
            return h.pasted and bool(h.inputs)

        def fake_run_request(llm, request, out, **kw):
            h.order.append("run_request")
            h.requests.append(request)
            return SimpleNamespace(status="VERIFIED")

        class FakeUI:
            def __init__(self, mode, goal="", **kw):
                h.order.append("ui_init")
                h.ui_goals.append(goal)

            def __enter__(self):
                h.order.append("ui_enter")
                return self

            def __exit__(self, *a):
                h.order.append("ui_exit")

            def __call__(self, name, data):
                pass

        monkeypatch.setattr(builtins, "input", fake_input)
        monkeypatch.setattr(cli, "_stdin_is_tty", lambda: tty)
        monkeypatch.setattr(cli, "_input_pending", pending)
        monkeypatch.setattr(cli, "_drain_stdin", lambda: h.drains.append(1))
        monkeypatch.setattr(cli, "run_request", fake_run_request)
        monkeypatch.setattr(cli, "run", lambda *a, **k: (_ for _ in ()).throw(AssertionError("run() not expected")))
        monkeypatch.setattr(cli, "_make_llm", lambda *a, **k: object())
        monkeypatch.setattr("verifyforge.ui.VerifyForgeUI", FakeUI)
        # user text must never be handed to a shell by the collector
        for fn in ("system", "popen"):
            monkeypatch.setattr(os, fn, lambda *a, **k: (_ for _ in ()).throw(AssertionError("shell used")))
        monkeypatch.setattr(subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(AssertionError("shell used")))


def main(tmp_path, *argv):
    return cli.main(["run", *argv, "--out", str(tmp_path / "out")])


def paste(text, submit=True):
    """The input() calls a terminal produces for pasted text: its lines, then the user's empty line."""
    return text.split("\n") + ([""] if submit else [])


# ===== inline requests are unchanged ============================================================================

def test_inline_prompt_never_collects_and_reaches_orchestrator_unchanged(monkeypatch, tmp_path, capsys):
    h = Harness(monkeypatch)
    assert main(tmp_path, REQUEST) == 0
    assert h.requests == [REQUEST] and "input" not in h.order and h.drains == []
    assert "Describe what you want me to build" not in capsys.readouterr().out


def test_inline_multiline_prompt_is_passed_exactly(monkeypatch, tmp_path):
    h = Harness(monkeypatch)
    main(tmp_path, INVENTORY)
    assert h.requests == [INVENTORY] and "input" not in h.order


def test_inline_prompt_with_ui_shows_it_as_user_goal(monkeypatch, tmp_path):
    h = Harness(monkeypatch)
    assert main(tmp_path, "Build a palindrome checker", "--ui") == 0
    assert h.ui_goals == ["Build a palindrome checker"] and "input" not in h.order


def test_demo_commands_unchanged_and_never_prompt(monkeypatch, tmp_path):
    h = Harness(monkeypatch)
    rc = cli.main(["demo", "rate-limiter", "--offline", "--ui", "--pace", "0", "--out", str(tmp_path / "d")])
    assert rc == 0 and "input" not in h.order and h.ui_goals[0].startswith("Build a thread-safe") and h.drains == []


# ===== the interactive collector ================================================================================

def test_banner_text_and_first_prompt(monkeypatch, tmp_path, capsys):
    h = Harness(monkeypatch, paste(REQUEST))
    main(tmp_path)
    out = capsys.readouterr().out
    for line in ("VERIFYFORGE", "AI PROPOSES · EXECUTION VERIFIES", "Describe what you want me to build.",
                 "Paste one or multiple lines.", "Submit with an empty line."):
        assert line in out
    assert h.prompts[0] == "> " and h.prompts[1:] == [""] * (len(h.prompts) - 1)


def test_one_line_request_needs_one_empty_line_to_submit(monkeypatch, tmp_path):
    h = Harness(monkeypatch, ["Build a palindrome checker.", ""])
    assert main(tmp_path) == 0
    assert h.requests == ["Build a palindrome checker."] and h.inputs == []


def test_simple_multiline_request_keeps_its_line_breaks(monkeypatch, tmp_path):
    text = "Build X.\nRequirements:\n- a(x)\n- b(y)"
    h = Harness(monkeypatch, paste(text))
    main(tmp_path)
    assert h.requests == [text]


def test_nothing_runs_until_the_empty_line(monkeypatch, tmp_path):
    h = Harness(monkeypatch, ["line one", "line two", EOFError()])  # no empty line, then EOF
    main(tmp_path)
    assert h.requests == ["line one\nline two"]  # EOF submits what was entered (decision A)
    h2 = Harness(monkeypatch, ["a", "b", ""])
    main(tmp_path)
    assert h2.order.index("input") < h2.order.index("run_request") and h2.requests == ["a\nb"]


def test_strips_only_blank_space_around_the_whole_request_inner_lines_untouched(monkeypatch, tmp_path):
    inner = "Build X.\n\n  Requirements:\n    - indented item  \n\n- last"
    h = Harness(monkeypatch, ["", "  ", *paste("\n" + inner + "\n\n"), ], pasted=True)
    main(tmp_path)
    assert h.requests == [inner]  # blank lines and indentation INSIDE are preserved byte-for-byte


# ===== the exact reported bug ===========================================================================================

def test_exact_inventory_paste_is_fully_consumed_with_its_inner_blank_line(monkeypatch, tmp_path):
    h = Harness(monkeypatch, paste(INVENTORY), pasted=True)  # a paste burst: more input is always queued
    assert main(tmp_path, "--ui") == 0
    assert h.requests == [INVENTORY]  # 1 + 2 + all 8 requirement lines, blank line included
    assert h.inputs == [] and h.drains  # ALL lines consumed; nothing left for the shell; stdin flushed
    assert h.ui_goals == [INVENTORY] and h.order.index("input") < h.order.index("ui_init")
    assert "product's" in h.requests[0] and "\n\nRequirements:" in h.requests[0]


def test_a_typed_empty_line_after_a_blank_inside_text_is_not_swallowed_without_a_paste(monkeypatch, tmp_path):
    # Not a paste (nothing queued behind the blank line): the first empty line submits.
    h = Harness(monkeypatch, ["Build X.", "", "Requirements:"], pasted=False)
    main(tmp_path)
    assert h.requests == ["Build X."] and h.inputs == ["Requirements:"]  # this is exactly why a flush is needed


def test_apostrophes_quotes_parens_and_shell_metacharacters_are_plain_data(monkeypatch, tmp_path):
    nasty = ("Build a \"cache\" for another product's inventory (see 'notes').\n"
             "- cost: $HOME ${PATH} $(rm -rf /tmp/x) `id`; a && b || c | d > e < f\n"
             "- globs: * ? [a-z] {1,2} ~ ! # & ; \\ \" '\n"
             "- hyphen-ated, snake_case(arg-one, arg_two), 50% done, C:\\path, a=b")
    h = Harness(monkeypatch, paste(nasty), pasted=True)
    assert main(tmp_path) == 0
    assert h.requests == [nasty]  # byte-for-byte; the fakes make os.system/subprocess raise if a shell were used


@pytest.mark.parametrize("blank_lines", [[""], ["", "", ""], ["   ", "\t", ""]])
def test_blank_or_whitespace_only_input_reprompts_and_never_runs_empty(monkeypatch, tmp_path, capsys, blank_lines):
    h = Harness(monkeypatch, [*blank_lines, *paste(REQUEST)])
    assert main(tmp_path, "--ui") == 0
    assert capsys.readouterr().out.count("Please enter a software requirement.") >= 1
    assert h.requests == [REQUEST] and h.ui_goals == [REQUEST] and "" not in h.requests


def test_only_blank_input_then_eof_runs_nothing(monkeypatch, tmp_path):
    h = Harness(monkeypatch, ["", "   ", EOFError()])
    assert main(tmp_path, "--ui") == 1
    assert h.requests == [] and "ui_init" not in h.order and "run_request" not in h.order and h.drains


# ===== Ctrl+C and EOF ===================================================================================================

def test_ctrl_c_at_the_prompt_cancels_cleanly(monkeypatch, tmp_path, capsys):
    h = Harness(monkeypatch, [KeyboardInterrupt()])
    assert main(tmp_path, "--ui") == 130
    out = capsys.readouterr()
    assert "Cancelled." in out.out and "Traceback" not in out.out + out.err
    assert h.requests == [] and "ui_init" not in h.order and h.drains
    assert not (tmp_path / "out").exists()  # no run directory


def test_ctrl_c_in_the_middle_of_multiline_entry_cancels_cleanly(monkeypatch, tmp_path, capsys):
    h = Harness(monkeypatch, ["Build X.", "Requirements:", KeyboardInterrupt()])
    assert main(tmp_path, "--ui") == 130
    out = capsys.readouterr()
    assert "Cancelled." in out.out and "Traceback" not in out.out + out.err
    assert h.requests == [] and "ui_init" not in h.order and not (tmp_path / "out").exists() and h.drains


def test_eof_with_nothing_entered_exits_cleanly(monkeypatch, tmp_path, capsys):
    h = Harness(monkeypatch, [EOFError()])
    assert main(tmp_path) == 1
    out = capsys.readouterr()
    assert "No request given." in out.out and "Traceback" not in out.out + out.err
    assert h.requests == [] and "ui_init" not in h.order and h.drains


def test_eof_after_text_submits_the_accumulated_text(monkeypatch, tmp_path):
    h = Harness(monkeypatch, ["Build X.", "- one", "- two", EOFError()])
    assert main(tmp_path) == 0
    assert h.requests == ["Build X.\n- one\n- two"] and h.drains


# ===== piped / redirected stdin =========================================================================================

def test_piped_stdin_is_one_whole_document_including_blank_lines(monkeypatch, tmp_path):
    h = Harness(monkeypatch, [*INVENTORY.split("\n"), EOFError()], tty=False)
    assert main(tmp_path) == 0
    assert h.requests == [INVENTORY]


def test_piped_empty_stdin_runs_nothing(monkeypatch, tmp_path):
    h = Harness(monkeypatch, [EOFError()], tty=False)
    assert main(tmp_path) == 1 and h.requests == []


# ===== the request reaches the Architect and the preservation gate in full =========================================

def inventory_spec(omit=(), words="thread-safe, guarded by a lock"):
    parts = {
        "add": "`add_stock(product_id, quantity)` increases the available stock of a product.",
        "reserve": "`reserve(product_id, quantity)` succeeds only when enough stock exists, otherwise it fails.",
        "cancel": "`cancel(product_id, quantity)` restores previously reserved stock.",
        "neg": "Stock never becomes negative.",
        "thread": f"All operations are {words}; multiple threads may reserve the same product concurrently.",
        "iso": "Operations on one product never corrupt another product's inventory.",
        "qty": "quantities must be positive integers.",
    }
    spec = " ".join(v for k, v in parts.items() if k not in omit)
    return json.dumps({"title": "Inventory service", "module": "inventory", "specification": spec,
                       "requirements": [v for k, v in parts.items() if k not in omit], "assumptions": [],
                       "tier1_tests": "from inventory import *\n\ndef test_R1():\n    assert True\n",
                       "requirement_traceability": []})


class GateLLM:
    """Records every prompt. The Builder is unreachable: the run stops at/just after the preservation gate."""

    def __init__(self, architect_reply):
        self.architect_reply, self.calls = architect_reply, []

    def complete(self, system, prompt):
        self.calls.append((system, prompt))
        if system == SYS_ARCHITECT:
            return self.architect_reply
        if system == SYS_PRESERVATION:
            return json.dumps({"requirements": [{"user_requirement": "everything", "mapped": ["SPEC"],
                                                 "status": "PRESERVED", "evidence": "lazy auditor"}]})
        if system == SYS_BUILDER:
            raise RuntimeError("builder reached (expected for a spec that passes the gate)")
        raise AssertionError(system)

    def prompts(self, role):
        return [p for s, p in self.calls if s == role]


def run_cli_with_llm(monkeypatch, tmp_path, llm, inputs, pasted=True):
    h = Harness(monkeypatch, inputs, pasted=pasted)
    monkeypatch.setattr(cli, "run_request", __import__("verifyforge.orchestrator", fromlist=["x"]).run_request)
    monkeypatch.setattr(cli, "_make_llm", lambda *a, **k: llm)
    rc = main(tmp_path)
    return h, rc


def test_architect_and_preservation_gate_receive_every_line_of_the_pasted_request(monkeypatch, tmp_path):
    llm = GateLLM(inventory_spec())
    h, rc = run_cli_with_llm(monkeypatch, tmp_path, llm, paste(INVENTORY))
    arch = llm.prompts(SYS_ARCHITECT)[0]
    audit = llm.prompts(SYS_PRESERVATION)[0]
    for line in INVENTORY.split("\n"):  # ALL lines, in order, reach both the Architect and the auditor
        if line.strip():
            assert line in arch, line
            assert line in audit, line
    assert arch.endswith(INVENTORY) and INVENTORY in audit  # verbatim, line breaks intact
    assert h.inputs == []  # every pasted line was consumed by VerifyForge
    report = json.loads((tmp_path / "out" / "verification_report.json").read_text())
    assert report["request"] == INVENTORY
    assert INVENTORY in (tmp_path / "out" / "verification_report.md").read_text()


def test_signatures_and_rules_from_the_pasted_request_are_all_seen_by_the_gate():
    sigs = dict(_signatures(INVENTORY))
    assert sigs["add_stock"] == ["product_id", "quantity"]
    assert sigs["reserve"] == ["product_id", "quantity"]
    assert sigs["cancel"] == ["product_id", "quantity"]
    ok = json.loads(inventory_spec())["specification"]
    assert deterministic_checks(INVENTORY, "## Requirements\n- " + ok) == []


@pytest.mark.parametrize("omit,expected", [
    (("add",), "add_stock(product_id, quantity)"),
    (("reserve",), "reserve(product_id, quantity)"),
    (("cancel",), "cancel(product_id, quantity)"),
])
def test_each_signature_dropped_by_the_architect_is_caught_by_name(monkeypatch, tmp_path, omit, expected):
    llm = GateLLM(inventory_spec(omit=omit))
    run_cli_with_llm(monkeypatch, tmp_path, llm, paste(INVENTORY))
    report = json.loads((tmp_path / "out" / "verification_report.json").read_text())
    assert report["status"] == "UNVERIFIED" and BUILDER_NOT_CALLED(llm)
    assert any(expected in m for rej in report["preservation"]["rejected"] for m in rej["missing"])


def BUILDER_NOT_CALLED(llm):
    return SYS_BUILDER not in [s for s, _ in llm.calls]


def test_a_dropped_concurrency_requirement_is_caught(monkeypatch, tmp_path):
    llm = GateLLM(inventory_spec(omit=("thread",)))
    run_cli_with_llm(monkeypatch, tmp_path, llm, paste(INVENTORY))
    report = json.loads((tmp_path / "out" / "verification_report.json").read_text())
    assert report["status"] == "UNVERIFIED" and BUILDER_NOT_CALLED(llm)
    assert any("thread" in m.lower() for rej in report["preservation"]["rejected"] for m in rej["missing"])


def test_never_negative_stock_and_every_other_rule_reach_the_auditor_text(monkeypatch, tmp_path):
    llm = GateLLM(inventory_spec())
    run_cli_with_llm(monkeypatch, tmp_path, llm, paste(INVENTORY))
    audit = llm.prompts(SYS_PRESERVATION)[0]
    for needle in ("stock must never become negative", "multiple threads may reserve the same product concurrently",
                   "quantities must be positive integers", "keep the public API minimal",
                   "operations for one product must not corrupt another product's inventory"):
        assert needle in audit


# ===== Rich UI and report show the real request ===================================================================

def test_ui_user_goal_shows_headline_and_the_users_own_requirement_lines():
    import io

    from rich.console import Console

    from verifyforge.ui import VerifyForgeUI

    ui = VerifyForgeUI("LIVE · CLAUDE CLI", INVENTORY, 3, 0, Console(file=io.StringIO(), width=132), "x")

    def text():
        c = Console(file=io.StringIO(), force_terminal=True, width=132, height=34, color_system=None)
        c.print(ui.render())
        return c.file.getvalue()

    before = text()
    assert "USER GOAL" in before and "Build a thread-safe in-memory" in before and "inventory reservation" in before
    assert "YOUR REQUIREMENTS (8)" in before and "add_stock(product_id, quantity)" in before and "•" in before
    ui("PRESERVATION_PASSED", {"preserved": 8, "total": 8, "attempt": 1})
    assert "✓ add_stock" in text()  # a tick only once the audit has confirmed preservation


# ===== a REAL interactive zsh in a pty: nothing may leak back to the shell ============================================

ZSH = shutil.which("zsh")
REPO = Path(__file__).resolve().parent.parent


def _drive_zsh(collector_src: str, src_dir: str, paste_text: str, tmp_path: Path) -> tuple[str, str]:
    import fcntl
    import pty
    import struct
    import termios

    collector = tmp_path / "collector.py"
    collector.write_text(collector_src)
    pid, fd = pty.fork()
    if pid == 0:
        os.environ.update(TERM="xterm-256color", PROMPT="VFREADY> ", RPROMPT="", HISTFILE="/dev/null")
        os.execvp(ZSH, [ZSH, "-f", "-i"])
    fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", 40, 140, 0, 0))
    buf = b""

    def pump(until: bytes, timeout: float = 20.0) -> bool:
        nonlocal buf
        end = time.time() + timeout
        while time.time() < end:
            if until in buf:
                return True
            if select.select([fd], [], [], 0.2)[0]:
                try:
                    buf += os.read(fd, 65536)
                except OSError:
                    return until in buf
        return until in buf

    assert pump(b"VFREADY> ") or pump(b"%") or True
    time.sleep(0.3)
    os.write(fd, f"PROMPT='VFREADY> '; cd {REPO}; PYTHONPATH={src_dir} {sys.executable} {collector}\r".encode())
    assert pump(b"Submit with an empty line."), buf.decode("utf8", "replace")
    time.sleep(0.5)
    mark = len(buf)
    os.write(fd, (paste_text.replace("\n", "\r") + "\r\r").encode())  # the paste burst, then the empty line
    got_result = pump(b"RESULT=<<")
    got_end = pump(b">>END")
    time.sleep(1.0)  # let a leaking shell execute whatever was left behind
    os.write(fd, b"echo SHELL_IS_CLEAN_$((40+2))\r")
    clean = pump(b"SHELL_IS_CLEAN_42", 8.0)
    os.write(fd, b"exit\r")
    time.sleep(0.3)
    try:
        os.close(fd)
    except OSError:
        pass
    text = buf.decode("utf8", "replace")
    assert got_result and got_end, text[mark:]
    return text[mark:], ("clean" if clean else "unresponsive")


COLLECTOR = '''import sys
from verifyforge import cli
try:
    text = cli._read_request()
except cli._NoRequest as e:
    sys.exit(e.code)
print("RESULT=<<" + text + ">>END")
'''


@pytest.mark.skipif(not ZSH or sys.platform == "win32", reason="needs zsh and a pty")
def test_real_zsh_paste_of_the_exact_inventory_prompt_leaks_nothing_to_the_shell(tmp_path):
    out, state = _drive_zsh(COLLECTOR, str(REPO / "src"), INVENTORY, tmp_path)
    assert "RESULT=<<" + INVENTORY.replace("\n", "\r\n") + ">>END" in out or INVENTORY.split("\n")[-1] in out
    for sign in ("command not found", "quote>", "dquote>", "missing delimiter", "no matches found", "parse error",
                 "not an identifier", "bad pattern"):
        assert sign not in out, f"shell executed leaked input: {sign!r}\n{out}"
    assert state == "clean"  # zsh is back at a normal prompt and answers
