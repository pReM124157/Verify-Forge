from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

from .llm import LLM, AnthropicLLM, ClaudeCLI
from .orchestrator import run, run_request


def _plain_printer(name: str, d: dict) -> None:
    """Minimal console subscriber; the Rich UI will subscribe to the same events."""
    ok = lambda p: "PASS ✓" if p else "FAIL ✗"
    if name == "SPEC_GENERATED":
        print(f"ARCHITECT   spec ready: {d['requirements']} requirements, module `{d['module']}`")
    elif name == "PRESERVATION_PASSED":
        print(f"ARCHITECT   {d['preserved']} / {d['total']} user requirements preserved (independent audit)")
    elif name == "SPEC_REJECTED":
        print(f"ARCHITECT   SPEC REJECTED: {d['preserved']} / {d['total']} user requirements preserved")
        for m in d["missing"][:6]:
            print(f"            - {m[:150]}")
    elif name == "SPEC_REGENERATING":
        print("ARCHITECT   regenerating the specification with the audit feedback ...")
    elif name == "TIER1_TESTS_READY":
        print("ARCHITECT   Tier-1 tests written before any implementation exists")
    elif name == "BUILD_COMPLETE":
        print(f"BUILDER     solution v{d['version']} written")
    elif name == "TIER1_RESULT":
        print(f"TIER-1      {ok(d['passed'])}  ({d['tests']} tests, {d['seconds']:.2f}s, exit {d['exit_code']})")
    elif name == "ADVERSARIAL_GENERATED":
        print("VERIFIER    hidden adversarial suite generated from the spec only")
    elif name == "ADVERSARIAL_RESULT":
        print(f"ADVERSARIAL {ok(d['passed'])}  ({d['tests']} tests, {d['seconds']:.2f}s, exit {d['exit_code']})")
        if not d["passed"]:
            lines = [l for l in d["text"].splitlines() if l.startswith(("E ", "FAILED"))][:4]
            print("            " + "\n            ".join(lines))
    elif name == "ADVERSARIAL_CAPPED":
        how = "truncated to a bounded subset" if d["truncated"] else "regenerated within the cap"
        print(f"VERIFIER    suite over cap ({d.get('first_attempt', d['generated'])} > {d['cap']}): {how}, kept {d['kept']}")
    elif name == "TRIAGE_STARTED":
        print(f"TRIAGE      auditing {d['test'].split('::')[-1]} (spec + test only, no implementation)")
    elif name == "TRIAGE_RESULT":
        print(f"TRIAGE      {d['status']}: {d['reason'][:140]}")
    elif name == "REPAIR_STARTED":
        print(f"REPAIR      {d['repair']} / {d['of']} ...")
    elif name == "REGRESSION_DETECTED":
        print("REGRESSION  repair broke something that previously passed ✗")
    elif name in ("REVERIFY", "FINAL_VERIFY"):
        print("REGRESSION  rerunning Tier-1 + adversarial on the patched code")
    elif name == "VERIFIED":
        print(f"\nVERIFIED ✓  ({d['repairs']} repair(s))")
    elif name == "UNVERIFIED":
        why = f"  reason: {d['reason']}" if d.get("reason") else ""
        print(f"\nUNVERIFIED ✗  ({d['repairs']} repair(s) used){why}")
    elif name == "ERROR":
        print(f"ERROR       {d['error']}", file=sys.stderr)


def _make_llm(provider: str, model: str | None) -> LLM:
    return AnthropicLLM(model) if provider == "api" else ClaudeCLI(model)


class _NoRequest(Exception):
    """The user cancelled (Ctrl+C, exit 130) or stdin closed (EOF, exit 1) before giving a request."""

    def __init__(self, code: int):
        self.code = code


PASTE_GAP = 0.3  # seconds: input still arriving within this window is part of the same paste


def _stdin_is_tty() -> bool:
    try:
        return sys.stdin.isatty()
    except Exception:
        return False


def _input_pending(timeout: float = PASTE_GAP) -> bool:
    """True if more terminal input is already queued (i.e. a paste is still arriving). Never true for pipes."""
    if not _stdin_is_tty():
        return False
    try:
        import select

        return bool(select.select([sys.stdin], [], [], timeout)[0])
    except Exception:
        return False


def _drain_stdin() -> None:
    """Discard unread terminal input so leftover pasted lines can never reach the shell after we exit."""
    if not _stdin_is_tty():
        return
    try:
        import termios

        fd = sys.stdin.fileno()
        for _ in range(10):  # a slow paste may still be arriving: flush, wait a moment, flush again
            termios.tcflush(fd, termios.TCIFLUSH)
            if not _input_pending(0.1):
                break
    except Exception:
        pass


def _read_request() -> str:
    """Collect a software requirement (one line or a pasted multi-line spec) on a plain terminal, before any Rich UI
    or model call. The text is treated purely as data.

    Interactive terminal: lines are read until an empty line. A blank line inside a paste (more input is already
    queued) is kept, so a pasted spec with blank lines is never cut short. EOF after some text submits that text.
    Piped/redirected stdin: the whole input is read until EOF.
    Returns the request with leading/trailing blank space removed (inner lines untouched). Raises _NoRequest on
    Ctrl+C (130) or on EOF with nothing entered (1). Unread terminal input is flushed on every path."""
    tty = _stdin_is_tty()
    print("VERIFYFORGE\nAI PROPOSES · EXECUTION VERIFIES\n\n"
          "Describe what you want me to build.\nPaste one or multiple lines.\nSubmit with an empty line.\n")
    lines: list[str] = []
    try:
        while True:
            try:
                line = input("> " if not lines else "")
            except KeyboardInterrupt:
                print("\nCancelled.")
                raise _NoRequest(130)
            except EOFError:
                if "\n".join(lines).strip():
                    print()
                    break  # EOF after real text: submit what was entered (deterministic)
                print("\nNo request given.")
                raise _NoRequest(1)
            if not tty or line.strip():
                lines.append(line)  # piped input is a whole document; typed/pasted text lines always accumulate
                continue
            if not "\n".join(lines).strip():  # blank line(s) before any text
                lines.clear()
                print("\nPlease enter a software requirement.\n")
                continue
            if _input_pending():  # blank line inside a paste that is still arriving: keep it
                lines.append("")
                continue
            break  # a deliberate empty line submits
    finally:
        _drain_stdin()
    return "\n".join(lines).strip()


def _default_out() -> Path:
    return Path("runs") / datetime.now().strftime("%Y-%m-%dT%H-%M-%S")


def main(argv: list[str] | None = None) -> int:
    state = {"prompted": False}
    try:
        return _main(argv, state)
    finally:
        if state["prompted"]:
            _drain_stdin()  # whatever the run ended with, never leave pasted leftovers for the shell to execute


def _main(argv: list[str] | None, state: dict) -> int:
    p = argparse.ArgumentParser(prog="verifyforge", description="Spec-driven verified code generation.")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--out", type=Path, default=None, help="default: runs/<timestamp>")
        sp.add_argument("--max-repairs", type=int, default=3)
        sp.add_argument("--max-adversarial", type=int, default=15, help="hard cap on hidden adversarial tests")
        sp.add_argument("--model", default=None)
        sp.add_argument("--ui", action="store_true", help="Rich three-pane presentation (event subscriber only)")
        sp.add_argument("--pace", type=float, default=None,
                        help="UI hold-time multiplier so an audience can read; default 1.0 offline, 0 live")
        sp.add_argument("--provider", choices=["cli", "api"], default="cli",
                        help="cli: headless Claude Code login (default); api: ANTHROPIC_API_KEY")

    r = sub.add_parser("run", help="Run on a spec .md file or a natural-language request (omit for a prompt)")
    r.add_argument("target", nargs="?")
    common(r)
    d = sub.add_parser("demo", help="Run a built-in demo")
    d.add_argument("name", choices=["rate-limiter"])
    d.add_argument("--offline", action="store_true", help="replay saved model outputs; pytest still runs for real")
    common(d)
    args = p.parse_args(argv)
    out = args.out or _default_out()
    offline = args.cmd == "demo" and args.offline

    # Resolve the user's request BEFORE building the UI or touching any model, so the Architect can never be
    # started with an empty goal and the prompt is visible on a normal terminal.
    target = None
    if args.cmd == "run":
        try:
            if args.target and args.target.strip():
                target = args.target  # inline request: used exactly as given, no interactive collector
            else:
                state["prompted"] = True
                target = _read_request()
        except _NoRequest as e:
            return e.code
    hook = _plain_printer
    ui = None
    if args.ui:
        from .ui import VerifyForgeUI

        mode = "OFFLINE DEMO" if offline else ("LIVE · ANTHROPIC API" if args.provider == "api" else "LIVE · CLAUDE CLI")
        pace = args.pace if args.pace is not None else (1.0 if offline else 0.0)
        goal = ""
        if args.cmd == "demo":
            from .demos import RATE_LIMITER_REQUEST as goal
        elif target:
            goal = target
        ui = VerifyForgeUI(mode, goal=goal, max_repairs=args.max_repairs, pace=pace,
                           subtitle="REPLAYED MODEL OUTPUT · REAL PYTEST" if offline else "LIVE MODEL OUTPUT · REAL PYTEST")
        hook = ui

    try:
        if ui:
            ui.__enter__()
        if args.cmd == "demo":
            from .demos import RATE_LIMITER_REQUEST, rate_limiter_replay

            llm = rate_limiter_replay() if args.offline else _make_llm(args.provider, args.model)
            report = run_request(llm, RATE_LIMITER_REQUEST, out, max_repairs=args.max_repairs, max_adversarial=args.max_adversarial, on_event=hook)
        else:
            llm = _make_llm(args.provider, args.model)
            path = Path(target)
            if "\n" not in target and path.suffix == ".md" and path.exists():
                report = run(llm, path.read_text(), out, args.max_repairs, on_event=hook, max_adversarial=args.max_adversarial)
            else:
                report = run_request(llm, target, out, max_repairs=args.max_repairs, max_adversarial=args.max_adversarial, on_event=hook)
    except ValueError as e:  # e.g. malformed Architect output: nothing was built
        if ui:
            ui.__exit__(None, None, None)
        print(f"ERROR       {e}", file=sys.stderr)
        return 2
    except BaseException:
        if ui:
            ui.__exit__(None, None, None)
        raise
    if ui:
        ui.__exit__(None, None, None)
    print(f"Report: {out}/verification_report.md")
    return 0 if report.status == "VERIFIED" else 1


if __name__ == "__main__":
    sys.exit(main())
