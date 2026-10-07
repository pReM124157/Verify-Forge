from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

from .llm import AnthropicLLM
from .orchestrator import run, run_request


def _plain_printer(name: str, d: dict) -> None:
    """Minimal console subscriber; the Rich UI will subscribe to the same events."""
    ok = lambda p: "PASS ✓" if p else "FAIL ✗"
    if name == "SPEC_GENERATED":
        print(f"ARCHITECT   spec ready: {d['requirements']} requirements, module `{d['module']}`")
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
    elif name == "REPAIR_STARTED":
        print(f"REPAIR      {d['repair']} / {d['of']} ...")
    elif name == "REGRESSION_DETECTED":
        print("REGRESSION  repair broke something that previously passed ✗")
    elif name in ("REVERIFY", "FINAL_VERIFY"):
        print("REGRESSION  rerunning Tier-1 + adversarial on the patched code")
    elif name == "VERIFIED":
        print(f"\nVERIFIED ✓  ({d['repairs']} repair(s))")
    elif name == "UNVERIFIED":
        print(f"\nUNVERIFIED ✗  ({d['repairs']} repair(s) used)")
    elif name == "ERROR":
        print(f"ERROR       {d['error']}", file=sys.stderr)


def _default_out() -> Path:
    return Path("runs") / datetime.now().strftime("%Y-%m-%dT%H-%M-%S")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="verifyforge", description="Spec-driven verified code generation.")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--out", type=Path, default=None, help="default: runs/<timestamp>")
        sp.add_argument("--max-repairs", type=int, default=3)
        sp.add_argument("--model", default=None)

    r = sub.add_parser("run", help="Run on a spec .md file or a natural-language request (omit for a prompt)")
    r.add_argument("target", nargs="?")
    common(r)
    d = sub.add_parser("demo", help="Run a built-in demo")
    d.add_argument("name", choices=["rate-limiter"])
    d.add_argument("--offline", action="store_true", help="replay saved model outputs; pytest still runs for real")
    common(d)
    args = p.parse_args(argv)
    out = args.out or _default_out()

    try:
        if args.cmd == "demo":
            from .demos import RATE_LIMITER_REQUEST, rate_limiter_replay

            llm = rate_limiter_replay() if args.offline else AnthropicLLM(args.model)
            report = run_request(llm, RATE_LIMITER_REQUEST, out, max_repairs=args.max_repairs, on_event=_plain_printer)
        else:
            target = args.target
            if not target:
                print("VERIFYFORGE\n\nWhat should I build?\n")
                target = input("> ").strip()
            llm = AnthropicLLM(args.model)
            path = Path(target)
            if path.suffix == ".md" and path.exists():
                report = run(llm, path.read_text(), out, args.max_repairs, on_event=_plain_printer)
            else:
                report = run_request(llm, target, out, max_repairs=args.max_repairs, on_event=_plain_printer)
    except ValueError as e:  # e.g. malformed Architect output: nothing was built
        print(f"ERROR       {e}", file=sys.stderr)
        return 2
    print(f"Report: {out}/verification_report.md")
    return 0 if report.status == "VERIFIED" else 1


if __name__ == "__main__":
    sys.exit(main())
