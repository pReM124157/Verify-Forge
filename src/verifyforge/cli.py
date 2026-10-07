from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .llm import AnthropicLLM
from .orchestrator import run, run_request


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="verifyforge", description="Spec-driven verified code generation.")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="Run on a spec file, or a natural-language request (omit for a prompt)")
    r.add_argument("target", nargs="?", help="path to spec .md, or a quoted natural-language request")
    r.add_argument("--out", type=Path, default=Path("build"))
    r.add_argument("--max-rounds", type=int, default=4)
    r.add_argument("--model", default=None)
    args = p.parse_args(argv)

    target = args.target
    if not target:
        print("VERIFYFORGE\n\nWhat should I build?\n")
        target = input("> ").strip()
    llm = AnthropicLLM(args.model)
    path = Path(target)
    if path.suffix == ".md" and path.exists():
        report = run(llm, path.read_text(), args.out, args.max_rounds)
    else:
        report = run_request(llm, target, args.out, max_rounds=args.max_rounds)
    print(f"{report.status} after {len(report.rounds)} round(s). Report: {args.out}/verification_report.md")
    if report.error:
        print(f"Error: {report.error}", file=sys.stderr)
    return 0 if report.status == "VERIFIED" else 1


if __name__ == "__main__":
    sys.exit(main())
