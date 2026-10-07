# VerifyForge — Autonomous Specification-Driven Software Engineering Agent

> **AI writes. Tests challenge. Failures heal. Software earns verification.**

VerifyForge turns a request into verified Python. The tests that define correctness never see the implementation:

```
request -> ARCHITECT (spec + Tier-1 tests) -> BUILDER -> Tier-1 run
        -> pass? -> VERIFIER (hidden adversarial tests, from the spec only) -> run
        -> fail? -> REPAIR (max 3) -> rerun Tier-1 + adversarial -> VERIFIED / UNVERIFIED
```

`VERIFIED` is computed only from real pytest exit codes. API failures are recorded as `UNVERIFIED`.

## Usage
```bash
python3.13 -m venv .venv && source .venv/bin/activate
pip install -e .
verifyforge demo rate-limiter --offline      # no API key: replays saved model output, real pytest
export ANTHROPIC_API_KEY=...                 # never commit this
verifyforge demo rate-limiter                # same demo, live model
verifyforge run "Build a ..."                # natural-language request
verifyforge run examples/spec_rate_limiter.md --max-repairs 3
```

Model defaults to `claude-sonnet-5-5`; override with `--model` or `VERIFYFORGE_MODEL`.

Each run writes an audit trail to `runs/<timestamp>/`: `specification.md`, `tier1_tests.py`, `solution_vN.py`,
`adversarial_tests.py`, `repair_N.patch`, `events.jsonl`, `verification_report.{json,md}`.

macOS note: if `import verifyforge` fails in the venv, run `chflags nohidden .venv/lib/python3.13/site-packages/*.pth`.

## Safety note
Generated code runs in a subprocess with a timeout inside a temp dir. This is **not** a hardened sandbox; run in a container for untrusted specs.
