# VerifyForge — Autonomous Specification-Driven Software Engineering Agent

> **AI writes. Tests challenge. Failures heal. Software earns verification.**

VerifyForge turns a request into verified Python. The tests that define correctness never see the implementation:

```
request -> ARCHITECT (spec + Tier-1 tests) -> BUILDER -> Tier-1 run
        -> pass? -> VERIFIER (hidden adversarial tests, from the spec only) -> run
        -> fail? -> REPAIR (max 3) -> rerun Tier-1 + adversarial -> VERIFIED / UNVERIFIED
```

`VERIFIED` is computed only from pytest's own record of the tests that ran (not a model's opinion). Provider and API failures are recorded as `UNVERIFIED`.

## Usage
```bash
python3.13 -m venv .venv && source .venv/bin/activate
pip install -e .
./vf demo rate-limiter --offline --ui      # stage demo: replayed model output, real pytest, three-pane UI
verifyforge demo rate-limiter --offline      # same engine, plain console output
export ANTHROPIC_API_KEY=...                 # never commit this
./vf demo rate-limiter --ui                # live model via Claude Code login (default provider: cli)
verifyforge demo rate-limiter --provider api # live model via ANTHROPIC_API_KEY
verifyforge run "Build a ..."                # natural-language request
verifyforge run examples/spec_rate_limiter.md --max-repairs 3
```

Model defaults to `claude-sonnet-5-5`; override with `--model` or `VERIFYFORGE_MODEL`.

Each run writes an audit trail to `runs/<timestamp>/`: `specification.md`, `tier1_tests.py`, `solution_vN.py`,
`adversarial_tests.py`, `repair_N.patch`, `events.jsonl`, `verification_report.{json,md}`.

macOS note: the venv's editable-install `.pth` can get marked hidden, which Python ignores. Use the `./vf` launcher
(`./vf demo rate-limiter --offline`) or run `chflags nohidden .venv/lib/python3.13/site-packages/*.pth`.

## UI
`--ui` is a pure event subscriber (`src/verifyforge/ui.py`): it never calls the orchestrator, models or pytest, and
the run produces identical results without it. The header always states `MODE: OFFLINE DEMO` (replayed model output) or
`MODE: LIVE · CLAUDE CLI`. `--pace` scales how long the UI lingers on each event (default 1.0 offline, 0 live).

## What VERIFIED means (and does not)
`VERIFIED` means: the specification kept every explicit requirement of your request (independent audit plus structural
checks), and the generated Tier-1 tests and a bounded hidden adversarial suite all really executed and passed under
pytest, with none skipped. It is **not** a formal proof. Concurrency is stress-tested, which cannot prove the absence of
races; tests and judges are LLM-generated and can be wrong (wrong tests are triaged, wrong verdicts remain a residual
risk); the hidden suite is capped (the report states generated vs kept).

## Safety note
Generated code runs in a subprocess with a timeout inside a temp dir. This is **not** a hardened sandbox: the code can
read files, write outside the temp dir and use the network (credential-looking environment variables are scrubbed, but
nothing else is isolated). The runner checks pytest's own record of what ran instead of trusting its exit code, which
defeats accidental or crude tampering (early exit, `atexit`, skipped or deselected tests) but not a determined attacker
who patches pytest in-process. Run VerifyForge in a container for untrusted requests.
