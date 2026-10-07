# VerifyForge — Autonomous Specification-Driven Software Engineering Agent

> **AI writes. Tests challenge. Failures heal. Software earns verification.**

VerifyForge turns a markdown spec into working Python code and a verification report:

1. **Spec** → structured requirements
2. **Generate** → implementation + tests
3. **Test** → pytest in a subprocess
4. **Adversarial verify** → hostile tests written by a separate pass
5. **Repair** → failures fed back, code patched, loop (bounded)
6. **Proof** → `verification_report.json` / `.md` with status `VERIFIED` or `UNVERIFIED`

## Usage
```bash
python3.13 -m venv .venv && source .venv/bin/activate
pip install -e .
export ANTHROPIC_API_KEY=...
verifyforge run examples/spec_slugify.md --out build --max-rounds 4
```

Model defaults to `claude-sonnet-5-5`; override with `--model` or `VERIFYFORGE_MODEL`.

## Spec format
Markdown with a `# Title` and `- ` bullets under `## Requirements`. The spec should name the module (`Module: slugify`).

## Safety note
Generated code runs in a subprocess with a timeout inside a temp dir. This is **not** a hardened sandbox; run in a container for untrusted specs.
