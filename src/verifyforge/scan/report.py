"""Markdown rendering of a repository scan report. Never says REPOSITORY VERIFIED."""
from __future__ import annotations


def _row(*cells) -> str:
    return "| " + " | ".join(str(c).replace("|", "/") for c in cells) + " |"


def render_markdown(r: dict, combined: list[dict]) -> str:
    s, suite = r["summary"], r.get("existing_suite") or {}
    L = ["# VerifyForge Repository Scan", "", f"**{r['status']}**", "",
         "> This is an evidence report for selected modules, not a repository-wide verification. Contracts are inferred from the repository, not authored by a user.", "",
         "## Repository", f"- name: `{r['name']}`", f"- root: `{r['root']}`", f"- language: {(r.get('repository') or {}).get('language') or 'none found'}",
         f"- Python modules: {s['python_modules']}", f"- existing test files: {s['existing_test_files']}",
         f"- test framework: {(r.get('repository') or {}).get('test_framework') or 'none'}",
         f"- config files: {', '.join((r.get('repository') or {}).get('config_files', [])) or 'none'}",
         f"- repository unchanged by the scan: **{'yes' if r.get('repository_unchanged') else 'NO: see fingerprints'}**",
         f"- provider: {r.get('provider') or 'n/a'}  models: {r.get('models') or 'n/a'}", ""]
    sk = (r.get("repository") or {}).get("skipped") or {}
    if sk.get("secret") or sk.get("binary") or sk.get("large") or sk.get("symlink_outside"):
        L += ["Files not read: " + ", ".join(f"{k} {len(v) if isinstance(v, list) else v}" for k, v in sk.items() if v)
              + " (names of secret-like files are listed in repository_map.json; their contents were never read, sent or logged).", ""]
    L += ["## Existing test suite", f"- status: **{suite.get('status', 'NOT RUN')}**"]
    if suite:
        L += [f"- collected {suite.get('collected', 0)}, passed {suite.get('passed', 0)}, failed {suite.get('failed', 0)}, skipped {suite.get('skipped', 0)}, "
              f"xfailed {suite.get('xfailed', 0)}", f"- command: `{suite.get('command', '')}`"]
        if suite.get("reason"):
            L.append(f"- reason: {suite['reason']}")
    L += ["", "## Verification priority ranking (not a vulnerability score)", "",
          _row("Module", "Deterministic", "AI", "Combined", "Contract", "Result"), _row("---", "---", "---", "---", "---", "---")]
    for c in combined[:15]:
        res = r["modules"].get(c["path"], {})
        L.append(_row(c["path"], c["score"], c["llm_priority"] if c["llm_priority"] is not None else "-", c["combined"],
                      (res.get("contract") or {}).get("module_confidence", "-"), res.get("status", "-")))
    L += ["", "## Deep verification", ""]
    deep = [(p, m) for p, m in r["modules"].items() if m["status"] != "NOT DEEPLY VERIFIED"]
    if not deep:
        L.append("No module was deeply verified in this scan.")
    for path, m in deep:
        L += [f"### {path}", f"**Result: {m['status']}**" + (f"  ({m['reason']})" if m.get("reason") else ""), ""]
        c = m.get("contract")
        if c:
            L.append(f"Inferred contract (module confidence {c['module_confidence']}):")
            for q in c["requirements"]:
                srcs = "; ".join(f"{x['kind']} `{x['file']}`" + ("" if x["valid"] else " (quote NOT verified)") for x in q["sources"]) or "none"
                L.append(f"- {q['id']} [{q['confidence']}{' (downgraded from ' + q['claimed_confidence'] + ')' if q['downgraded'] else ''}] {q['text']}  \n  sources: {srcs}")
            L.append("")
        if m.get("existing_tests"):
            L.append("Existing tests mapped: " + ", ".join(f"`{t}`" for t in m["existing_tests"]))
        h = m.get("hidden")
        if h:
            L += [f"Hidden verification: {'PASS' if h['passed'] else 'FAIL'}",
                  f"Scope: {h['generated']} generated / {h['kept']} kept / {h['executed']} executed (cap {h['cap']}"
                  + (", bounded" if h["truncated"] else "") + f"), quarantined by triage: {h['quarantined']}"]
        for ev in m.get("failure_evidence", []):
            L += ["", f"Failing hidden test `{ev['test']}`:", "```", ev["assertion"] or "(no assertion text captured)", "```"]
        for t in m.get("triage", []):
            L.append(f"- triage `{t['test']}`: {t['verdict']} -> {t['status']}: {t['reason'][:200]}")
        L.append("")
    other = [(p, m) for p, m in r["modules"].items() if m["status"] == "NOT DEEPLY VERIFIED"]
    L += ["## Modules not deeply verified", f"{len(other)} modules were mapped but not selected. They are NOT verified.", ""]
    L += [f"- `{p}`" for p, _ in other[:40]] + (["- ..."] if len(other) > 40 else [])
    u = r.get("model_usage") or {}
    if u:
        L += ["", "## Model usage", _row("Stage", "Model", "Calls", "Input tokens", "Output tokens", "Seconds"), _row("---", "---", "---", "---", "---", "---")]
        L += [_row(k, v["model"], v["calls"], v["input_tokens"], v["output_tokens"], v["seconds"]) for k, v in u["roles"].items()]
        t = u["total"]
        L.append(_row("**total**", "", t["calls"], t["input_tokens"], t["output_tokens"], t["seconds"]))
    if r.get("errors"):
        L += ["", "## Errors"] + [f"- {e['stage']}: {e['code']} {e.get('message', '')[:160]}" for e in r["errors"]]
    L += ["", "## Limitations",
          "- Repository-wide VERIFIED is not claimed; only the modules listed under Deep verification have evidence, and only for the stated scope.",
          "- Contracts are INFERRED from README, docstrings, signatures and existing tests; they are not user-authored requirements.",
          "- Repository code executes (existing tests, import probes, generated tests) in a staged copy. This is NOT hardened isolation: do not scan untrusted repositories.",
          "- Only Python and pytest are supported. Dependencies are never installed.",
          "- Hidden suites are bounded (generated vs kept is stated) and generated by a model; testing is not formal proof.",
          "- The scan is read-only: nothing was repaired or modified.", ""]
    return "\n".join(L)
