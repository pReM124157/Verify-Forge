"""Model-facing stages of a repository scan: risk review and CONTRACT INFERENCE with provenance.

Rule: a requirement is only as trusted as the evidence behind it. Every inferred requirement must cite sources with an
exact quote; the quote is re-checked against the real file, and the confidence the model claims is CAPPED by the
provenance that actually verifies. Nothing here decides VERIFIED, and no stage sees implementation bodies."""
from __future__ import annotations

import re
from pathlib import Path

from ..llm import LLM, SYS_SCAN_CONTRACT, SYS_SCAN_REVIEW, extract_json
from ..spec import Requirement, Spec
from .safety import has_secret_like_value, read_text_safe, redact_secrets

CONF = ("insufficient", "low", "medium", "high")
MAX_CONTEXT_CHARS = 48_000  # hard bound on what any single model call may contain
README_CHARS, DOC_CHARS, TEST_CHARS, TEST_TOTAL = 3500, 1500, 3500, 9000
SOURCE_KINDS = {"readme", "doc", "docstring", "test", "signature", "comment", "config"}
STRONG_KINDS = {"readme", "doc", "docstring", "test"}


# ---------------------------------------------------------------------------------------------------------------------
def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[`*_\"'>#]", "", s.lower())).strip()


def _clip(s: str, n: int) -> str:
    return s if len(s) <= n else s[:n] + "\n...[truncated]"


def readme_excerpt(root: Path, repo_map: dict, limit: int = README_CHARS) -> tuple[str, str]:
    for d in repo_map["documentation_files"]:
        if Path(d).name.upper().startswith("README"):
            t = read_text_safe(root, d, 40_000)
            if t:
                return d, redact_secrets(_clip(t, limit))
    return "", ""


# ---------------------------------------------------------------------------------------------------------------------
# Stage: repository review (ranking). Sees a bounded SUMMARY, never source bodies.
def build_review_prompt(root: Path, repo_map: dict, det_rank: list[dict]) -> tuple[str, dict]:
    rd, readme = readme_excerpt(root, repo_map)
    lines, used = [], 0
    for r in det_rank[:60]:
        m = next(x for x in repo_map["modules"] if x["path"] == r["path"])
        api = [f["name"] for f in m["functions"] if f["public"]][:5] + [c["name"] for c in m["classes"] if c["public"]][:4]
        tests = [t["test_file"] for t in repo_map["test_mapping"].get(m["path"], []) if t["confidence"] in ("high", "medium")][:3]
        line = (f"- {r['path']} | loc {m['loc']} | det.priority {r['score']} | signals: {','.join(r['high_signals'] + r['medium_signals']) or 'none'}"
                f" | public: {', '.join(api) or 'none'} | doc: {m['docstring'][:100]!r} | tests: {', '.join(tests) or 'none'}")
        line = redact_secrets(line)
        if used + len(line) > MAX_CONTEXT_CHARS - README_CHARS - 4000:
            break
        lines.append(line)
        used += len(line)
    meta = {"modules_in_prompt": len(lines), "modules_total": len(det_rank), "truncated": len(lines) < len(det_rank),
            "readme": rd, "context_chars": used + len(readme)}
    prompt = (
        "You are reviewing a Python repository to decide where independent verification effort is best spent. "
        "You see only a summary: no implementation source.\n\n"
        "Questions: (1) which modules carry the highest CORRECTNESS risk, and why? (2) which externally observable "
        "invariants appear intended? (3) for which modules is there enough evidence (README, docstrings, signatures, "
        "existing tests) to infer a testable contract, and for which is there not?\n\n"
        'Return JSON: {"ranked_modules": [{"path": "<one of the listed paths>", "priority": <0-10>, "reasons": ["..."], '
        '"contract_confidence": "high|medium|low|insufficient"}]}\n'
        "Only list paths from the table. Rank at most 12. Priority is verification priority, not a vulnerability score. "
        "You do NOT decide whether anything is verified.\n\n"
        f"REPOSITORY: {repo_map['name']} (python, {repo_map['source_files']} source modules, {repo_map['test_files']} test files)\n"
        f"README ({rd or 'none'}):\n{readme or '(no README)'}\n\nMODULES (deterministic signals; det.priority is a heuristic):\n" + "\n".join(lines))
    return prompt, meta


def parse_review(raw: str, allowed_paths: set[str]) -> list[dict]:
    data = extract_json(raw)
    rows = data.get("ranked_modules")
    if not isinstance(rows, list):
        raise ValueError("review reply has no ranked_modules list")
    out, seen = [], set()
    for r in rows:
        if not isinstance(r, dict) or r.get("path") not in allowed_paths or r["path"] in seen:
            continue  # hallucinated or duplicate paths are dropped, never trusted
        try:
            pr = max(0.0, min(10.0, float(r.get("priority", 0))))
        except (TypeError, ValueError):
            pr = 0.0
        conf = str(r.get("contract_confidence", "low")).lower()
        out.append({"path": r["path"], "priority": round(pr, 2), "reasons": [str(x)[:200] for x in (r.get("reasons") or [])][:5],
                    "contract_confidence": conf if conf in CONF else "low"})
        seen.add(r["path"])
    return out


# ---------------------------------------------------------------------------------------------------------------------
# Stage: contract inference for ONE module. Sees docs, docstrings, signatures and existing tests: never the code bodies.
PROPERTY_DECOS = {"property", "cached_property", "functools.cached_property"}


def _is_property(x: dict) -> bool:
    return any(d in PROPERTY_DECOS or d.endswith((".setter", ".getter", ".deleter")) for d in x.get("decorators", []))


def _member_line(x: dict, indent: str = "") -> str:
    """One API line that states HOW a member is used: a @property is an attribute, never called."""
    deco = [d for d in x.get("decorators", []) if d in ("classmethod", "staticmethod")]
    doc = f"   # {x['doc']}" if x["doc"] else ""
    if _is_property(x):
        ret = f": {x['returns']}" if x.get("returns") else ""
        return f"{indent}@property {x['name']}{ret}   (an attribute: read it WITHOUT calling it){doc.replace('   #', ' #')}"
    return f"{indent}" + "".join(f"@{d} " for d in deco) + f"def {x['signature']}" + doc


def _module_api_text(m: dict) -> str:
    out = [f"module docstring: {m['docstring'][:400]!r}" if m["docstring"] else "module docstring: (none)"]
    for f in m["functions"]:
        if f["public"]:
            out.append(_member_line(f))
    for c in m["classes"]:
        if c["public"]:
            out.append(f"class {c['name']}" + (f"   # {c['doc']}" if c["doc"] else ""))
            seen = set()
            for x in c["methods"]:
                if (x["name"], _is_property(x)) in seen:
                    continue  # e.g. a property and its setter
                seen.add((x["name"], _is_property(x)))
                out.append(_member_line(x, "    "))
    return "\n".join(out)


def gather_evidence(root: Path, repo_map: dict, m: dict) -> dict:
    """The files a contract may legitimately cite for this module, with the (redacted, bounded) text shown to the model."""
    files: dict[str, str] = {}
    shown: dict[str, str] = {}
    rd, readme = readme_excerpt(root, repo_map)
    if rd:
        files[rd] = read_text_safe(root, rd, 40_000) or ""
        shown[rd] = readme
    stem = Path(m["path"]).stem.lower()
    for d in repo_map["documentation_files"]:
        if d != rd and stem in (read_text_safe(root, d, 40_000) or "").lower() and len(shown) < 3:
            t = read_text_safe(root, d, 40_000) or ""
            files[d], shown[d] = t, redact_secrets(_clip(t, DOC_CHARS))
    src = read_text_safe(root, m["path"], 200_000) or ""
    files[m["path"]] = src  # docstrings are quoted from the module file itself
    total = 0
    for t in sorted(repo_map["test_mapping"].get(m["path"], []), key=lambda x: ("high", "medium", "low").index(x["confidence"])):
        if t["confidence"] == "low" or total >= TEST_TOTAL:
            continue
        txt = read_text_safe(root, t["test_file"], 100_000)
        if txt is None:
            continue
        files[t["test_file"]] = txt
        shown[t["test_file"]] = redact_secrets(_clip(txt, TEST_CHARS))
        total += min(len(txt), TEST_CHARS)
    return {"files": files, "shown": shown, "api_text": _module_api_text(m)}


def build_contract_prompt(m: dict, ev: dict, dotted: str | None) -> str:
    docs = "\n\n".join(f"### {p}\n{t}" for p, t in ev["shown"].items())
    return (
        f"Infer the INTENDED, externally observable behaviour of the Python module `{m['path']}` "
        f"(import name: `{dotted}`) from the evidence below. You are NOT shown the implementation.\n\n"
        "Rules:\n"
        "- Only state behaviour the evidence supports. Never invent a product specification.\n"
        "- Every requirement needs sources: kind (readme|doc|docstring|test|signature|comment), file (a file shown below) "
        "and an EXACT quote copied from that file (at least 12 characters). Quotes are machine-checked.\n"
        "- confidence: high (two independent sources agree), medium (one clear source), low (weak or inferred). "
        "Do not claim more than the sources justify.\n"
        "- Requirements must be checkable through the public API only. At most 8. Prefer fewer, sharper ones.\n"
        "- If the evidence is too thin to establish intended behaviour, return an empty list and say why.\n\n"
        'Return JSON: {"requirements": [{"id": "IR1", "text": "...", "confidence": "high|medium|low", '
        '"sources": [{"kind": "...", "file": "...", "quote": "..."}]}], "insufficient_reason": "..."}\n\n'
        f"PUBLIC API:\n{ev['api_text']}\n\nEVIDENCE FILES:\n{docs}")[:MAX_CONTEXT_CHARS]


def _quote_ok(quote: str, text: str) -> bool:
    q = _norm(quote)
    return len(q) >= 12 and q in _norm(text)


def validate_contract(raw: str, ev: dict, module_path: str) -> dict:
    """Parse the model's contract and re-verify every source quote against the real files. Confidence is capped by
    provenance: the model's claim is advisory."""
    data = extract_json(raw)
    reqs_in = data.get("requirements")
    if not isinstance(reqs_in, list):
        raise ValueError("contract reply has no requirements list")
    api_norm = _norm(ev["api_text"])
    reqs = []
    for i, r in enumerate(reqs_in[:8], 1):
        if not isinstance(r, dict) or not str(r.get("text", "")).strip():
            continue
        claimed = str(r.get("confidence", "low")).lower()
        claimed = claimed if claimed in CONF else "low"
        srcs, valid_pairs, kinds = [], set(), set()
        for s in (r.get("sources") or [])[:6]:
            if not isinstance(s, dict):
                continue
            kind, f, q = str(s.get("kind", "")).lower(), str(s.get("file", "")), str(s.get("quote", ""))
            ok = False
            if kind in SOURCE_KINDS:
                if kind == "signature":
                    ok = len(_norm(q)) >= 6 and _norm(q) in api_norm
                elif f in ev["files"]:
                    ok = _quote_ok(q, ev["files"][f])
            srcs.append({"kind": kind, "file": f, "quote": redact_secrets(q)[:300], "valid": ok})
            if ok:
                valid_pairs.add((f, _norm(q)[:80]))
                kinds.add(kind)
        n = len(valid_pairs)
        if n == 0:
            cap = "insufficient"
        elif n == 1:
            cap = "medium" if kinds & STRONG_KINDS else "low"
        else:
            cap = "high" if kinds & STRONG_KINDS else "medium"
        final = CONF[min(CONF.index(claimed), CONF.index(cap))]
        reqs.append({"id": str(r.get("id") or f"IR{i}"), "text": redact_secrets(str(r["text"]).strip())[:400], "confidence": final.upper(),
                     "claimed_confidence": claimed.upper(), "validated_sources": n, "sources": srcs,
                     "downgraded": final != claimed})
    strong = [r for r in reqs if r["confidence"] in ("HIGH", "MEDIUM")]
    if any(r["confidence"] == "HIGH" for r in reqs):
        module_conf = "HIGH"
    elif strong:
        module_conf = "MEDIUM"
    elif any(r["confidence"] == "LOW" for r in reqs):
        module_conf = "LOW"
    else:
        module_conf = "INSUFFICIENT"
    return {"module": module_path, "requirements": reqs, "module_confidence": module_conf,
            "insufficient_reason": str(data.get("insufficient_reason", ""))[:300],
            "eligible_requirements": [r["id"] for r in strong]}


def contract_to_spec(contract: dict, m: dict, dotted: str) -> Spec:
    """A Spec the EXISTING hidden-verifier machinery can consume: only HIGH/MEDIUM requirements, public API only."""
    elig = [r for r in contract["requirements"] if r["confidence"] in ("HIGH", "MEDIUM")]
    api = _module_api_text(m)
    lines = [f"# Inferred contract for {dotted}", "", f"Module: {dotted}", "",
             f"The code under test is the REAL repository module `{dotted}`. Import it exactly like this in the tests: "
             f"`import {dotted}` or `from {dotted} import <name>`. Do not mock it, copy it or reimplement it. "
             "These requirements were INFERRED from repository evidence (not written by a user); only the ones below are "
             "sufficiently evidenced.", "", "## Public API (signatures only)", api, "", "## Requirements"]
    lines += [f"- {r['id']}: {r['text']} [confidence {r['confidence']}]" for r in elig]
    raw = "\n".join(lines) + "\n"
    return Spec(f"Inferred contract: {dotted}", dotted, [Requirement(r["id"], r["text"]) for r in elig], raw)


def has_secret(text: str) -> bool:
    return has_secret_like_value(text)
