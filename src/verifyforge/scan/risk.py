"""Deterministic VERIFICATION PRIORITY SCORE. This ranks where independent verification is most worth its cost.
It is NOT a vulnerability or severity score and says nothing about whether the code is correct."""
from __future__ import annotations

HIGH = ("threading", "lock", "asyncio", "decimal", "money", "auth", "crypto", "state_machine", "subprocess", "database",
        "retry", "queue", "cache", "rate_limit", "network", "inventory_balance", "global_state")
MEDIUM = ("time", "file_writes", "serialization", "validation", "config_mutation", "filesystem")
HIGH_W, MEDIUM_W = 3.0, 1.5


def priority_score(module: dict) -> dict:
    sig = module.get("signals", {})
    hi = [s for s in HIGH if sig.get(s)]
    med = [s for s in MEDIUM if sig.get(s)]
    raw = len(hi) * HIGH_W + len(med) * MEDIUM_W
    raw += min(module.get("public_api", 0), 12) * 0.15 + min(module.get("loc", 0) / 400.0, 1.0)
    if module.get("public_api", 0) == 0:
        raw *= 0.3  # nothing externally callable to test
    if module.get("parse_error"):
        raw = 0.0
    score = round(min(10.0, raw), 2)
    level = "HIGH" if score >= 6 else "MEDIUM" if score >= 3 else "LOW"
    return {"score": score, "level": level, "high_signals": hi, "medium_signals": med,
            "label": "verification priority score (not a vulnerability score)"}


def rank_deterministically(repo_map: dict) -> list[dict]:
    """Candidates ordered by (-score, path): fully deterministic. Modules with no public API are not candidates."""
    rows = []
    for m in repo_map["modules"]:
        r = priority_score(m)
        r.update(path=m["path"], loc=m["loc"], public_api=m.get("public_api", 0), importable=m.get("dotted") is not None,
                 existing_tests=sum(1 for t in repo_map["test_mapping"].get(m["path"], []) if t["confidence"] in ("high", "medium")))
        rows.append(r)
    rows.sort(key=lambda r: (-r["score"], r["path"]))
    return rows
