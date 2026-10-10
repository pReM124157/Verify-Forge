"""Requirement-preservation gate: does the generated specification keep every explicit USER requirement?

Runs after the Architect and before any code exists. The Architect never grades itself: a fresh auditor call sees
the original request and the spec, and deterministic structural checks run alongside it. Either layer can reject.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .llm import LLM, SYS_PRESERVATION, extract_json
from .spec import parse_spec

REASON_NOT_PRESERVED = "SPECIFICATION_DID_NOT_PRESERVE_USER_REQUIREMENTS"
REASON_AUDIT_UNAVAILABLE = "PRESERVATION_AUDIT_UNAVAILABLE"

PRESERVED = "PRESERVED"
BAD_STATUSES = {"MISSING", "CHANGED", "WEAKENED"}

# Names that look like calls in prose but are not part of a requested public API.
_NOT_API = {
    "len", "range", "str", "int", "float", "bool", "list", "dict", "set", "tuple", "print", "sum", "min", "max",
    "sorted", "isinstance", "type", "open", "abs", "round", "any", "all", "map", "filter", "zip", "enumerate", "id",
    "hash", "repr", "next", "iter", "pow", "divmod", "bytes", "super", "input", "format", "eval", "exec",
}

_CONCURRENCY_ASK = re.compile(r"thread[- ]?safe|thread|concurren|parallel|race condition|simultaneous", re.I)
_CONCURRENCY_SPEC = re.compile(r"thread|concurren|\block|atomic|\brace\b|parallel|simultaneous", re.I)
_LRU_ASK = re.compile(r"\blru\b|least[- ]recently[- ]used", re.I)
_LRU_SPEC = re.compile(r"\blru\b|least[- ]recently[- ]used", re.I)
_FIFO_ASK = re.compile(r"\bfifo\b|first[- ]in[, ]+first[- ]out", re.I)
_FIFO_SPEC = _FIFO_ASK


@dataclass
class Preservation:
    passed: bool
    items: list[dict] = field(default_factory=list)  # {"user_requirement","mapped","status","evidence","source"}
    audit_ok: bool = True
    audit_error: str = ""

    @property
    def total(self) -> int:
        return len(self.items)

    @property
    def preserved(self) -> int:
        return sum(i["status"] == PRESERVED for i in self.items)

    @property
    def missing(self) -> list[str]:
        return [f"{i['user_requirement']} — {i['status']}: {i['evidence']}".strip(" :—")
                for i in self.items if i["status"] != PRESERVED]


# ---- deterministic structural checks ---------------------------------------------------------------------------

def _signatures(request: str) -> list[tuple[str, list[str]]]:
    """Explicit call signatures in the request, e.g. put(key, value, ttl_seconds) -> ('put', [key, value, ttl_seconds])."""
    out = []
    for m in re.finditer(r"(?<![\w.])([A-Za-z_][A-Za-z0-9_]*)\(([^()]*)\)", request):
        name = m.group(1)
        if name.lower() in _NOT_API:
            continue
        args = []
        ok = True
        for raw in [a.strip() for a in m.group(2).split(",") if a.strip()]:
            ident = re.match(r"^\*{0,2}([A-Za-z_][A-Za-z0-9_]*)\s*(?::[^=]+)?(?:=.*)?$", raw)
            if not ident:
                ok = False
                break
            args.append(ident.group(1))
        if ok:
            out.append((name, args))
    return out


def _spec_has_signature(spec_text: str, name: str, args: list[str]) -> bool:
    for m in re.finditer(rf"(?<![\w]){re.escape(name)}\s*\(([^()]*)\)", spec_text):
        group, pos, ok = m.group(1), 0, True
        for a in args:
            hit = re.compile(rf"(?<![\w]){re.escape(a)}(?![\w])").search(group, pos)
            if not hit:
                ok = False
                break
            pos = hit.end()
        if ok:
            return True
    return False


def _numbers(request: str) -> list[str]:
    nums = []
    for m in re.finditer(r"(?<![\w.])(\d+(?:\.\d+)?)(?![\w])", request):
        n = m.group(1)
        before = request[max(0, m.start() - 10): m.start()].lower()
        if n in ("0", "1") or re.search(r"(python|version|\bv|tier-?|step|phase)\s*$", before):
            continue  # trivial constants, interpreter versions and labels are not behavioural limits
        nums.append(n)
    return list(dict.fromkeys(nums))


def _spec_body(spec_text: str) -> str:
    """The text that actually states requirements: not the title, the Module line or the Assumptions section."""
    keep, skip = [], False
    for line in spec_text.splitlines():
        st = line.strip()
        if st.startswith("# ") or st.lower().startswith("module:"):
            continue
        if st.startswith("## "):
            skip = st[3:].strip().lower().startswith("assumption")
            continue
        if not skip:
            keep.append(line)
    return "\n".join(keep)


def deterministic_checks(request: str, spec_text: str) -> list[dict]:
    """Clear structural failures only. Returns MISSING items; an empty list means 'nothing structurally wrong'."""
    fails: list[dict] = []
    spec_text = _spec_body(spec_text)

    def fail(what: str, why: str) -> None:
        fails.append({"user_requirement": what, "mapped": [], "status": "MISSING", "evidence": why,
                      "source": "structural check"})

    for name, args in _signatures(request):
        if not _spec_has_signature(spec_text, name, args):
            fail(f"{name}({', '.join(args)})",
                 f"the specification has no `{name}(...)` containing all of: {', '.join(args) or '(no arguments)'}")
    for n in _numbers(request):
        if not re.search(rf"(?<![\d.]){re.escape(n)}(?![\d])", spec_text):
            fail(f"numeric limit {n}", f"the number {n} does not appear anywhere in the specification")
    if _CONCURRENCY_ASK.search(request) and not _CONCURRENCY_SPEC.search(spec_text):
        fail("thread-safety / concurrency", "the specification never mentions threads, concurrency, locking or atomicity")
    if _LRU_ASK.search(request) and not _LRU_SPEC.search(spec_text):
        fail("LRU eviction", "the specification never states least-recently-used eviction")
    if _FIFO_ASK.search(request) and not _FIFO_SPEC.search(spec_text):
        fail("FIFO ordering", "the specification never states first-in-first-out ordering")
    return fails


# ---- independent auditor ----------------------------------------------------------------------------------------

def _audit_prompt(request: str, spec_text: str, claims: list[dict]) -> str:
    reqs = "\n".join(f"R{i}: {r.text}" for i, r in enumerate(parse_spec(spec_text).requirements, 1))
    claim_txt = "\n".join(
        f"- {c.get('user_requirement', '?')} -> {c.get('mapped_requirements', [])} ({c.get('status', '?')})"
        for c in claims) or "(none supplied)"
    return (
        "Audit whether a generated SPECIFICATION preserves every explicit requirement in the original USER REQUEST. "
        "Extract the explicit requirements from the request yourself: public function/class/method names, exact "
        "signatures and arguments, numeric limits and capacities, behavioural rules, concurrency/thread-safety, "
        "eviction/ordering/expiry policies, error behaviour, and every MUST/NEVER. Then judge each one against the "
        "specification.\n\n"
        "Status per requirement: PRESERVED (the spec states it exactly or stricter), CHANGED (renamed, a parameter "
        "moved to different scope, different semantics, or a substitute behaviour), WEAKENED (a MUST/NEVER became "
        "optional, or a limit was relaxed), MISSING (absent). Vague restatements do not count as PRESERVED. A renamed "
        "public method is CHANGED. A per-operation parameter turned into constructor/global configuration is CHANGED. "
        "Extra features in the spec are not your concern; only what the user asked for.\n"
        "The Architect's own claims below are UNVERIFIED and may be wrong; do not rely on them.\n\n"
        'Return JSON: {"requirements": [{"user_requirement": "<short quote>", "mapped": ["R2"], '
        '"status": "PRESERVED|CHANGED|WEAKENED|MISSING", "evidence": "<one sentence citing the spec>"}]}\n'
        "Use mapped ids from the list below, or [\"SPEC\"] when it is stated in the specification prose.\n\n"
        f"USER REQUEST:\n{request}\n\nSPECIFICATION:\n{spec_text}\n\nSPEC REQUIREMENT IDS:\n{reqs}\n\n"
        f"ARCHITECT'S OWN CLAIMS (unverified):\n{claim_txt}"
    )


def _parse_audit(raw: str) -> list[dict]:
    data = extract_json(raw)
    rows = data.get("requirements")
    if not isinstance(rows, list) or not rows:
        raise ValueError("auditor returned no requirements")
    items = []
    for r in rows:
        if not isinstance(r, dict) or not str(r.get("user_requirement", "")).strip():
            raise ValueError("auditor returned a malformed requirement entry")
        status = str(r.get("status", "")).upper()
        items.append({
            "user_requirement": str(r["user_requirement"]).strip(),
            "mapped": [str(m) for m in (r.get("mapped") or [])],
            "status": status if status == PRESERVED or status in BAD_STATUSES else "MISSING",  # unknown => not preserved
            "evidence": str(r.get("evidence", "")).strip(),
            "source": "auditor",
        })
    return items


_STOP = {"that", "this", "with", "must", "never", "when", "where", "which", "their", "there", "should", "would", "could",
         "each", "only", "than", "them", "then", "have", "from", "into", "does", "same", "also", "used", "uses", "using",
         "make", "keep", "more", "most", "some", "such"}


def _words(text: str) -> set[str]:
    words = {w for w in re.findall(r"[a-z_][a-z0-9_]{3,}", text.lower()) if w not in _STOP}
    return {w[:-1] if len(w) > 4 and w.endswith("s") and not w.endswith("ss") else w for w in words}  # threads ~ thread


def _coverage_gaps(request: str, items: list[dict]) -> list[dict]:
    """Every bullet/numbered requirement line of the request must be touched by at least one audit item (shares a
    content word). A lone vague item can no longer stand in for a whole list of requirements."""
    covered = [_words(i["user_requirement"] + " " + i.get("evidence", "")) for i in items]
    gaps = []
    for line in request.splitlines():
        m = re.match(r"^\s*(?:[-*•]|\d+[.)])\s+(.*\S)", line)
        if m and not any(_words(m.group(1)) & c for c in covered):
            gaps.append({"user_requirement": m.group(1).strip(), "mapped": [], "status": "MISSING", "source": "coverage check",
                         "evidence": "no audit item covers this requirement line of the request"})
    return gaps


def _validate_ids(items: list[dict], spec_text: str) -> list[dict]:
    """Requirement ids cited by the auditor must exist in the spec. A PRESERVED claim that cites only ids that do not
    exist is not accepted."""
    valid = {f"R{i}" for i, _ in enumerate(parse_spec(spec_text).requirements, 1)}
    out = []
    for it in items:
        it = dict(it)
        cited = [m for m in it["mapped"] if re.fullmatch(r"R\d+", m)]
        bogus = [m for m in cited if m not in valid]
        if bogus:
            it["mapped"] = [m for m in it["mapped"] if m not in bogus]
            if it["status"] == PRESERVED and not it["mapped"]:
                it["status"] = "MISSING"
                it["evidence"] = f"cites requirement ids that do not exist in the specification: {', '.join(bogus)}"
        out.append(it)
    return out


def _merge(items: list[dict], structural: list[dict]) -> list[dict]:
    """Fold structural failures into the auditor's list: a finding downgrades the auditor item it concerns (so one
    defect is never counted twice); a finding concerning something the auditor never listed is added as an item."""
    merged = [dict(i) for i in items]
    for f in structural:
        key = f["user_requirement"]
        needles = [key.lower(), key.split("(")[0].lower() + "("]  # e.g. "put(key, ...)" -> also match "put("
        if key.startswith("numeric limit"):
            needles = [key.split()[-1]]
        elif key.startswith("thread"):
            needles = ["thread", "concurren"]
        elif key.startswith("LRU"):
            needles = ["lru", "least recently", "least-recently"]
        elif key.startswith("FIFO"):
            needles = ["fifo", "first in", "first-in"]
        hit = next((m for m in merged if any(n and n in m["user_requirement"].lower() for n in needles)), None)
        if hit is None:
            merged.append(f)
            continue
        if hit["status"] == PRESERVED:
            hit["status"], hit["source"] = "MISSING", "structural check overrode auditor"
        hit["evidence"] = (hit["evidence"] + " | " if hit["evidence"] and hit["status"] != PRESERVED else "") + f["evidence"]
    return merged


def check_preservation(llm: LLM, request: str, spec_text: str, claims: list[dict] | None = None) -> Preservation:
    """Both layers must agree. Unusable audit output (after one retry) fails closed."""
    structural = deterministic_checks(request, spec_text)
    items: list[dict] = []
    err = ""
    for _ in range(2):  # one retry for flaky/malformed auditor output
        try:
            items = _parse_audit(llm.complete(SYS_PRESERVATION, _audit_prompt(request, spec_text, claims or [])))
            err = ""
            break
        except Exception as e:  # malformed JSON, CLI/API failure, etc.
            err = f"{type(e).__name__}: {e}"
    if err:
        return Preservation(False, structural, audit_ok=False, audit_error=err)
    items = _validate_ids(items, spec_text)
    items = _merge(items, structural) + _coverage_gaps(request, items)
    return Preservation(all(i["status"] == PRESERVED for i in items), items)
