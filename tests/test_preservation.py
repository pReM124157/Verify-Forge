"""Regression tests for the requirement-preservation gate.

The first test group reproduces the exact live failure: the user asked for put(key, value, ttl_seconds), capacity 100,
LRU eviction and thread safety, and the Architect silently produced TTLCache(ttl) / set(key, value) / thread safety.
"""
import io
import json
from pathlib import Path

import pytest
from rich.console import Console

from verifyforge.demos import RATE_LIMITER_REQUEST, rate_limiter_replay
from verifyforge.llm import SYS_ARCHITECT, SYS_BUILDER, SYS_PRESERVATION, SYS_REPAIR, SYS_TRIAGE, SYS_VERIFIER
from verifyforge.orchestrator import run_request
from verifyforge.preservation import (REASON_AUDIT_UNAVAILABLE, REASON_NOT_PRESERVED, check_preservation,
                                      deterministic_checks)
from verifyforge.report import Report, Round
from verifyforge.spec import ARCHITECT_PROMPT, architect_full
from verifyforge.ui import VerifyForgeUI

USER_TTL = """Build a thread-safe in-memory TTL cache in Python.

Requirements:
- Support get(key) and put(key, value, ttl_seconds)
- Expired entries must never be returned
- Maximum capacity is 100 items
- When capacity is exceeded, evict the least recently used item
- Multiple threads may call get and put at the same time
- The implementation must never corrupt internal state under concurrency
- Keep the public API small and do not add unnecessary features"""

# ---- a correct TTL-cache contract, assembled from parts so single defects can be introduced --------------------

TIER1 = '''import pytest
from ttl_cache import TTLCache


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def make():
    c = Clock()
    return c, TTLCache(clock=c)


def test_R1_put_then_get():
    c, cache = make()
    cache.put("a", 1, 10)
    assert cache.get("a") == 1 and cache.get("missing") is None


def test_R3_expired_never_returned():
    c, cache = make()
    cache.put("a", 1, 10)
    c.t = 10.0
    assert cache.get("a") is None


def test_R4_R5_capacity_100_evicts_least_recently_used():
    c, cache = make()
    for i in range(100):
        cache.put(f"k{i}", i, 1000)
    assert cache.get("k0") == 0          # k0 is now the most recently used
    cache.put("new", 1, 1000)            # exceeds capacity: k1 is the least recently used
    assert cache.get("k1") is None and cache.get("k0") == 0 and cache.get("new") == 1
'''

IMPL = '''import threading
import time
from collections import OrderedDict


class TTLCache:
    CAPACITY = 100

    def __init__(self, clock=time.monotonic):
        self._clock = clock
        self._data = OrderedDict()
        self._lock = threading.Lock()

    def put(self, key, value, ttl_seconds):
        with self._lock:
            now = self._clock()
            self._data.pop(key, None)
            self._data[key] = (value, now + ttl_seconds)
            while len(self._data) > self.CAPACITY:
                self._data.popitem(last=False)

    def get(self, key):
        with self._lock:
            item = self._data.get(key)
            if item is None:
                return None
            value, expires = item
            if self._clock() >= expires:
                del self._data[key]
                return None
            self._data.move_to_end(key)
            return value
'''

ADV = '''import threading
from ttl_cache import TTLCache


def test_ADV_concurrent_puts_never_exceed_capacity():
    cache = TTLCache()
    ts = [threading.Thread(target=cache.put, args=(f"k{i}", i, 60)) for i in range(300)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert sum(cache.get(f"k{i}") is not None for i in range(300)) <= 100
'''

SPEC_PARTS = {
    "intro": "Module `ttl_cache` exposes `class TTLCache` with constructor `TTLCache(clock=time.monotonic)`.",
    "get": "`get(key)` returns the stored value, or None if the key is absent or its entry has expired.",
    "put": "`put(key, value, ttl_seconds)` stores the value; it expires ttl_seconds after the call.",
    "cap": "The cache holds at most 100 items.",
    "lru": "When a put would exceed capacity, the least recently used (LRU) item is evicted; get and put both count as use.",
    "thread": "All methods are thread-safe: a single lock makes every get and put atomic.",
}
REQ_PARTS = {
    "get": "get(key) returns the stored value, or None when the key is absent.",
    "put": "put(key, value, ttl_seconds) stores value with its own time-to-live in seconds.",
    "expiry": "An expired entry is never returned by get.",
    "cap": "Maximum capacity is 100 items.",
    "lru": "When capacity is exceeded, the least recently used (LRU) item is evicted.",
    "thread": "get and put are safe to call from multiple threads at the same time (thread-safe, lock-protected).",
    "state": "Internal state is never corrupted under concurrent get/put.",
}


def architect_json(omit=(), edit=None):
    """A valid Architect reply for the TTL request. `omit` drops parts; `edit` maps old->new text everywhere."""
    spec = " ".join(v for k, v in SPEC_PARTS.items() if k not in omit)
    reqs = [v for k, v in REQ_PARTS.items() if k not in omit]
    for old, new in (edit or {}).items():
        spec, reqs = spec.replace(old, new), [r.replace(old, new) for r in reqs]
    return json.dumps({
        "title": "Thread-safe TTL cache", "module": "ttl_cache", "specification": spec, "requirements": reqs,
        "assumptions": [], "tier1_tests": TIER1,
        "requirement_traceability": [  # the Architect's own (untrusted) claims: always 'preserved'
            {"user_requirement": "get(key) and put(key, value, ttl_seconds)", "mapped_requirements": ["R1", "R2"],
             "status": "preserved"},
            {"user_requirement": "maximum capacity 100", "mapped_requirements": ["R4"], "status": "preserved"}]})


# The exact malformed contract from the live failure.
BAD_JSON = json.dumps({
    "title": "TTL cache", "module": "ttl_cache",
    "specification": "Module `ttl_cache` exposes `class TTLCache(ttl)` with `set(key, value)` and `get(key)`; every entry "
                     "shares the constructor ttl. Operations are thread safety protected.",
    "requirements": ["TTLCache(ttl) is constructed with one global time-to-live.",
                     "get(key) returns the value passed to set(key, value).",
                     "Expired entries are not returned.", "concurrency safe"],
    "assumptions": [], "tier1_tests": TIER1,
    "requirement_traceability": [{"user_requirement": "get and put", "mapped_requirements": ["R1", "R2"],
                                  "status": "preserved"}]})

GOOD_JSON = architect_json()


def audit_reply(status="PRESERVED", mapped=("SPEC",), items=None):
    items = items or ["get(key)", "put(key, value, ttl_seconds)", "capacity 100", "LRU eviction",
                      "expired entries never returned", "thread safety"]
    return json.dumps({"requirements": [{"user_requirement": u, "mapped": list(mapped), "status": status,
                                         "evidence": "stated in the spec"} for u in items]})


class Scripted:
    """Role-routed LLM double that records every call."""

    def __init__(self, architects, auditor=None):
        self.architects = list(architects)
        self.auditor = auditor or (lambda prompt: audit_reply())
        self.calls = []  # (system, prompt)

    @property
    def roles(self):
        return [s for s, _ in self.calls]

    def prompts(self, system):
        return [p for s, p in self.calls if s == system]

    def complete(self, system, prompt):
        self.calls.append((system, prompt))
        if system == SYS_ARCHITECT:
            return self.architects.pop(0) if len(self.architects) > 1 else self.architects[0]
        if system == SYS_PRESERVATION:
            out = self.auditor(prompt)
            if isinstance(out, BaseException):
                raise out
            return out
        if system == SYS_BUILDER:
            return f"```python\n{IMPL}```"
        if system == SYS_VERIFIER:
            return f"```python\n{ADV}```"
        raise AssertionError(f"unexpected model role called: {system[:40]}")


def spec_of(architect_reply):
    return architect_full(Scripted([architect_reply]), USER_TTL).spec_md


# ---- 1. the exact reproduction ----------------------------------------------------------------------------------

def test_exact_ttl_failure_is_rejected_before_builder_even_with_a_lazy_auditor(tmp_path: Path):
    llm = Scripted([BAD_JSON], auditor=lambda p: audit_reply())  # worst case: the auditor rubber-stamps everything
    events = []
    r = run_request(llm, USER_TTL, tmp_path, on_event=lambda n, d: events.append(n))
    assert r.status == "UNVERIFIED" and r.unverified_reason == REASON_NOT_PRESERVED
    assert SYS_BUILDER not in llm.roles and SYS_VERIFIER not in llm.roles  # no code, no hidden tests
    assert "BUILD_STARTED" not in events and "SPEC_GENERATED" not in events and "TIER1_RESULT" not in events
    assert llm.roles.count(SYS_ARCHITECT) == 2  # exactly one regeneration attempt
    missing = " | ".join(m for rej in r.preservation["rejected"] for m in rej["missing"])
    assert "put(key, value, ttl_seconds)" in missing and "capacity 100" in missing and "LRU" in missing
    assert "the number 100 does not appear" in missing  # the structural evidence is attached to the auditor's item
    assert not (tmp_path / "solution_v1.py").exists() and (tmp_path / "specification_rejected_1.md").exists()
    md = (tmp_path / "verification_report.md").read_text()
    assert "Status: UNVERIFIED" in md and REASON_NOT_PRESERVED in md and "Specification rejected (attempt 1)" in md
    assert "| put(key, value, ttl_seconds) |" in md and "MISSING" in md


def test_architect_self_claims_of_preserved_are_ignored(tmp_path: Path):
    # BAD_JSON claims everything is "preserved"; the independent checks must not accept the claim.
    assert '"status": "preserved"' in BAD_JSON
    r = run_request(Scripted([BAD_JSON]), USER_TTL, tmp_path)
    assert r.status == "UNVERIFIED"


# ---- 2. independent defects (structural layer only: the auditor is lazy and says PRESERVED) -----------------------

DEFECTS = {
    "missing_numeric_limit": (dict(omit=("cap",), edit={"100": "N"}), "the number 100 does not appear"),
    "changed_method_name": (dict(edit={"put(": "set("}), "put(key, value, ttl_seconds)"),
    "removed_method_argument": (dict(edit={"put(key, value, ttl_seconds)": "put(key, value)"}),
                                "put(key, value, ttl_seconds)"),
    "per_operation_ttl_moved_to_constructor": (
        dict(edit={"put(key, value, ttl_seconds)": "put(key, value)",
                   "TTLCache(clock=time.monotonic)": "TTLCache(ttl_seconds, clock=time.monotonic)"}),
        "put(key, value, ttl_seconds)"),
    "missing_lru_policy": (dict(omit=("lru",)), "never states least-recently-used"),
    "missing_concurrency_requirement": (dict(omit=("thread", "state")), "never mentions threads"),
}


@pytest.mark.parametrize("name", list(DEFECTS))
def test_each_defect_fails_preservation(name):
    kw, expected = DEFECTS[name]
    p = check_preservation(Scripted([]), USER_TTL, spec_of(architect_json(**kw)))
    assert not p.passed
    assert any(expected in m for m in p.missing), p.missing
    assert p.preserved < p.total


def test_all_requirements_correctly_represented_passes_and_maps_every_requirement():
    p = check_preservation(Scripted([]), USER_TTL, spec_of(GOOD_JSON))
    assert p.passed and p.preserved == p.total == 6 and p.missing == []
    assert deterministic_checks(USER_TTL, spec_of(GOOD_JSON)) == []


# ---- 3. the LLM auditor layer (defects the structural layer cannot see) ---------------------------------------------

@pytest.mark.parametrize("status", ["CHANGED", "WEAKENED", "MISSING", "nonsense"])
def test_auditor_verdict_alone_can_reject(status):
    reply = json.dumps({"requirements": [
        {"user_requirement": "get(key)", "mapped": ["R1"], "status": "PRESERVED", "evidence": "ok"},
        {"user_requirement": "must be thread-safe", "mapped": ["R6"], "status": status, "evidence": "now says SHOULD"}]})
    p = check_preservation(Scripted([], auditor=lambda _: reply), USER_TTL, spec_of(GOOD_JSON))
    assert not p.passed and any("thread-safe" in m for m in p.missing)


@pytest.mark.parametrize("bad", ["not json", "{}", '{"requirements": []}', '{"requirements": [{"status": "PRESERVED"}]}',
                                 RuntimeError("usage limit")])
def test_unusable_audit_fails_closed_and_does_not_regenerate_or_build(tmp_path: Path, bad):
    llm = Scripted([GOOD_JSON], auditor=lambda _: bad)
    r = run_request(llm, USER_TTL, tmp_path)
    assert r.status == "UNVERIFIED" and r.unverified_reason == REASON_AUDIT_UNAVAILABLE
    assert llm.roles.count(SYS_ARCHITECT) == 1 and SYS_BUILDER not in llm.roles
    assert llm.roles.count(SYS_PRESERVATION) == 2  # one retry of the audit itself


def test_auditor_never_sees_implementation_or_tests_only_request_spec_and_claims():
    llm = Scripted([GOOD_JSON])
    arch = architect_full(llm, USER_TTL)
    check_preservation(llm, USER_TTL, arch.spec_md, arch.traceability)
    p = llm.prompts(SYS_PRESERVATION)[0]
    assert USER_TTL in p and arch.spec_md in p and "def test_R1" not in p and "OrderedDict" not in p


# ---- 4. pipeline behaviour: regeneration, ordering, request untouched ------------------------------------------------

def smart_auditor(prompt):
    """Behaves like a competent auditor: flags the set()/ctor-TTL contract, passes the faithful one."""
    if "set(key, value)" in prompt:
        return json.dumps({"requirements": [
            {"user_requirement": "get(key)", "mapped": ["R2"], "status": "PRESERVED", "evidence": "R2"},
            {"user_requirement": "put(key, value, ttl_seconds)", "mapped": [], "status": "CHANGED",
             "evidence": "spec has set(key, value) with a constructor ttl"},
            {"user_requirement": "capacity 100", "mapped": [], "status": "MISSING", "evidence": "no capacity"},
            {"user_requirement": "LRU eviction", "mapped": [], "status": "MISSING", "evidence": "no eviction"}]})
    return audit_reply()


def test_regeneration_repairs_a_rejected_specification_and_the_run_verifies(tmp_path: Path):
    llm = Scripted([BAD_JSON, GOOD_JSON], auditor=smart_auditor)
    events = []
    r = run_request(llm, USER_TTL, tmp_path, on_event=lambda n, d: events.append((n, d)))
    names = [n for n, _ in events]
    assert r.status == "VERIFIED" and r.preservation["attempts"] == 2 and len(r.preservation["rejected"]) == 1
    assert names.index("SPEC_REJECTED") < names.index("SPEC_REGENERATING") < names.index("PRESERVATION_PASSED") \
        < names.index("BUILD_STARTED")
    fb = llm.prompts(SYS_ARCHITECT)[1]
    assert "PREVIOUS SPECIFICATION REJECTED" in fb and "put(key, value, ttl_seconds)" in fb and "capacity 100" in fb
    assert (tmp_path / "specification_rejected_1.md").exists() and (tmp_path / "specification.md").exists()
    assert all(t["status"] == "PRESERVED" for t in r.traceability)
    rejected_evt = next(d for n, d in events if n == "SPEC_REJECTED")
    assert rejected_evt["preserved"] == 1 and rejected_evt["total"] == 4


def test_original_user_request_is_never_modified(tmp_path: Path):
    llm = Scripted([BAD_JSON, GOOD_JSON], auditor=smart_auditor)
    r = run_request(llm, USER_TTL, tmp_path)
    first, second = llm.prompts(SYS_ARCHITECT)
    assert first == ARCHITECT_PROMPT + USER_TTL  # byte-for-byte on the first attempt
    assert second.startswith(ARCHITECT_PROMPT + USER_TTL + "\n\n")  # feedback is appended AFTER the untouched request
    assert r.request == USER_TTL and json.loads((tmp_path / "verification_report.json").read_text())["request"] == USER_TTL
    assert all(USER_TTL in p for p in llm.prompts(SYS_PRESERVATION))


def test_good_spec_run_verifies_with_full_traceability_table(tmp_path: Path):
    llm = Scripted([GOOD_JSON])
    r = run_request(llm, USER_TTL, tmp_path)
    assert r.status == "VERIFIED" and r.preservation["attempts"] == 1 and r.preservation["rejected"] == []
    assert len(r.traceability) == 6 and all(t["status"] == "PRESERVED" for t in r.traceability)
    md = (tmp_path / "verification_report.md").read_text()
    assert "## User requirement traceability" in md and "6 / 6 user requirements preserved" in md
    assert "| put(key, value, ttl_seconds) | SPEC | PRESERVED |" in md and "| capacity 100 |" in md
    assert SYS_BUILDER in llm.roles


# ---- 5. VERIFIED is impossible without full preservation --------------------------------------------------------------

def test_verified_is_forbidden_when_any_user_requirement_is_not_preserved():
    passing = [Round(1, True, True, "", tests_run=3)]
    ok = [{"user_requirement": "a", "mapped": ["R1"], "status": "PRESERVED", "evidence": ""}]
    bad = ok + [{"user_requirement": "b", "mapped": [], "status": "MISSING", "evidence": "gone"}]
    assert Report("t", "m", [], rounds=list(passing), traceability=ok).status == "VERIFIED"
    assert Report("t", "m", [], rounds=list(passing), traceability=bad).status == "UNVERIFIED"
    assert Report("t", "m", [], rounds=list(passing), unverified_reason=REASON_NOT_PRESERVED).status == "UNVERIFIED"
    assert Report("t", "m", [], rounds=list(passing)).status == "VERIFIED"  # spec-file runs carry no user traceability


# ---- 6. the offline demo behaves identically -------------------------------------------------------------------------

def test_offline_rate_limiter_demo_unchanged_and_fully_preserved(tmp_path: Path):
    r = run_request(rate_limiter_replay(), RATE_LIMITER_REQUEST, tmp_path)
    assert (r.status, r.repairs) == ("VERIFIED", 1)
    assert [(x.tier1_passed, x.adversarial_passed, x.tests_run) for x in r.rounds] == [(True, False, 7), (True, True, 7)]
    assert r.preservation["attempts"] == 1 and r.preservation["rejected"] == []
    assert len(r.traceability) == 8 and all(t["status"] == "PRESERVED" for t in r.traceability)
    assert "observed 100" in r.rounds[0].output  # the real concurrency failure is still real


# ---- 7. UI -----------------------------------------------------------------------------------------------------------

def frame(ui):
    c = Console(file=io.StringIO(), force_terminal=True, width=132, height=34, color_system=None)
    c.print(ui.render())
    return c.file.getvalue()


def test_ui_shows_rejection_then_preserved_count_and_never_started_builder(tmp_path: Path):
    ui = VerifyForgeUI("LIVE · CLAUDE CLI", USER_TTL, 3, 0, Console(file=io.StringIO(), width=132), "x")
    seen = []

    def hook(n, d):
        ui(n, d)
        if n in ("SPEC_REJECTED", "PRESERVATION_PASSED"):
            seen.append((n, frame(ui)))

    run_request(Scripted([BAD_JSON, GOOD_JSON], auditor=smart_auditor), USER_TTL, tmp_path, on_event=hook)
    rejected, passed = seen[0][1], seen[1][1]
    assert "SPEC REJECTED" in rejected and "1 / 4 user requirements preserved" in rejected
    assert "6 / 6 user requirements preserved" in passed

    ui2 = VerifyForgeUI("LIVE · CLAUDE CLI", USER_TTL, 3, 0, Console(file=io.StringIO(), width=132), "x")
    run_request(Scripted([BAD_JSON]), USER_TTL, tmp_path / "rej", on_event=ui2)
    out = Console(file=io.StringIO(), force_terminal=True, width=100, color_system=None)
    out.print(ui2.takeover_panel())
    shown = out.file.getvalue()
    assert "U N V E R I F I E D" in shown and "NEVER STARTED" in shown and "Software earned" not in shown
    assert ui2.ui_errors == []


def test_a_title_alone_cannot_satisfy_the_concurrency_requirement():
    # The title says "Thread-safe", but the specification body dropped the requirement.
    spec = spec_of(architect_json(omit=("thread", "state")))
    assert "Thread-safe" in spec.splitlines()[0]
    assert any(i["user_requirement"].startswith("thread") for i in deterministic_checks(USER_TTL, spec))


def test_one_defect_is_counted_once_not_twice():
    lazy = check_preservation(Scripted([]), USER_TTL, spec_of(architect_json(omit=("lru",))))
    assert lazy.total == 6 and lazy.preserved == 5  # the auditor's LRU item was downgraded, not duplicated
