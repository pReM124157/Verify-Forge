"""Tier-1 triage: a Tier-1 test that keeps failing after a repair is audited (spec + test + helpers + assertion, never the
implementation) before more code is bent to it. INVALID -> quarantine + replace; VALID/AMBIGUOUS -> keeps blocking."""
import io
import json
from pathlib import Path

import pytest
from rich.console import Console

from test_preservation import ADV, IMPL, USER_TTL, architect_json, audit_reply
from test_triage import verdict
from verifyforge.llm import (SYS_ARCHITECT, SYS_BUILDER, SYS_PRESERVATION, SYS_REPAIR, SYS_TIER1, SYS_TRIAGE,
                             SYS_VERIFIER)
from verifyforge.orchestrator import run_request
from verifyforge.ui import VerifyForgeUI

SECRET = "# IMPL_SECRET_MARKER"

# The live failure: the helper reads keys with get(), which marks 0..99 as used IN ORDER, so key 0 becomes the least
# recently used; the test then wrongly expects key 0 to survive the next put.
BAD_TIER1 = '''from ttl_cache import TTLCache

CAP = 100


def _live_keys(cache, keys):
    return [k for k in keys if cache.get(k) is not None]


def test_R1_put_get():
    c = TTLCache()
    c.put("a", 1, 60)
    assert c.get("a") == 1


def test_R5_put_existing_key_refreshes_recency():
    cache = TTLCache()
    for i in range(CAP):
        cache.put(i, f"v{i}", 60)
    cache.put(0, "new", 60)
    assert len(_live_keys(cache, range(CAP))) == CAP
    cache.put(CAP, "x", 60)
    assert cache.get(0) == "new"
'''

REPLACEMENT = '''def test_R_replacement_1():
    from ttl_cache import TTLCache
    cache = TTLCache()
    for i in range(100):
        cache.put(i, f"v{i}", 60)
    cache.put(0, "new", 60)  # key 0 is now the most recently used; key 1 is the least recently used
    cache.put(100, "x", 60)  # exceeds capacity: evicts key 1
    assert cache.get(0) == "new" and cache.get(1) is None
'''


def reply_with(tier1):
    d = json.loads(architect_json())
    d["tier1_tests"] = tier1
    return json.dumps(d)


class T1LLM:
    def __init__(self, tier1=BAD_TIER1, triage_reply=None, replacement=REPLACEMENT, impl=IMPL, repair=None):
        self.tier1, self.impl = tier1, impl
        self.triage_reply = triage_reply or verdict("INVALID", "The helper uses get(), which touches keys 0..99 in "
                                                    "order, so key 0 is the least recently used and must be evicted.")
        self.replacement, self.repair = replacement, repair
        self.calls = []

    def roles(self):
        return [s for s, _ in self.calls]

    def prompts(self, role):
        return [p for s, p in self.calls if s == role]

    def complete(self, system, prompt):
        self.calls.append((system, prompt))
        if system == SYS_ARCHITECT:
            return reply_with(self.tier1)
        if system == SYS_PRESERVATION:
            return audit_reply()
        if system == SYS_BUILDER:
            return f"```python\n{self.impl}\n{SECRET}\n```"
        if system == SYS_REPAIR:
            return f"```python\n{self.repair or self.impl}\n{SECRET}\n```"
        if system == SYS_VERIFIER:
            return f"```python\n{ADV}```"
        if system == SYS_TRIAGE:
            r = self.triage_reply
            if isinstance(r, list):
                return r.pop(0)
            return r
        if system == SYS_TIER1:  # replacement authored by the Architect role
            assert prompt.startswith("Replace an invalid test")
            return f"```python\n{self.replacement}```"
        raise AssertionError(f"unexpected role {system[:40]}")


def test_exact_live_tier1_failure_is_triaged_quarantined_replaced_and_run_verifies(tmp_path: Path):
    llm = T1LLM()
    events = []
    r = run_request(llm, USER_TTL, tmp_path, on_event=lambda n, d: events.append((n, d)))
    assert r.status == "VERIFIED" and r.repairs == 1 and r.quarantined == 1
    t = r.triage[0]
    assert t["suite"] == "Tier-1" and t["status"] == "QUARANTINED" and t["verdict"] == "INVALID" and t["replacement"]
    assert t["test"].endswith("test_R5_put_existing_key_refreshes_recency")
    assert "test_R_replacement_1" in (tmp_path / "tier1_tests.py").read_text()
    # the rerun on the same code did not consume a repair; Tier-1 then passed and the hidden suite ran
    assert [(x.tier1_passed, x.adversarial_passed) for x in r.rounds] == [(False, None), (False, None), (True, True)]
    names = [n for n, _ in events]
    assert names.index("REPAIR_COMPLETE") < names.index("TRIAGE_STARTED") < names.index("ADVERSARIAL_GENERATING")
    assert next(d for n, d in events if n == "TRIAGE_STARTED")["suite"] == "tier1"
    md = (tmp_path / "verification_report.md").read_text()
    assert "Suite: Tier-1" in md and "Replacement test generated: YES" in md
    assert json.loads((tmp_path / "quarantine.json").read_text())[0]["suite"] == "Tier-1"


def test_tier1_triage_sees_spec_helper_exact_case_and_assertion_but_never_the_implementation(tmp_path: Path):
    llm = T1LLM()
    run_request(llm, USER_TTL, tmp_path)
    p = llm.prompts(SYS_TRIAGE)[0]
    assert "test_tier1.py::test_R5_put_existing_key_refreshes_recency" in p  # exact node id
    assert "def _live_keys(cache, keys)" in p and "cache.get(k)" in p  # the helper that explains the failure
    assert "put(key, value, ttl_seconds)" in p  # the specification
    assert "assert None == 'new'" in p and "get(0)" in p  # assertion evidence
    assert SECRET not in p and "OrderedDict" not in p and "_data" not in p  # no implementation


def test_no_triage_on_first_failure_only_after_a_repair_did_not_fix_it(tmp_path: Path):
    # A genuine Builder bug (evicts the MOST recently used item). The first Tier-1 failure goes to repair, not triage.
    wrong = IMPL.replace("self._data.popitem(last=False)", "self._data.popitem(last=True)")
    tier1 = ("from ttl_cache import TTLCache\n\n\ndef test_R5_lru_eviction():\n    c = TTLCache()\n"
             "    for i in range(100):\n        c.put(i, i, 60)\n    c.put(100, 100, 60)\n"
             "    assert c.get(0) is None and c.get(99) == 99\n")
    llm = T1LLM(tier1=tier1, impl=wrong, repair=IMPL)
    r = run_request(llm, USER_TTL, tmp_path)
    assert r.status == "VERIFIED" and r.repairs == 1 and r.triage == [] and llm.prompts(SYS_TRIAGE) == []


@pytest.mark.parametrize("reply", [verdict("VALID", "the spec demands it"), verdict("AMBIGUOUS", "spec is silent"),
                                   "not json at all", json.dumps({"verdict": "MAYBE"})])
def test_valid_ambiguous_or_unusable_verdicts_never_quarantine_a_tier1_test(tmp_path: Path, reply):
    llm = T1LLM(triage_reply=reply)
    r = run_request(llm, USER_TTL, tmp_path, max_repairs=2)
    assert r.status == "UNVERIFIED" and r.quarantined == 0 and r.triage[0]["status"].startswith("KEPT")
    assert len(llm.prompts(SYS_TRIAGE)) == 1  # each test is triaged once
    assert SYS_TIER1 not in llm.roles() and "test_R_replacement" not in (tmp_path / "tier1_tests.py").read_text()


def test_a_real_tier1_defect_stays_blocking_when_triage_says_valid(tmp_path: Path):
    wrong = IMPL.replace("self._data.popitem(last=False)", "self._data.popitem(last=True)")
    tier1 = BAD_TIER1.replace('assert cache.get(0) == "new"', "assert cache.get(1) is None")
    llm = T1LLM(tier1=tier1, impl=wrong, repair=wrong, triage_reply=verdict("VALID", "key 1 is the LRU and must go"))
    r = run_request(llm, USER_TTL, tmp_path, max_repairs=2)
    assert r.status == "UNVERIFIED" and r.quarantined == 0


def test_quarantine_cap_is_shared_and_enforced_for_tier1(tmp_path: Path):
    tests = "".join(f"def test_R{i}_bad():\n    assert 1 == 2\n\n" for i in range(5))
    tier1 = "from ttl_cache import TTLCache\n\n" + tests
    llm = T1LLM(tier1=tier1, triage_reply=verdict("INVALID", "1 != 2 is nonsense"),
                replacement="def test_R_replacement_1():\n    assert True\n")
    r = run_request(llm, USER_TTL, tmp_path, max_repairs=1)
    assert r.quarantined == 3 and len(llm.prompts(SYS_TRIAGE)) == 3  # MAX_QUARANTINES, no more audits
    assert r.status == "UNVERIFIED"  # the remaining bad tests still block


def test_tier1_replacement_cannot_redefine_helpers(tmp_path: Path):
    shadow = ("def _live_keys(cache, keys):\n    return []\n\n"
              "def test_R_replacement_1():\n    assert True\n")
    r = run_request(T1LLM(replacement=shadow), USER_TTL, tmp_path, max_repairs=2)
    assert r.triage[0]["status"] == "QUARANTINED" and r.triage[0]["replacement"] is False
    assert "return []" not in (tmp_path / "tier1_tests.py").read_text()


def test_ui_labels_tier1_triage(tmp_path: Path):
    ui = VerifyForgeUI("LIVE · CLAUDE CLI", USER_TTL, 3, 0, Console(file=io.StringIO(), width=132), "x")
    shown = []

    def hook(n, d):
        ui(n, d)
        if n == "TRIAGE_RESULT":
            c = Console(file=io.StringIO(), force_terminal=True, width=132, height=34, color_system=None)
            c.print(ui.render())
            shown.append(c.file.getvalue())

    run_request(T1LLM(), USER_TTL, tmp_path, on_event=hook)
    assert "TIER-1 TRIAGE" in shown[0] and "INVALID TEST" in shown[0] and ui.ui_errors == []
