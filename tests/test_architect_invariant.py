import json

import pytest

from verifyforge.spec import ARCHITECT_PROMPT, architect

TIER1 = "from adder import add\n\ndef test_R1():\n    assert add(1, 2) == 3\n"


class Recorder:
    """Records every model call; returns a valid Architect reply."""

    def __init__(self):
        self.calls = []

    def complete(self, system, prompt):
        self.calls.append((system, prompt))
        return json.dumps({"title": "Add", "module": "adder", "specification": "add(a, b) returns the sum.",
                           "requirements": ["add returns the sum."], "tier1_tests": TIER1})


@pytest.mark.parametrize("bad", ["", " ", "   ", "\t", "\n", " \t\n  "])
def test_empty_and_whitespace_requests_rejected_without_llm_call(bad):
    llm = Recorder()
    with pytest.raises(ValueError, match="request must not be empty"):
        architect(llm, bad)
    assert llm.calls == []


@pytest.mark.parametrize("bad", [None, 0, 123, b"build x", ["build x"], {"r": "x"}])
def test_non_string_requests_rejected_without_llm_call(bad):
    llm = Recorder()
    with pytest.raises(ValueError, match="request must not be empty"):
        architect(llm, bad)
    assert llm.calls == []


def test_valid_request_reaches_the_llm_verbatim():
    llm = Recorder()
    request = "  Build an adder.\n  Two ints in, one int out.  "  # surrounding/inner whitespace must survive untouched
    spec_md, tier1 = architect(llm, request)
    assert len(llm.calls) == 1
    system, prompt = llm.calls[0]
    assert prompt == ARCHITECT_PROMPT + request  # nothing stripped, nothing added
    assert prompt.endswith(request)
    assert "Module: adder" in spec_md and "def test_R1" in tier1  # output handling unchanged


def test_plain_valid_request_unchanged():
    llm = Recorder()
    architect(llm, "make an adder")
    assert llm.calls[0][1] == ARCHITECT_PROMPT + "make an adder"
