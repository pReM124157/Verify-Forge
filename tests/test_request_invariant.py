from pathlib import Path

import pytest

from test_orchestrator import GOOD, FakeLLM
from verifyforge.orchestrator import run_request


class ForbiddenLLM:
    """Fails the test if any model call is attempted."""

    def __init__(self):
        self.calls = 0

    def complete(self, system, prompt):
        self.calls += 1
        raise AssertionError("the model must not be called for an empty request")


@pytest.mark.parametrize("bad", ["", "   ", "\t", "\n", " \t\n  "])
def test_empty_or_whitespace_request_is_rejected(tmp_path: Path, bad):
    with pytest.raises(ValueError, match="request must not be empty"):
        run_request(ForbiddenLLM(), bad, tmp_path / "out")


def test_non_string_request_is_rejected(tmp_path: Path):
    with pytest.raises(ValueError, match="request must not be empty"):
        run_request(ForbiddenLLM(), None, tmp_path / "out")


@pytest.mark.parametrize("bad", ["", "   "])
def test_rejection_makes_no_llm_call_no_events_and_no_run_directory(tmp_path: Path, bad):
    llm, events, out = ForbiddenLLM(), [], tmp_path / "out"
    with pytest.raises(ValueError):
        run_request(llm, bad, out, on_event=lambda n, d: events.append(n))
    assert llm.calls == 0  # no Architect / spec generation
    assert events == []  # no ARCHITECT_DONE, SPEC_GENERATED, ... and nothing for a UI to render
    assert not out.exists()  # no run directory, no artifacts
    assert list(tmp_path.iterdir()) == []


def test_valid_request_is_unchanged(tmp_path: Path):
    request = "  make an adder  "  # surrounding whitespace is the caller's business; it is passed through as-is
    llm, events = FakeLLM(GOOD), []
    r = run_request(llm, request, tmp_path, on_event=lambda n, d: events.append((n, d)))
    assert r.status == "VERIFIED" and r.request == request
    done = [d for n, d in events if n == "ARCHITECT_DONE"]
    assert done == [{"request": request}]  # same payload as before; the audit events now precede it
    assert any(request in p for p in llm.prompts)  # the Architect received exactly that request
    assert (tmp_path / "verification_report.md").exists()
