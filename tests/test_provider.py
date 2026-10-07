import json
import subprocess

import pytest

from verifyforge.llm import ClaudeCLI


def fake_run(stdout="", returncode=0, stderr="", capture=None):
    def _run(cmd, **kw):
        if capture is not None:
            capture.update(cmd=cmd, **kw)
        return subprocess.CompletedProcess(cmd, returncode, stdout, stderr)
    return _run


def test_cli_returns_result_and_is_locked_down(monkeypatch):
    seen = {}
    monkeypatch.setattr(subprocess, "run", fake_run(json.dumps({"result": "hi", "is_error": False}), capture=seen))
    assert ClaudeCLI().complete("SYS", "PROMPT") == "hi"
    cmd = seen["cmd"]
    assert cmd[cmd.index("--tools") + 1] == "" and "--no-session-persistence" in cmd
    assert "--bare" not in cmd  # --bare would disable OAuth login
    assert cmd[cmd.index("--system-prompt") + 1] == "SYS"
    assert "Verify-Forge" not in seen["cwd"]  # runs from a neutral temp dir


@pytest.mark.parametrize("stdout,rc", [("not json", 0), (json.dumps({"result": "x", "is_error": True}), 0),
                                       (json.dumps({"result": "x"}), 1), (json.dumps({"result": "  "}), 0)])
def test_cli_failures_raise(monkeypatch, stdout, rc):
    monkeypatch.setattr(subprocess, "run", fake_run(stdout, rc))
    with pytest.raises(RuntimeError):
        ClaudeCLI().complete("s", "p")


def test_cli_timeout_and_missing_binary(monkeypatch):
    def boom(cmd, **kw):
        raise subprocess.TimeoutExpired(cmd, 1)
    monkeypatch.setattr(subprocess, "run", boom)
    with pytest.raises(TimeoutError):
        ClaudeCLI().complete("s", "p")
    def missing(cmd, **kw):
        raise FileNotFoundError("claude")
    monkeypatch.setattr(subprocess, "run", missing)
    with pytest.raises(RuntimeError, match="not found"):
        ClaudeCLI().complete("s", "p")
