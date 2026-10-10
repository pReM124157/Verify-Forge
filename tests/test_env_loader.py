import os
from pathlib import Path

import pytest

from verifyforge import cli


def test_loads_keys_without_overriding_and_ignores_comments_quotes_and_export(tmp_path: Path, monkeypatch):
    f = tmp_path / ".env"
    f.write_text("# a comment\n\nANTHROPIC_API_KEY=fake-key-value-123\nexport OTHER='quoted value'\nQ=\"dq\"\nBROKEN LINE\nALREADY=from-file\n")
    for k in ("ANTHROPIC_API_KEY", "OTHER", "Q"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("ALREADY", "from-shell")
    loaded = cli.load_env_file(f)
    assert sorted(loaded) == ["ANTHROPIC_API_KEY", "OTHER", "Q"]
    assert os.environ["ANTHROPIC_API_KEY"] == "fake-key-value-123" and os.environ["OTHER"] == "quoted value" and os.environ["Q"] == "dq"
    assert os.environ["ALREADY"] == "from-shell"  # a variable you already set always wins


def test_missing_file_is_a_no_op(tmp_path: Path):
    assert cli.load_env_file(tmp_path / "nope.env") == []


def test_api_provider_reads_dotenv_from_the_current_directory(tmp_path: Path, monkeypatch):
    (tmp_path / ".env").write_text("ANTHROPIC_API_KEY=fake-key-from-dotenv\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    seen = {}
    monkeypatch.setattr(cli, "AnthropicLLM", lambda model=None, **kw: seen.setdefault("key", os.environ.get("ANTHROPIC_API_KEY")) or object())
    cli._make_llm("api", None)
    assert seen["key"] == "fake-key-from-dotenv"


def test_api_provider_without_any_key_is_a_clear_error_not_a_traceback(tmp_path: Path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)  # no .env here
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(cli, "load_env_file", lambda *a, **k: [])  # also ignore the project-root .env
    rc = cli.main(["demo", "rate-limiter", "--provider", "api", "--out", str(tmp_path / "o")])
    assert rc == 2 and "ANTHROPIC_API_KEY is not set" in capsys.readouterr().err


def test_default_cli_provider_never_touches_dotenv(tmp_path: Path, monkeypatch):
    called = []
    monkeypatch.setattr(cli, "load_env_file", lambda *a, **k: called.append(1) or [])
    cli._make_llm("cli", None)
    assert called == []
