"""Path, secret and binary safety for repository scanning. Everything the scanner reads goes through here."""
from __future__ import annotations

import fnmatch
import os
import re
from pathlib import Path

IGNORED_DIRS = {
    ".git", ".hg", ".svn", ".venv", "venv", "env", ".env.d", "__pycache__", "node_modules", "dist", "build", "coverage",
    "htmlcov", "runs", ".mypy_cache", ".pytest_cache", ".ruff_cache", ".tox", ".nox", "generated", "vendor", "site-packages",
    ".idea", ".vscode", ".eggs",
}
IGNORED_DIR_GLOBS = ("*.egg-info", "*.dist-info")

# Never read or send these, whatever their content.
SECRET_FILE_GLOBS = (".env", ".env.*", "*.pem", "*.key", "*.p12", "*.pfx", "*.jks", "*.keystore", "id_rsa*", "id_ed25519*",
                     "id_ecdsa*", "*credential*", "*secret*", ".netrc", ".npmrc", ".pypirc", "*.kdbx", "*.gpg", "*.asc")
SECRET_FILE_ALLOW = (".env.example", ".env.sample", ".env.template")

MAX_FILE_BYTES = 1_000_000  # a source file larger than this is skipped (generated/minified/data)
MAX_READ_CHARS = 200_000

_SECRET_PATTERNS = [
    re.compile(r"sk-ant-[A-Za-z0-9_\-]{8,}"),
    re.compile(r"sk-[A-Za-z0-9]{20,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"),
    re.compile(r"xox[baprs]-[A-Za-z0-9\-]{10,}"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?(?:-----END [A-Z ]*PRIVATE KEY-----|$)"),
    re.compile(r"eyJ[A-Za-z0-9_\-]{20,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{5,}"),
    re.compile(r"(?i)\b(?:api[_-]?key|secret|token|passw(?:or)?d|auth)\w*\s*[:=]\s*['\"][^'\"\s]{8,}['\"]"),
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._\-]{20,}"),
]


def is_secret_file(name: str) -> bool:
    low = name.lower()
    if low in SECRET_FILE_ALLOW:
        return False
    return any(fnmatch.fnmatch(low, g) for g in SECRET_FILE_GLOBS)


def has_secret_like_value(text: str) -> bool:
    return any(p.search(text) for p in _SECRET_PATTERNS)


def redact_secrets(text: str) -> str:
    """Remove secret-shaped values before text can reach a model, a log or a report."""
    for p in _SECRET_PATTERNS:
        text = p.sub("[REDACTED]", text)
    return text


def is_ignored_dir(name: str) -> bool:
    return name in IGNORED_DIRS or any(fnmatch.fnmatch(name, g) for g in IGNORED_DIR_GLOBS)


def is_binary(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            return b"\x00" in f.read(8192)
    except OSError:
        return True


def resolve_inside(root: Path, rel: str | Path) -> Path | None:
    """The resolved path of `rel` under `root`, or None if it escapes the root (via .., absolute paths or symlinks)."""
    root = root.resolve()
    try:
        cand = (root / rel).resolve()
    except (OSError, RuntimeError):
        return None
    return cand if cand == root or root in cand.parents else None


def read_text_safe(root: Path, rel: str | Path, limit: int = MAX_READ_CHARS) -> str | None:
    """Text of a file inside the repository, or None if it is outside, secret-like, binary, too large or unreadable."""
    p = resolve_inside(root, rel)
    if p is None or not p.is_file() or is_secret_file(p.name) or is_binary(p):
        return None
    try:
        if p.stat().st_size > MAX_FILE_BYTES:
            return None
        return p.read_text(errors="replace")[:limit]
    except OSError:
        return None


def walk_repo(root: Path):
    """Yield (relative path, absolute path, kind) for every regular file inside the repository: kind is 'ok', 'secret',
    'binary', 'large' or 'symlink-outside'. Ignored directories are never entered; symlinks are never followed out."""
    root = root.resolve()
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = sorted(d for d in dirnames if not is_ignored_dir(d) and not (Path(dirpath) / d).is_symlink())
        for fn in sorted(filenames):
            p = Path(dirpath) / fn
            rel = p.relative_to(root)
            if p.is_symlink():
                target = resolve_inside(root, rel)
                if target is None or not target.is_file():
                    yield rel, p, "symlink-outside"
                    continue
            if is_secret_file(fn):
                yield rel, p, "secret"
                continue
            try:
                size = p.stat().st_size
            except OSError:
                continue
            if size > MAX_FILE_BYTES:
                yield rel, p, "large"
            elif is_binary(p):
                yield rel, p, "binary"
            else:
                yield rel, p, "ok"
