"""Deterministic repository discovery. Pure AST + file reading: repository code is NEVER executed here."""
from __future__ import annotations

import ast
import re
from pathlib import Path

from .safety import is_secret_file, read_text_safe, walk_repo

CONFIG_NAMES = {"pyproject.toml", "setup.py", "setup.cfg", "tox.ini", "pytest.ini", "conftest.py", "Makefile", "Pipfile",
                "poetry.lock", "mypy.ini", ".flake8", "noxfile.py", "MANIFEST.in"}
DOC_PREFIXES = ("README", "CHANGELOG", "CONTRIBUTING", "ARCHITECTURE", "DESIGN", "API")

_ID_SPLIT = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+")


def _terms(name: str) -> set[str]:
    return {t.lower() for t in _ID_SPLIT.findall(name)}


TERM_SIGNALS = {
    "retry": {"retry", "retries", "backoff", "attempt", "attempts"},
    "auth": {"auth", "login", "logout", "password", "permission", "permissions", "oauth", "credential", "credentials", "session", "token", "acl", "role"},
    "crypto": {"encrypt", "decrypt", "cipher", "hash", "hmac", "signature", "sign", "digest", "salt", "crypto"},
    "state_machine": {"transition", "fsm", "machine", "workflow", "lifecycle"},
    "cache": {"cache", "memo", "memoize", "ttl", "lru", "expire", "expiry"},
    "rate_limit": {"throttle", "limiter", "ratelimit", "bucket", "quota"},
    "inventory_balance": {"inventory", "stock", "balance", "account", "ledger", "reserve", "reservation", "refund", "payment",
                          "wallet", "order", "invoice", "deposit", "withdraw", "withdrawal", "transfer", "charge", "credit", "debit"},
    "money": {"money", "currency", "price", "amount", "decimal", "cents", "fee", "tax", "total"},
    "validation": {"validate", "validator", "validation", "sanitize", "schema", "parse", "parser"},
    "queue": {"queue", "deque", "worker", "dispatch", "scheduler"},
}
IMPORT_SIGNALS = {
    "threading": {"threading", "concurrent", "multiprocessing", "_thread"},
    "asyncio": {"asyncio", "trio", "anyio"},
    "queue": {"queue"},
    "decimal": {"decimal", "fractions"},
    "network": {"socket", "requests", "httpx", "urllib", "urllib3", "aiohttp", "http", "ssl", "websockets", "websocket", "ftplib", "smtplib"},
    "database": {"sqlite3", "sqlalchemy", "psycopg", "psycopg2", "pymongo", "redis", "mysql", "peewee", "django", "asyncpg", "aiosqlite"},
    "subprocess": {"subprocess"},
    "crypto": {"hashlib", "hmac", "secrets", "cryptography", "jwt", "Crypto", "nacl", "bcrypt"},
    "time": {"time", "datetime", "sched"},
    "serialization": {"json", "pickle", "yaml", "marshal", "csv", "toml", "tomllib", "xml", "msgpack"},
    "retry": {"tenacity", "backoff", "retrying"},
    "filesystem": {"shutil", "tempfile", "glob", "pathlib"},
}
LOCK_NAMES = {"Lock", "RLock", "Semaphore", "BoundedSemaphore", "Condition", "Event", "Barrier"}
MUTABLE_CALLS = {"list", "dict", "set", "defaultdict", "OrderedDict", "deque", "Counter"}
WRITE_CALLS = {"write_text", "write_bytes", "remove", "unlink", "rename", "replace", "rmtree", "move", "copy", "copyfile", "makedirs", "mkdir", "touch"}


def _sig(node) -> str:
    try:
        args = ast.unparse(node.args)
    except Exception:
        args = "..."
    ret = f" -> {ast.unparse(node.returns)}" if getattr(node, "returns", None) is not None else ""
    return f"{node.name}({args}){ret}"


def _decorators(node) -> list[str]:
    out = []
    for d in getattr(node, "decorator_list", []):
        target = d.func if isinstance(d, ast.Call) else d
        out.append(ast.unparse(target))
    return out


def _doc(node) -> str:
    d = ast.get_docstring(node) or ""
    return d.strip().splitlines()[0][:200] if d.strip() else ""


def analyze_module(rel: str, text: str) -> dict:
    """AST-derived facts for one module. Never executes the code."""
    loc = sum(1 for l in text.splitlines() if l.strip() and not l.strip().startswith("#"))
    out = {"path": rel, "loc": loc, "docstring": "", "imports": [], "classes": [], "functions": [], "exceptions": [],
           "signals": {}, "parse_error": None}
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError) as e:
        out["parse_error"] = f"{type(e).__name__}: {e}"
        return out
    out["docstring"] = (ast.get_docstring(tree) or "").strip()[:600]
    imports: set[str] = set()
    names: set[str] = set()
    sig = {k: False for k in ("threading", "lock", "asyncio", "queue", "decimal", "money", "filesystem", "file_writes", "network",
                              "database", "subprocess", "retry", "time", "global_state", "crypto", "auth", "state_machine",
                              "cache", "rate_limit", "inventory_balance", "validation", "serialization", "config_mutation")}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".")[0])
            names |= {a.name for a in node.names}
        if isinstance(node, ast.AsyncFunctionDef):
            sig["asyncio"] = True
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
        if isinstance(node, ast.Call):
            fn = node.func
            fname = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
            if fname in WRITE_CALLS:
                sig["file_writes"] = True
            if fname == "open" and len(node.args) >= 2 and isinstance(node.args[1], ast.Constant) and set(str(node.args[1].value)) & set("wax+"):
                sig["file_writes"] = True
            if fname == "open":
                sig["filesystem"] = True
            if fname in ("system", "popen") and isinstance(fn, ast.Attribute) and getattr(fn.value, "id", "") == "os":
                sig["subprocess"] = True
            if fname in ("setattr", "putenv", "setdefault") or (isinstance(fn, ast.Attribute) and fname == "update"
                                                                  and getattr(fn.value, "attr", getattr(fn.value, "id", "")) in ("environ", "config", "settings")):
                sig["config_mutation"] = True
    for k, mods in IMPORT_SIGNALS.items():
        if imports & mods:
            sig[k] = True
    if names & LOCK_NAMES:
        sig["lock"] = True
        sig["threading"] = sig["threading"] or "asyncio" not in imports
    if "Decimal" in names:
        sig["decimal"] = True
    words = set().union(*[_terms(n) for n in names]) if names else set()
    words |= {w.lower() for w in re.findall(r"[A-Za-z]{4,}", out["docstring"])}
    for k, ts in TERM_SIGNALS.items():
        if words & ts:
            sig[k] = True
    if "decimal" in sig and sig["decimal"]:
        sig["money"] = True
    if {"lru_cache", "cache", "cached_property"} & names:
        sig["cache"] = True
    # module-level mutable state
    for node in tree.body:
        targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) else []
        val = getattr(node, "value", None)
        if not targets or val is None:
            continue
        tname = getattr(targets[0], "id", "")
        if tname in ("__all__", "__slots__") or tname.startswith("__"):
            continue
        if isinstance(val, (ast.List, ast.Dict, ast.Set, ast.ListComp, ast.DictComp, ast.SetComp)):
            sig["global_state"] = True
        elif isinstance(val, ast.Call) and getattr(val.func, "id", getattr(val.func, "attr", "")) in MUTABLE_CALLS:
            sig["global_state"] = True
    out["imports"] = sorted(imports)
    out["signals"] = {k: v for k, v in sig.items() if v}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out["functions"].append({"name": node.name, "public": not node.name.startswith("_"), "signature": _sig(node),
                                     "async": isinstance(node, ast.AsyncFunctionDef), "doc": _doc(node), "decorators": _decorators(node)})
        elif isinstance(node, ast.ClassDef):
            bases = [ast.unparse(b) for b in node.bases]
            if any(b.endswith(("Exception", "Error", "Warning")) for b in bases):
                out["exceptions"].append(node.name)
            methods = [{"name": m.name, "public": not m.name.startswith("_"), "signature": _sig(m),
                        "async": isinstance(m, ast.AsyncFunctionDef), "doc": _doc(m), "decorators": _decorators(m),
                        "returns": ast.unparse(m.returns) if m.returns is not None else ""}
                       for m in node.body if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))
                       and (not m.name.startswith("_") or m.name == "__init__")]
            out["classes"].append({"name": node.name, "public": not node.name.startswith("_"), "bases": bases, "doc": _doc(node),
                                   "methods": methods})
    return out


def is_test_path(rel: str) -> bool:
    p = Path(rel)
    parts = {x.lower() for x in p.parts[:-1]}
    n = p.name.lower()
    return n.startswith("test_") or n.endswith("_test.py") or n == "conftest.py" or bool(parts & {"tests", "test"})


def _import_root(files: list[str]) -> str:
    return "src" if any(f.startswith("src/") and f.endswith(".py") for f in files) else ""


def dotted_name(rel: str, import_root: str) -> str | None:
    p = Path(rel)
    if import_root and p.parts and p.parts[0] == import_root:
        p = Path(*p.parts[1:])
    parts = list(p.with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    if not parts or not all(x.isidentifier() for x in parts):
        return None
    return ".".join(parts)


def public_api_count(m: dict) -> int:
    return sum(f["public"] for f in m["functions"]) + sum(c["public"] for c in m["classes"])


def discover(root: Path) -> dict:
    """The structured repository map (JSON-serialisable)."""
    root = root.resolve()
    skipped = {"secret": [], "binary": 0, "large": 0, "symlink_outside": 0}
    py, tests, configs, docs, other = [], [], [], [], 0
    texts: dict[str, str] = {}
    for rel, abs_, kind in walk_repo(root):
        r = rel.as_posix()
        if kind == "secret":
            skipped["secret"].append(r)
            continue
        if kind == "binary":
            skipped["binary"] += 1
            continue
        if kind == "large":
            skipped["large"] += 1
            continue
        if kind == "symlink-outside":
            skipped["symlink_outside"] += 1
            continue
        name = rel.name
        if name.endswith(".py"):
            (tests if is_test_path(r) else py).append(r)
            t = read_text_safe(root, r)
            if t is not None:
                texts[r] = t
        elif name in CONFIG_NAMES or name.startswith("requirements") and name.endswith(".txt") or r.startswith(".github/"):
            configs.append(r)
        elif name.upper().startswith(DOC_PREFIXES) or (r.startswith("docs/") and name.endswith((".md", ".rst", ".txt"))):
            docs.append(r)
        else:
            other += 1
    # conftest.py is configuration, not a test file to count
    test_files = [t for t in tests if Path(t).name != "conftest.py"]
    all_py = py + tests
    import_root = _import_root(all_py)
    modules = []
    for rel in sorted(py):
        a = analyze_module(rel, texts.get(rel, ""))
        a["dotted"] = dotted_name(rel, import_root)
        a["public_api"] = public_api_count(a)
        modules.append(a)
    index = {m["dotted"]: m["path"] for m in modules if m["dotted"]}
    stems: dict[str, list[str]] = {}
    for m in modules:
        stems.setdefault(Path(m["path"]).stem, []).append(m["path"])
    # test discovery and module -> tests mapping
    test_info, mapping = [], {m["path"]: [] for m in modules}
    for rel in sorted(test_files):
        text = texts.get(rel, "")
        funcs, imported = [], set()
        try:
            tree = ast.parse(text)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported |= {a.name for a in node.names}
                elif isinstance(node, ast.ImportFrom):
                    base = ("." * node.level) + (node.module or "")
                    imported.add(base)
                    imported |= {f"{node.module}.{a.name}" for a in node.names if node.module}
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test"):
                    funcs.append(node.name)
                elif isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
                    funcs += [f"{node.name}.{m.name}" for m in node.body if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)) and m.name.startswith("test")]
        except (SyntaxError, ValueError):
            pass
        test_info.append({"path": rel, "tests": funcs, "count": len(funcs)})
        low = text.lower()
        for m in modules:
            via, conf = None, None
            d = m["dotted"]
            if d and any(i == d or i.startswith(d + ".") or i.endswith("." + d) or i.lstrip(".") == d for i in imported):
                via, conf = "import", "high"
            elif Path(rel).stem in (f"test_{Path(m['path']).stem}", f"{Path(m['path']).stem}_test"):
                via, conf = "name", "medium"
            elif len(Path(m["path"]).stem) > 3 and Path(m["path"]).stem in low:
                via, conf = "mention", "low"
            if via:
                mapping[m["path"]].append({"test_file": rel, "via": via, "confidence": conf})
    framework = None
    cfg_text = " ".join((read_text_safe(root, c, 20000) or "") for c in configs if c in ("pyproject.toml", "setup.cfg", "tox.ini", "pytest.ini"))
    if test_files:
        framework = "pytest" if ("pytest" in cfg_text or any("pytest" in texts.get(t, "") for t in test_files)
                                 or "conftest.py" in tests or any(Path(c).name == "pytest.ini" for c in configs)) else "unittest/pytest"
    packages = sorted({str(Path(m["path"]).parent) for m in modules if (root / Path(m["path"]).parent / "__init__.py").exists() and str(Path(m["path"]).parent) != "."})
    return {
        "root": str(root), "name": root.name, "language": "python" if (py or tests) else None,
        "python_files": len(py) + len(tests), "source_files": len(py), "test_files": len(test_files),
        "import_root": import_root or ".", "modules": modules, "packages": packages, "tests": test_info,
        "test_mapping": mapping, "test_framework": framework, "config_files": sorted(configs), "documentation_files": sorted(docs),
        "skipped": skipped, "secret_like_files": sorted(skipped["secret"]),
        "ignored_note": "ignored directories are never entered; symlinks are never followed out of the root",
    }


def api_map(repo_map: dict) -> dict:
    """module -> public API (AST-derived; underscore-prefixed names are treated as internal by convention only)."""
    out = {}
    for m in repo_map["modules"]:
        def disp(x):
            return f"{x['name']} (property)" if any(d in ("property", "cached_property") or d.endswith(".setter") for d in x.get("decorators", [])) else x["signature"]

        api = {"classes": [{"name": c["name"], "public": c["public"], "methods": [disp(x) for x in c["methods"] if x["public"] or x["name"] == "__init__"]}
                           for c in m["classes"]],
               "functions": [{"name": f["name"], "public": f["public"], "signature": f["signature"], "async": f["async"]} for f in m["functions"]]}
        if api["classes"] or api["functions"]:
            out[m["path"]] = api
    return out
