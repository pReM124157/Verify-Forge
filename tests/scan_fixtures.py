"""Deterministic fixture repositories and a scripted model for Repository Scan tests (no network, no paid API)."""
import json
import re
from pathlib import Path

from verifyforge.llm import SYS_SCAN_CONTRACT, SYS_SCAN_REVIEW, SYS_TRIAGE, SYS_VERIFIER


def write(root: Path, rel: str, text: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)


def snapshot(root: Path) -> dict[str, bytes]:
    """Every file under root (hidden dirs included except .git) as bytes: proves a scan changed nothing."""
    return {str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file() and ".git/" not in str(p)}


# ---- fixture A: good simple package ----------------------------------------------------------------------------------
def calculator(root: Path) -> Path:
    write(root, "README.md", "# calc\n\nThe calculator adds, subtracts and divides numbers. Division by zero raises ZeroDivisionError.\n")
    write(root, "pyproject.toml", "[project]\nname='calc'\nversion='0.1'\n\n[tool.pytest.ini_options]\ntestpaths=['tests']\n")
    write(root, "src/calculator.py", '"""Calculator: add, subtract and divide numbers."""\n\n\ndef add(a, b):\n    """Return the sum of a and b."""\n    return a + b\n\n\n'
          'def divide(a, b):\n    """Divide a by b; division by zero raises ZeroDivisionError."""\n    return a / b\n')
    write(root, "tests/test_calculator.py", "from calculator import add, divide\n\n\ndef test_add():\n    assert add(1, 2) == 3\n\n\ndef test_divide():\n    assert divide(6, 3) == 2\n")
    return root


CALC_CONTRACT = {"requirements": [
    {"id": "IR1", "text": "divide(a, b) raises ZeroDivisionError when b is zero", "confidence": "high",
     "sources": [{"kind": "readme", "file": "README.md", "quote": "Division by zero raises ZeroDivisionError"},
                 {"kind": "docstring", "file": "src/calculator.py", "quote": "division by zero raises ZeroDivisionError"}]},
    {"id": "IR2", "text": "add(a, b) returns the sum of a and b", "confidence": "medium",
     "sources": [{"kind": "docstring", "file": "src/calculator.py", "quote": "Return the sum of a and b."}]}], "insufficient_reason": ""}
CALC_HIDDEN = ("import pytest\nfrom calculator import add, divide\n\n\ndef test_ADV_divide_by_zero():\n    with pytest.raises(ZeroDivisionError):\n        divide(1, 0)\n\n\n"
               "def test_ADV_add_sum():\n    assert add(2, 3) == 5 and add(-1, 1) == 0\n")


# ---- fixtures B/C: inventory, with and without a check-then-act race --------------------------------------------------
INV_BAD = '''"""Inventory: shared stock levels per product."""
import time


class Inventory:
    def __init__(self):
        self._stock = {}

    def add_stock(self, product_id, quantity):
        """Add quantity units of a product."""
        self._stock[product_id] = self._stock.get(product_id, 0) + quantity

    def reserve(self, product_id, quantity):
        """Reserve quantity units; return False when not enough stock is available."""
        if self._stock.get(product_id, 0) >= quantity:
            time.sleep(0)  # e.g. an audit-log write between the check and the update
            self._stock[product_id] = self._stock.get(product_id, 0) - quantity
            return True
        return False

    def available(self, product_id):
        """Units currently available."""
        return self._stock.get(product_id, 0)
'''
INV_GOOD = INV_BAD.replace("import time\n", "import threading\nimport time\n").replace(
    "        self._stock = {}\n", "        self._stock = {}\n        self._lock = threading.Lock()\n").replace(
    '''        if self._stock.get(product_id, 0) >= quantity:
            time.sleep(0)  # e.g. an audit-log write between the check and the update
            self._stock[product_id] = self._stock.get(product_id, 0) - quantity
            return True
        return False''', '''        with self._lock:
            if self._stock.get(product_id, 0) >= quantity:
                time.sleep(0)
                self._stock[product_id] = self._stock.get(product_id, 0) - quantity
                return True
            return False''')


def inventory(root: Path, good: bool = False) -> Path:
    write(root, "README.md", "# inventory\n\nInventory.reserve never oversells: available stock must never become negative, even with concurrent callers.\n")
    write(root, "src/inventory.py", INV_GOOD if good else INV_BAD)
    write(root, "tests/test_inventory.py", "from inventory import Inventory\n\n\ndef test_reserve_when_enough_stock():\n    inv = Inventory()\n    inv.add_stock('p', 5)\n"
          "    assert inv.reserve('p', 3)\n    assert not inv.reserve('p', 3)\n    assert inv.available('p') == 2\n")
    return root


INV_CONTRACT = {"requirements": [
    {"id": "IR1", "text": "reserve() must never allow available stock to become negative, even under concurrent callers", "confidence": "high",
     "sources": [{"kind": "readme", "file": "README.md", "quote": "available stock must never become negative"},
                 {"kind": "docstring", "file": "src/inventory.py", "quote": "return False when not enough stock is available"}]}], "insufficient_reason": ""}
INV_HIDDEN = '''import sys
import threading

from inventory import Inventory


def test_ADV_never_oversell():
    old = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        for _ in range(60):
            inv = Inventory()
            inv.add_stock("p", 8)
            wins = []

            barrier = threading.Barrier(32)

            def w():
                barrier.wait()
                wins.append(inv.reserve("p", 1))

            ts = [threading.Thread(target=w) for _ in range(32)]
            [t.start() for t in ts]
            [t.join() for t in ts]
            ok = sum(1 for x in wins if x)
            assert ok <= 8, f"oversold: {ok} reservations succeeded for 8 units"
    finally:
        sys.setswitchinterval(old)
'''


# ---- other fixtures ----------------------------------------------------------------------------------------------------
def no_tests(root: Path) -> Path:
    calculator(root)
    (root / "tests" / "test_calculator.py").unlink()
    (root / "tests").rmdir()
    return root


def broken_tests(root: Path) -> Path:
    calculator(root)
    write(root, "tests/test_calculator.py", "from calculator import add\n\n\ndef test_add_is_wrong():\n    assert add(1, 2) == 4\n")
    return root


def missing_dependency(root: Path) -> Path:
    write(root, "src/pipeline.py", '"""Pipeline using a third-party library."""\nimport not_installed_pkg_xyz\n\n\ndef run(x):\n    """Run x through the pipeline."""\n    return not_installed_pkg_xyz.go(x)\n')
    write(root, "tests/test_pipeline.py", "from pipeline import run\n\n\ndef test_run():\n    assert run(1)\n")
    return root


def no_python(root: Path) -> Path:
    write(root, "README.md", "# nothing here\n")
    write(root, "index.js", "console.log(1)\n")
    return root


def insufficient(root: Path) -> Path:
    write(root, "src/mystery.py", "def f(x, y):\n    return (x * 31 + y) % 97\n")
    return root


def threaded_and_plain(root: Path) -> Path:
    write(root, "src/plain.py", "def double(x):\n    return x * 2\n")
    write(root, "src/shared.py", "import threading\n\n_lock = threading.Lock()\n_state = {}\n\n\ndef put(k, v):\n    with _lock:\n        _state[k] = v\n")
    write(root, "src/money.py", "from decimal import Decimal\n\n\ndef total(prices):\n    return sum((Decimal(p) for p in prices), Decimal(0))\n")
    return root


FAKE_SECRET = "sk-" + "ant-api03-" + "FIXTURESECRET" * 3  # built at runtime: no key-shaped literal in the repository


def with_secrets(root: Path) -> Path:
    calculator(root)
    write(root, ".env", f"API_KEY={FAKE_SECRET}\n")
    write(root, "deploy.pem", "-----BEGIN PRIVATE KEY-----\nMIIFAKEFAKEFAKE\n-----END PRIVATE KEY-----\n")
    write(root, "config.py", f'API_KEY = "{FAKE_SECRET}"\n\n\ndef load():\n    """Load the configuration."""\n    return API_KEY\n')
    return root


# ---- the scripted model ------------------------------------------------------------------------------------------------------
class ScanFx:
    """Role-routed scan model. Records every call; can fail on demand."""

    def __init__(self, ranking, contracts, hidden, triage=None, replacement=None, fail=None):
        self.ranking, self.contracts, self.hidden = ranking, contracts, hidden
        self.triage, self.replacement, self.fail = triage, replacement, fail or {}
        self.calls: list[tuple[str, str]] = []

    def n(self, system):
        return sum(1 for s, _ in self.calls if s == system)

    def complete(self, system, prompt):
        self.calls.append((system, prompt))
        for sysname, exc in self.fail.items():
            if system == sysname and self.n(system) >= exc[1]:
                raise exc[0]
        if system == SYS_SCAN_REVIEW:
            return json.dumps({"ranked_modules": [{"path": p, "priority": pr, "reasons": ["shared state", "concurrency"],
                                                   "contract_confidence": "high"} for p, pr in self.ranking]})
        if system == SYS_SCAN_CONTRACT:
            path = re.search(r"module `([^`]+)`", prompt).group(1)
            return json.dumps(self.contracts.get(path, {"requirements": [], "insufficient_reason": "no evidence"}))
        if system == SYS_VERIFIER:
            if prompt.startswith("Replace an invalid test"):
                return f"```python\n{self.replacement}```"
            mod = re.search(r"for module `([^`]+)`", prompt).group(1)
            return f"```python\n{self.hidden[mod]}```"
        if system == SYS_TRIAGE:
            return self.triage or json.dumps({"verdict": "VALID", "reason": "the contract demands it", "spec_reference": "", "replacement_required": False})
        raise AssertionError(f"unexpected model role in a read-only scan: {system[:50]}")
