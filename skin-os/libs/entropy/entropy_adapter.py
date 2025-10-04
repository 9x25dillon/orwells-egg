"""
Adapter that wraps the user's entropy_engine.py and exposes
process(), stats(), graph() for the API.

Drop your repo's entropy_engine.py into libs/entropy/ or install it on PYTHONPATH.
This adapter auto-builds a small sample graph if none is provided.
"""
from __future__ import annotations
from typing import Tuple, Any

try:
    # Prefer local copy placed in libs/entropy/
    from entropy_engine import Token, EntropyNode, EntropyEngine  # type: ignore
except Exception:
    Token = None
    EntropyNode = None
    EntropyEngine = None


class EntropyHarness:
    def __init__(self):
        self.engine = None
        if EntropyEngine and EntropyNode:
            self.engine = EntropyEngine(self._build_graph(), max_depth=3)

    def _build_graph(self):
        # mirrors example usage: reverse -> add_char & multiply -> duplicate
        def reverse_string(v, ent):
            return str(v)[::-1]

        def add_random_char(v, ent):
            import random
            return str(v) + random.choice("abcdefghijklmnopqrstuvwxyz")

        def multiply_by_entropy(v, ent):
            try:
                return float(v) * ent
            except Exception:
                return f"{v}*{ent:.2f}"

        def duplicate_string(v, ent):
            k = max(1, int(ent * 2))
            return str(v) * k

        root = EntropyNode("reverse", reverse_string, entropy_limit=8.0)
        addc = EntropyNode("add_char", add_random_char, entropy_limit=9.0)
        mult = EntropyNode("multiply", multiply_by_entropy, entropy_limit=7.0)
        dupl = EntropyNode("duplicate", duplicate_string, entropy_limit=6.0)
        root.add_child(addc)
        root.add_child(mult)
        mult.add_child(dupl)
        return root

    def process(self, content: str) -> Tuple[float, float]:
        if not self.engine:
            # pseudo-entropy when engine absent
            import hashlib
            h = hashlib.sha256(content.encode()).hexdigest()
            ent = sum(int(c, 16) for c in h) / len(h)
            return (ent, ent)
        tok = Token(content)
        before = tok.entropy
        self.engine.run(tok)
        after = tok.entropy
        return (before, after)

    def stats(self) -> Any:
        if not self.engine:
            return {"error": "entropy_engine.py not found"}
        try:
            return self.engine.entropy_stats()
        except Exception:
            return {"note": "stats method unavailable in this revision"}

    def graph(self) -> Any:
        if not self.engine:
            return {"error": "entropy_engine.py not found"}
        try:
            return self.engine.export_graph()
        except Exception:
            return {"note": "graph export unavailable in this revision"}
