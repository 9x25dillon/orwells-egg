#!/usr/bin/env python3
"""
Enhanced DAG Orchestrator with Topological Sorting, Pluggable Backends,
Ghost Score Feedback, and Serialization
SPDX-License-Identifier: Apache-2.0
"""

from __future__ import annotations
import json
import time
import hashlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, Set, runtime_checkable
from collections import deque
import numpy as np

# ==============================
# Local Math Fallbacks
# ==============================
def entropy_local(data: List[List[float]]) -> Dict[str, float]:
    """Shannon entropy + simple 'surprise' and 'complexity' fallbacks."""
    if not data or not any(data):
        return {"shannon": 0.0, "surprise": 0.0, "complexity": 0.0}

    flat = np.array(data).flatten()
    flat = flat[np.isfinite(flat)]
    if len(flat) == 0:
        return {"shannon": 0.0, "surprise": 0.0, "complexity": 0.0}

    p = np.abs(flat) / (np.sum(np.abs(flat)) + 1e-12)
    p = p[p > 0]
    shannon = -np.sum(p * np.log2(p))
    log_p = np.log2(p + 1e-12)
    surprise = np.var(log_p)
    complexity = float(shannon * surprise)
    return {"shannon": float(shannon), "surprise": float(surprise), "complexity": complexity}

def chebyshev_local(matrix: List[List[float]], degree: int) -> List[List[float]]:
    """Chebyshev polynomial projection fallback, matrix-wise recurrence."""
    if not matrix or not any(matrix):
        return matrix
    M = np.array(matrix, dtype=np.float64)
    if degree == 0:
        return np.ones_like(M).tolist()
    elif degree == 1:
        return M.tolist()
    T_prev = np.ones_like(M)
    T_curr = M.copy()
    for _n in range(2, degree + 1):
        T_next = 2 * M * T_curr - T_prev
        T_prev, T_curr = T_curr, T_next
    return T_curr.tolist()

# ==============================
# Node + Protocol Definitions
# ==============================
@dataclass
class Node:
    name: str
    dependencies: List[str]
    func: Callable[..., Any]

@runtime_checkable
class BackendProtocol(Protocol):
    def analyze_entropy(self, data: List[List[float]]) -> Dict[str, float]: ...
    def project_chebyshev(self, matrix: List[List[float]], degree: int) -> List[List[float]]: ...
    def optimize(self, matrix: List[List[float]], method: str) -> Dict[str, Any]: ...
    def ghost_score(self, signature: Dict[str, Any]) -> float: ...

# ==============================
# Backend Implementations
# ==============================
class MockBackend(BackendProtocol):
    def analyze_entropy(self, data: List[List[float]]) -> Dict[str, float]:
        return entropy_local(data)

    def project_chebyshev(self, matrix: List[List[float]], degree: int) -> List[List[float]]:
        return chebyshev_local(matrix, degree)

    def optimize(self, matrix: List[List[float]], method: str = "gradient") -> Dict[str, Any]:
        M = np.array(matrix)
        if M.size == 0:
            return {"optimized": matrix, "loss": 0.0, "iterations": 0, "converged": True}
        optimized = M + np.random.normal(0, 0.01, M.shape)
        return {
            "optimized": optimized.tolist(),
            "loss": float(np.random.uniform(0.1, 0.5)),
            "iterations": int(np.random.randint(5, 20)),
            "converged": bool(np.random.random() > 0.2)
        }

    def ghost_score(self, signature: Dict[str, Any]) -> float:
        if "shape" in signature:
            size = max(1.0, float(np.prod(signature["shape"])))
            norm = float(signature.get("norm", 1.0))
            structure = min(1.0, norm / (size ** 0.5 + 1e-12))
            return float(structure * 0.3)
        return 0.1

class JuliaBackend(BackendProtocol):
    """HTTP bridge to a Julia microservice. Falls back to local on import/HTTP error."""
    def __init__(self, base_url: str = "http://localhost:9000"):
        self.base_url = base_url
        try:
            import requests
            self._requests = requests
        except Exception as e:
            self._requests = None
            print(f"[JuliaBackend] requests not available: {e}")

    def _post(self, path: str, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if self._requests is None:
            return None
        try:
            r = self._requests.post(f"{self.base_url}{path}", json=payload, timeout=10)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            print(f"[JuliaBackend] HTTP error: {e}")
            return None

    def analyze_entropy(self, data: List[List[float]]) -> Dict[str, float]:
        resp = self._post("/qvnm/estimate_id", {"V": data, "mode": "local"})
        return resp if isinstance(resp, dict) else entropy_local(data)

    def project_chebyshev(self, matrix: List[List[float]], degree: int) -> List[List[float]]:
        resp = self._post("/chebyshev/project", {"matrix": matrix, "degree": degree})
        if isinstance(resp, dict) and "projected" in resp:
            return resp["projected"]
        return chebyshev_local(matrix, degree)

    def optimize(self, matrix: List[List[float]], method: str) -> Dict[str, Any]:
        resp = self._post("/optimize", {"matrix": matrix, "method": method})
        if isinstance(resp, dict):
            return resp
        return MockBackend().optimize(matrix, method)

    def ghost_score(self, signature: Dict[str, Any]) -> float:
        resp = self._post("/ghost/score", {"signature": signature})
        if isinstance(resp, dict) and "score" in resp:
            return float(resp["score"])
        return MockBackend().ghost_score(signature)

class GPUBackend(BackendProtocol):
    """CuPy-based accelerated entropy/projection where practical."""
    def __init__(self):
        try:
            import cupy as cp
            self.cp = cp
        except Exception as e:
            raise ImportError(f"CuPy required for GPU backend: {e}")

    def analyze_entropy(self, data: List[List[float]]) -> Dict[str, float]:
        x = self.cp.array(data)
        z = self.cp.abs(x)
        p = z / (self.cp.sum(z) + 1e-12)
        p = p[p > 0]
        shannon = -self.cp.sum(p * self.cp.log2(p))
        return {"shannon": float(shannon.get()), "surprise": 0.0, "complexity": 0.0}

    def project_chebyshev(self, matrix: List[List[float]], degree: int) -> List[List[float]]:
        return chebyshev_local(matrix, degree)

    def optimize(self, matrix: List[List[float]], method: str) -> Dict[str, Any]:
        M = self.cp.array(matrix)
        M = M + self.cp.random.normal(0, 0.01, M.shape)
        return {
            "optimized": self.cp.asnumpy(M).tolist(),
            "loss": 0.2,
            "iterations": 10,
            "converged": True
        }

    def ghost_score(self, signature: Dict[str, Any]) -> float:
        return MockBackend().ghost_score(signature)

# ==============================
# DAG Orchestrator (Kahn topo-sort)
# ==============================
class DAGOrchestrator:
    def __init__(self):
        self.nodes: Dict[str, Node] = {}
        self.cache: Dict[str, Any] = {}
        self._graph: Dict[str, Set[str]] = {}
        self._in_degree: Dict[str, int] = {}

    def add_node(self, name: str, dependencies: List[str], func: Callable[..., Any]):
        self.nodes[name] = Node(name, dependencies, func)
        if name not in self._graph:
            self._graph[name] = set()
        if name not in self._in_degree:
            self._in_degree[name] = 0
        for dep in dependencies:
            if dep not in self._graph:
                self._graph[dep] = set()
            self._graph[dep].add(name)
            self._in_degree[name] = self._in_degree.get(name, 0) + 1

    def _resolve_order(self) -> List[str]:
        """Kahn's algorithm for true topological ordering."""
        queue = deque([n for n, deg in self._in_degree.items() if deg == 0 and n in self.nodes])
        result: List[str] = []
        in_deg = dict(self._in_degree)

        while queue:
            node = queue.popleft()
            result.append(node)
            for nbr in self._graph.get(node, set()):
                if nbr in in_deg:
                    in_deg[nbr] -= 1
                    if in_deg[nbr] == 0 and nbr in self.nodes:
                        queue.append(nbr)

        if len(result) != len(self.nodes):
            cycle_nodes = set(self.nodes.keys()) - set(result)
            raise ValueError(f"Cycle detected in DAG involving nodes: {cycle_nodes}")
        return result

    def _get_cache_key(self, node_name: str, context: Dict[str, Any]) -> str:
        deps_data = {}
        for dep in self.nodes[node_name].dependencies:
            if dep in context:
                deps_data[dep] = context[dep]
        if f"{node_name}_context" in context:
            deps_data["node_context"] = context[f"{node_name}_context"]
        deps_str = json.dumps(deps_data, sort_keys=True, default=str)
        return hashlib.sha256(deps_str.encode()).hexdigest()[:16]

# ==============================
# Adaptive + Serializable Orchestrator
# ==============================
class AdaptiveDAGOrchestrator(DAGOrchestrator):
    def __init__(self, backend: BackendProtocol, ghost_threshold: float = 0.25):
        super().__init__()
        self.backend = backend
        self.ghost_threshold = ghost_threshold
        self.adaptation_history: List[Dict[str, Any]] = []

    def _calculate_adaptive_params(self, ghost_score: Optional[float]) -> Dict[str, Any]:
        if ghost_score is None:
            return {"aggressiveness": 0.5, "max_iterations": 10, "learning_rate": 0.1}
        if ghost_score < self.ghost_threshold * 0.5:
            return {"aggressiveness": 0.8, "max_iterations": 20, "learning_rate": 0.2}
        elif ghost_score > self.ghost_threshold:
            return {"aggressiveness": 0.2, "max_iterations": 5, "learning_rate": 0.05}
        return {"aggressiveness": 0.5, "max_iterations": 10, "learning_rate": 0.1}

    def execute_with_adaptation(self, start_context: Dict[str, Any]) -> Dict[str, Any]:
        context = start_context.copy()
        for node_name in self._resolve_order():
            node = self.nodes[node_name]
            missing = [d for d in node.dependencies if d not in context]
            if missing:
                raise ValueError(f"Missing dependencies for {node_name}: {missing}")

            cache_key = self._get_cache_key(node_name, context)
            if cache_key in self.cache:
                context[node_name] = self.cache[cache_key]
                continue

            node_inputs = {dep: context[dep] for dep in node.dependencies}
            if getattr(node.func, "accepts_ghost_feedback", False) and "ghost_check" in context:
                node_inputs["previous_ghost_score"] = context["ghost_check"].get("ghost_score")
                node_inputs["adaptive_params"] = self._calculate_adaptive_params(
                    node_inputs["previous_ghost_score"]
                )

            result = node.func(**node_inputs)
            if isinstance(result, dict) and "adaptive_params" in result:
                self.adaptation_history.append({
                    "node": node_name,
                    "ghost_score": node_inputs.get("previous_ghost_score"),
                    "params": result.get("adaptive_params"),
                    "timestamp": time.time(),
                })

            context[node_name] = result
            self.cache[cache_key] = result
        return context

class SerializableDAGOrchestrator(AdaptiveDAGOrchestrator):
    def __init__(self, backend: BackendProtocol, log_dir: Path = Path("dag_logs")):
        super().__init__(backend)
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.current_run_id: Optional[str] = None

    def _generate_run_id(self) -> str:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        rand = hashlib.sha256(str(time.time()).encode()).hexdigest()[:8]
        return f"run_{ts}_{rand}"

    def serialize_run(self, context: Dict[str, Any], run_id: Optional[str] = None) -> str:
        if run_id is None:
            run_id = self.current_run_id or self._generate_run_id()
        run_data = {
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "context": context,
            "cache_state": self.cache,
            "adaptation_history": self.adaptation_history,
            "node_definitions": {
                name: {"dependencies": node.dependencies, "func_name": getattr(node.func, "__name__", str(node.func))}
                for name, node in self.nodes.items()
            },
        }
        out = self.log_dir / f"{run_id}.json"
        with open(out, "w") as f:
            json.dump(run_data, f, indent=2, default=str)
        return run_id

    def replay_run(self, run_id: str) -> Dict[str, Any]:
        path = self.log_dir / f"{run_id}.json"
        if not path.exists():
            raise FileNotFoundError(f"Run {run_id} not found")
        with open(path, "r") as f:
            data = json.load(f)
        self.cache = data.get("cache_state", {})
        self.adaptation_history = data.get("adaptation_history", [])
        return data.get("context", {})

    def execute_serializable(self, start_context: Dict[str, Any]) -> Dict[str, Any]:
        self.current_run_id = self._generate_run_id()
        try:
            result = self.execute_with_adaptation(start_context)
            self.serialize_run(result, self.current_run_id)
            return result
        except Exception:
            partial = {"inputs": start_context, "partial_cache": self.cache}
            self.serialize_run(partial, f"{self.current_run_id}_error")
            raise

# ==============================
# Example Wiring and Usage
# ==============================
def build_orchestrator(backend: BackendProtocol) -> SerializableDAGOrchestrator:
    orch = SerializableDAGOrchestrator(backend=backend, log_dir=Path("lattice_experiments"))

    def entropy_node(chunks: List[List[List[float]]]) -> List[Dict[str, float]]:
        return [orch.backend.analyze_entropy(chunk) for chunk in chunks]

    def projection_node(chunks: List[List[List[float]]], degree: int) -> Dict[str, Any]:
        arrays = [np.array(c) for c in chunks if c]
        if not arrays:
            combined = np.zeros((0, 0))
        else:
            try:
                combined = np.vstack(arrays)
            except ValueError:
                max_cols = max(a.shape[1] for a in arrays)
                normed = [np.pad(a, ((0,0),(0,max_cols - a.shape[1])), mode="constant") for a in arrays]
                combined = np.vstack(normed)
        proj = orch.backend.project_chebyshev(combined.tolist(), degree)
        return {"projected": proj, "original_shape": list(combined.shape)}

    def ghost_check_node(projection: Dict[str, Any]) -> Dict[str, Any]:
        proj = projection["projected"]
        shape = projection["original_shape"]
        norm = float(np.linalg.norm(np.array(proj))) if proj else 0.0
        sig = {"shape": shape, "norm": norm}
        score = orch.backend.ghost_score(sig)
        return {"ghost_score": float(score), "signature": sig}

    def adaptive_optimization_node(
        projection: Dict[str, Any],
        previous_ghost_score: Optional[float] = None,
        adaptive_params: Optional[Dict[str, Any]] = None,
        method: str = "gradient",
    ) -> Dict[str, Any]:
        out = orch.backend.optimize(projection["projected"], method)
        if isinstance(out, dict):
            out["previous_ghost_score"] = previous_ghost_score
            out["adaptive_params"] = adaptive_params
        return out

    adaptive_optimization_node.accepts_ghost_feedback = True

    orch.add_node("optimization", ["projection", "ghost_check"], adaptive_optimization_node)
    orch.add_node("projection", ["chunks", "degree"], projection_node)
    orch.add_node("entropy_analysis", ["chunks"], entropy_node)
    orch.add_node("ghost_check", ["projection"], ghost_check_node)

    return orch

if __name__ == "__main__":
    HAS_GPU = False
    JULIA_AVAILABLE = False
    
    backend: BackendProtocol
    if HAS_GPU:
        try:
            backend = GPUBackend()
        except Exception as e:
            print(f"[warn] GPU backend unavailable: {e}")
            backend = MockBackend()
    elif JULIA_AVAILABLE:
        backend = JuliaBackend()
    else:
        backend = MockBackend()

    orch = build_orchestrator(backend)

    sample_chunks = [
        [[1.0, 2.0], [3.0, 4.0]],
        [[0.5, 1.5], [2.5, 3.5]],
        [[0.1, 0.2], [0.3, 0.4]],
    ]

    result = orch.execute_serializable({
        "chunks": sample_chunks,
        "degree": 3,
        "method": "adaptive_gradient"
    })

    print(f"\nRun ID: {orch.current_run_id}")
    print("\n=== Results ===")
    for k, v in result.items():
        if k in ("chunks", "degree"):
            continue
        print(f"\n[{k}]")
        print(json.dumps(v, indent=2, default=str) if isinstance(v, (dict, list)) else v)

    gs = result["ghost_check"]["ghost_score"]
    print(f"\nGhost Score: {gs:.3f}")
    if gs > orch.ghost_threshold:
        print("⚠️  Warning: Ghost score approaching threshold")
    else:
        print("✅ Ghost score within safe limits")
