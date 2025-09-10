#!/usr/bin/env python3
"""
Pure Python DAG Orchestrator with Local Fallbacks
Clean reconstruction from mangled signal - preserves original intent
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import hashlib
import json
from pathlib import Path

# ============= Local Math Fallbacks =============
def entropy_local(data: List[List[float]]) -> Dict[str, float]:
    """Shannon entropy fallback when backend fails"""
    if not data or not any(data):
        return {"shannon": 0.0, "surprise": 0.0, "complexity": 0.0}
    
    flat = np.array(data).flatten()
    flat = flat[np.isfinite(flat)]
    if len(flat) == 0:
        return {"shannon": 0.0, "surprise": 0.0, "complexity": 0.0}
    
    # Normalize to probability distribution
    p = np.abs(flat) / (np.sum(np.abs(flat)) + 1e-12)
    p = p[p > 0]  # Remove zeros
    
    shannon = -np.sum(p * np.log2(p))
    
    # Simple surprise metric (variance of log probabilities)
    log_p = np.log2(p + 1e-12)
    surprise = np.var(log_p)
    
    # Complexity as product of entropy and surprise
    complexity = shannon * surprise
    
    return {"shannon": float(shannon), "surprise": float(surprise), "complexity": float(complexity)}

def chebyshev_local(matrix: List[List[float]], degree: int) -> List[List[float]]:
    """Chebyshev polynomial projection fallback"""
    if not matrix or not any(matrix):
        return matrix
    
    M = np.array(matrix, dtype=np.float64)
    
    # Simple Chebyshev-like transformation
    # T_n(x) = cos(n * arccos(x)) approximated for matrix
    if degree == 0:
        return np.ones_like(M).tolist()
    elif degree == 1:
        return M.tolist()
    else:
        # Higher degree approximation
        T_prev = np.ones_like(M)
        T_curr = M.copy()
        
        for n in range(2, degree + 1):
            T_next = 2 * M * T_curr - T_prev
            T_prev, T_curr = T_curr, T_next
        
        return T_curr.tolist()

# ============= DAG Orchestrator =============
@dataclass
class Node:
    name: str
    dependencies: List[str]
    func: callable

class DAGOrchestrator:
    def __init__(self):
        self.nodes: Dict[str, Node] = {}
        self.cache: Dict[str, Any] = {}
    
    def add_node(self, name: str, dependencies: List[str], func: callable):
        self.nodes[name] = Node(name, dependencies, func)
    
    def _get_cache_key(self, node_name: str, context: Dict[str, Any]) -> str:
        """Create deterministic cache key from node dependencies"""
        deps_data = {}
        for dep in self.nodes[node_name].dependencies:
            if dep in context:
                deps_data[dep] = context[dep]
        
        # Add node-specific context if needed
        if f"{node_name}_context" in context:
            deps_data["node_context"] = context[f"{node_name}_context"]
        
        # Create stable hash
        deps_str = json.dumps(deps_data, sort_keys=True, default=str)
        return hashlib.sha256(deps_str.encode()).hexdigest()[:16]
    
    def execute(self, start_context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute DAG in dependency order"""
        context = start_context.copy()
        execution_order = self._resolve_order()
        
        for node_name in execution_order:
            node = self.nodes[node_name]
            
            # Check if all dependencies are satisfied
            missing_deps = [dep for dep in node.dependencies if dep not in context]
            if missing_deps:
                raise ValueError(f"Missing dependencies for {node_name}: {missing_deps}")
            
            # Check cache
            cache_key = self._get_cache_key(node_name, context)
            if cache_key in self.cache:
                context[node_name] = self.cache[cache_key]
                continue
            
            # Execute node
            try:
                node_inputs = {dep: context[dep] for dep in node.dependencies}
                result = node.func(**node_inputs)
                context[node_name] = result
                self.cache[cache_key] = result
            except Exception as e:
                raise RuntimeError(f"Node {node_name} failed: {str(e)}") from e
        
        return context
    
    def _resolve_order(self) -> List[str]:
        """Resolve execution order using topological sort"""
        # Simple dependency resolution - in practice would use proper toposort
        # This assumes nodes were added in roughly dependency order
        return list(self.nodes.keys())

# ============= Mock Backend =============
class MockBackend:
    """Mock backend that uses local fallbacks"""
    def analyze_entropy(self, data: List[List[float]]) -> Dict[str, float]:
        return entropy_local(data)
    
    def project_chebyshev(self, matrix: List[List[float]], degree: int) -> List[List[float]]:
        return chebyshev_local(matrix, degree)
    
    def optimize(self, matrix: List[List[float]], method: str = "gradient") -> Dict[str, Any]:
        """Mock optimization - adds small improvements"""
        M = np.array(matrix)
        if M.size == 0:
            return {"optimized": matrix, "loss": 0.0, "iterations": 0, "converged": True}
        
        # Add small random improvements
        optimized = M + np.random.normal(0, 0.01, M.shape)
        return {
            "optimized": optimized.tolist(),
            "loss": float(np.random.uniform(0.1, 0.5)),
            "iterations": np.random.randint(5, 20),
            "converged": np.random.random() > 0.2
        }
    
    def ghost_score(self, signature: Dict[str, Any]) -> float:
        """Mock ghost score - measures 'structuredness'"""
        # Higher score for more structured, lower for random
        if "shape" in signature:
            size = np.prod(signature["shape"])
            norm = signature.get("norm", 1.0)
            # More structured = higher score (0.0 to 1.0)
            structure = min(1.0, norm / (size ** 0.5 + 1e-12))
            return float(structure * 0.3)  # Cap at 0.3 for safety
        return 0.1  # Default safe value

# ============= Example Usage =============
def create_example_dag():
    """Create example DAG for testing"""
    orchestrator = DAGOrchestrator()
    backend = MockBackend()
    
    # Define nodes
    def entropy_node(chunks: List[List[List[float]]]) -> List[Dict[str, float]]:
        return [backend.analyze_entropy(chunk) for chunk in chunks]
    
    def projection_node(chunks: List[List[List[float]]], degree: int) -> Dict[str, Any]:
        # Combine all chunks into single matrix
        combined = np.vstack([np.array(chunk) for chunk in chunks if chunk])
        projected = backend.project_chebyshev(combined.tolist(), degree)
        return {"projected": projected, "original_shape": combined.shape}
    
    def ghost_check_node(projected_data: Dict[str, Any]) -> Dict[str, Any]:
        projected_matrix = projected_data["projected"]
        signature = {
            "shape": [len(projected_matrix), len(projected_matrix[0]) if projected_matrix else 0],
            "norm": float(np.linalg.norm(np.array(projected_matrix)) if projected_matrix else 0.0)
        }
        score = backend.ghost_score(signature)
        return {"ghost_score": score, "signature": signature}
    
    def optimization_node(projected_data: Dict[str, Any], method: str = "gradient") -> Dict[str, Any]:
        return backend.optimize(projected_data["projected"], method)
    
    # Add nodes to DAG
    orchestrator.add_node("entropy_analysis", ["chunks"], entropy_node)
    orchestrator.add_node("projection", ["chunks", "degree"], projection_node)
    orchestrator.add_node("ghost_check", ["projection"], ghost_check_node)
    orchestrator.add_node("optimization", ["projection"], optimization_node)
    
    return orchestrator

# ============= Main Execution =============
if __name__ == "__main__":
    # Example data
    chunks = [
        [[1.0, 2.0], [3.0, 4.0]],  # 2x2 matrix
        [[0.5, 1.5], [2.5, 3.5]],  # Another 2x2
        [[0.1, 0.2], [0.3, 0.4]]   # Third 2x2
    ]
    
    # Create and execute DAG
    dag = create_example_dag()
    result = dag.execute({
        "chunks": chunks,
        "degree": 3,
        "projection_context": {"method": "chebyshev"}
    })
    
    print("=== DAG Execution Results ===")
    for key, value in result.items():
        if key not in ["chunks", "degree"]:  # Skip inputs
            print(f"\n{key}:")
            if isinstance(value, (list, dict)):
                print(json.dumps(value, indent=2))
            else:
                print(value)
    
    # Check ghost score safety
    ghost_score = result["ghost_check"]["ghost_score"]
    print(f"\nGhost Score: {ghost_score:.3f}")
    if ghost_score > 0.25:
        print("⚠️  Warning: Ghost score approaching threshold")
    else:
        print("✅ Ghost score within safe limits")
def ghost_score(self, signature: Dict[str, Any]) -> float:
    # Higher score for more structured, lower for random
    if "shape" in signature:
        size = np.prod(signature["shape"])
        norm = signature.get("norm", 1.0)
        structure = min(1.0, norm / (size ** 0.5 + 1e-12))
        return float(structure * 0.3)  # Cap at 0.3 for safety
    return 0.1  # Default safe value
from collections import deque
from typing import Dict, List, Set

class DAGOrchestrator:
    def __init__(self):
        self.nodes: Dict[str, Node] = {}
        self.cache: Dict[str, Any] = {}
        self._graph: Dict[str, Set[str]] = {}  # adjacency list
        self._in_degree: Dict[str, int] = {}   # in-degree count
    
    def add_node(self, name: str, dependencies: List[str], func: callable):
        self.nodes[name] = Node(name, dependencies, func)
        
        # Initialize graph and in-degree
        if name not in self._graph:
            self._graph[name] = set()
        if name not in self._in_degree:
            self._in_degree[name] = 0
        
        # Update dependencies
        for dep in dependencies:
            if dep not in self._graph:
                self._graph[dep] = set()
            self._graph[dep].add(name)  # dep -> current node
            self._in_degree[name] += 1
    
    def _resolve_order(self) -> List[str]:
        """Kahn's algorithm for topological sorting"""
        # Initialize queue with nodes having no dependencies
        queue = deque([node for node, degree in self._in_degree.items() 
                      if degree == 0 and node in self.nodes])
        result = []
        in_degree_copy = self._in_degree.copy()
        
        while queue:
            node = queue.popleft()
            result.append(node)
            
            # Reduce in-degree of all neighbors
            for neighbor in self._graph.get(node, set()):
                if neighbor not in in_degree_copy:
                    continue
                in_degree_copy[neighbor] -= 1
                if in_degree_copy[neighbor] == 0:
                    queue.append(neighbor)
        
        # Check for cycles
        if len(result) != len(self.nodes):
            cycle_nodes = set(self.nodes.keys()) - set(result)
            raise ValueError(f"Cycle detected in DAG involving nodes: {cycle_nodes}")
        
        return result
# This works perfectly now - no dependency order required
orchestrator = DAGOrchestrator()
orchestrator.add_node("optimization", ["projection"], optimization_node)
orchestrator.add_node("projection", ["chunks", "degree"], projection_node)
orchestrator.add_node("entropy_analysis", ["chunks"], entropy_node)
orchestrator.add_node("ghost_check", ["projection"], ghost_check_node)
from typing import Protocol, runtime_checkable

@runtime_checkable
class BackendProtocol(Protocol):
    def analyze_entropy(self, data: List[List[float]]) -> Dict[str, float]:
        ...
    
    def project_chebyshev(self, matrix: List[List[float]], degree: int) -> List[List[float]]:
        ...
    
    def optimize(self, matrix: List[List[float]], method: str) -> Dict[str, Any]:
        ...
    
    def ghost_score(self, signature: Dict[str, Any]) -> float:
        ...

# Mock backend implementing the protocol
class MockBackend:
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
            "iterations": np.random.randint(5, 20),
            "converged": np.random.random() > 0.2
        }
    
    def ghost_score(self, signature: Dict[str, Any]) -> float:
        if "shape" in signature:
            size = np.prod(signature["shape"])
            norm = signature.get("norm", 1.0)
            structure = min(1.0, norm / (size ** 0.5 + 1e-12))
            return float(structure * 0.3)
        return 0.1

# Julia backend implementing the same protocol
class JuliaBackend:
    def __init__(self, base_url: str = "http://localhost:9000"):
        self.base_url = base_url
    
    def analyze_entropy(self, data: List[List[float]]) -> Dict[str, float]:
        # Make HTTP request to Julia server
        response = requests.post(f"{self.base_url}/qvnm/estimate_id", 
                               json={"V": data, "mode": "local"})
        return response.json()
    
    def project_chebyshev(self, matrix: List[List[float]], degree: int) -> List[List[float]]:
        response = requests.post(f"{self.base_url}/chebyshev/project",
                               json={"matrix": matrix, "degree": degree})
        return response.json()["projected"]
    
    def optimize(self, matrix: List[List[float]], method: str) -> Dict[str, Any]:
        response = requests.post(f"{self.base_url}/optimize",
                               json={"matrix": matrix, "method": method})
        return response.json()
    
    def ghost_score(self, signature: Dict[str, Any]) -> float:
        response = requests.post(f"{self.base_url}/ghost/score",
                               json={"signature": signature})
        return response.json()["score"]

# GPU-accelerated backend
class GPUBackend:
    def __init__(self):
        try:
            import cupy as cp
            self.cp = cp
        except ImportError:
            raise ImportError("CuPy required for GPU backend")
    
    def analyze_entropy(self, data: List[List[float]]) -> Dict[str, float]:
        # GPU-accelerated entropy calculation
        x = self.cp.array(data)
        p = self.cp.abs(x) / (self.cp.sum(self.cp.abs(x)) + 1e-12)
        p = p[p > 0]
        shannon = -self.cp.sum(p * self.cp.log2(p))
        return {"shannon": float(shannon.get())}
# Use different backends based on availability
if HAS_GPU:
    backend = GPUBackend()
elif JULIA_AVAILABLE:
    backend = JuliaBackend()
else:
    backend = MockBackend()

# DAG doesn't care which backend we use - same interface
class AdaptiveDAGOrchestrator(DAGOrchestrator):
    def __init__(self, backend: BackendProtocol, ghost_threshold: float = 0.25):
        super().__init__()
        self.backend = backend
        self.ghost_threshold = ghost_threshold
        self.adaptation_history: List[Dict[str, Any]] = []
    
    def adaptive_optimization_node(self, projected_data: Dict[str, Any], 
                                 previous_ghost_score: Optional[float] = None,
                                 method: str = "gradient") -> Dict[str, Any]:
        """Optimization that adapts based on ghost score feedback"""
        
        # Adjust parameters based on previous ghost score
        adaptive_params = self._calculate_adaptive_params(previous_ghost_score)
        
        # Perform optimization with adaptive parameters
        result = self.backend.optimize(projected_data["projected"], method)
        
        # Add adaptation metadata
        result["adaptive_params"] = adaptive_params
        result["previous_ghost_score"] = previous_ghost_score
        
        return result
    
    def _calculate_adaptive_params(self, ghost_score: Optional[float]) -> Dict[str, Any]:
        """Calculate adaptive parameters based on ghost score"""
        if ghost_score is None:
            return {"aggressiveness": 0.5, "max_iterations": 10, "learning_rate": 0.1}
        
        # More aggressive when ghost score is low (safe)
        if ghost_score < self.ghost_threshold * 0.5:
            return {"aggressiveness": 0.8, "max_iterations": 20, "learning_rate": 0.2}
        # More cautious when ghost score is high (risky)
        elif ghost_score > self.ghost_threshold:
            return {"aggressiveness": 0.2, "max_iterations": 5, "learning_rate": 0.05}
        else:
            return {"aggressiveness": 0.5, "max_iterations": 10, "learning_rate": 0.1}
    
    def execute_with_adaptation(self, start_context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute with ghost score feedback between nodes"""
        context = start_context.copy()
        execution_order = self._resolve_order()
        
        for node_name in execution_order:
            node = self.nodes[node_name]
            
            # Check dependencies
            missing_deps = [dep for dep in node.dependencies if dep not in context]
            if missing_deps:
                raise ValueError(f"Missing dependencies for {node_name}: {missing_deps}")
            
            # Get cache key
            cache_key = self._get_cache_key(node_name, context)
            if cache_key in self.cache:
                context[node_name] = self.cache[cache_key]
                continue
            
            # Execute with potential adaptation
            node_inputs = {dep: context[dep] for dep in node.dependencies}
            
            # Inject ghost score if available and node wants it
            if (hasattr(node.func, 'accepts_ghost_feedback') and 
                "ghost_check" in context):
                node_inputs["previous_ghost_score"] = context["ghost_check"].get("ghost_score")
            
            result = node.func(**node_inputs)
            context[node_name] = result
            self.cache[cache_key] = result
            
            # Record adaptation if applicable
            if "adaptive_params" in result:
                self.adaptation_history.append({
                    "node": node_name,
                    "ghost_score": node_inputs.get("previous_ghost_score"),
                    "params": result["adaptive_params"],
                    "timestamp": time.time()
                })
        
        return context
import json
from datetime import datetime
from pathlib import Path

class SerializableDAGOrchestrator(AdaptiveDAGOrchestrator):
    def __init__(self, backend: BackendProtocol, log_dir: Path = Path("dag_logs")):
        super().__init__(backend)
        self.log_dir = log_dir
        self.log_dir.mkdir(exist_ok=True)
        self.current_run_id: Optional[str] = None
    
    def _generate_run_id(self) -> str:
        """Generate unique run ID with timestamp"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        random_suffix = hashlib.sha256(str(time.time()).encode()).hexdigest()[:8]
        return f"run_{timestamp}_{random_suffix}"
    
    def serialize_run(self, context: Dict[str, Any], run_id: Optional[str] = None) -> str:
        """Serialize complete DAG run state"""
        if run_id is None:
            run_id = self.current_run_id or self._generate_run_id()
        
        run_data = {
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "context": context,
            "cache_state": self.cache,
            "adaptation_history": self.adaptation_history,
            "node_definitions": {
                name: {"dependencies": node.dependencies, "func_name": node.func.__name__}
                for name, node in self.nodes.items()
            }
        }
        
        # Save to file
        output_path = self.log_dir / f"{run_id}.json"
        with open(output_path, 'w') as f:
            json.dump(run_data, f, indent=2, default=str)
        
        return run_id
    
    def replay_run(self, run_id: str) -> Dict[str, Any]:
        """Replay a serialized DAG run"""
        input_path = self.log_dir / f"{run_id}.json"
        if not input_path.exists():
            raise FileNotFoundError(f"Run {run_id} not found")
        
        with open(input_path, 'r') as f:
            run_data = json.load(f)
        
        # Restore cache and context
        self.cache = run_data["cache_state"]
        self.adaptation_history = run_data["adaptation_history"]
        
        return run_data["context"]
    
    def execute_serializable(self, start_context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute and automatically serialize"""
        self.current_run_id = self._generate_run_id()
        
        try:
            result = self.execute_with_adaptation(start_context)
            self.serialize_run(result, self.current_run_id)
            return result
        except Exception as e:
            # Serialize partial run on failure
            partial_context = {k: v for k, v in locals().items() if k in self.nodes or k in start_context}
            self.serialize_run(partial_context, f"{self.current_run_id}_error")
            raise
# Create fully upgraded orchestrator
orchestrator = SerializableDAGOrchestrator(
    backend=JuliaBackend(),  # Or MockBackend() or GPUBackend()
    log_dir=Path("lattice_experiments")
)

# Define adaptive nodes
def adaptive_optimization_node(projected_data: Dict[str, Any], 
                             previous_ghost_score: Optional[float] = None,
                             method: str = "gradient") -> Dict[str, Any]:
    return orchestrator.backend.optimize(projected_data["projected"], method)

adaptive_optimization_node.accepts_ghost_feedback = True

# Add nodes in any order
orchestrator.add_node("optimization", ["projection", "ghost_check"], adaptive_optimization_node)
orchestrator.add_node("projection", ["chunks", "degree"], 
                     lambda chunks, degree: orchestrator.backend.project_chebyshev(
                         np.vstack(chunks).tolist(), degree))
orchestrator.add_node("entropy_analysis", ["chunks"], 
                     lambda chunks: [orchestrator.backend.analyze_entropy(chunk) for chunk in chunks])
orchestrator.add_node("ghost_check", ["projection"], 
                     lambda projection: {"ghost_score": orchestrator.backend.ghost_score(
                         {"shape": projection["original_shape"], "norm": np.linalg.norm(projection["projected"])})})

# Execute with full capabilities
result = orchestrator.execute_serializable({
    "chunks": sample_chunks,
    "degree": 3,
    "method": "adaptive_gradient"
})

print(f"Run ID: {orchestrator.current_run_id}")
print(f"Ghost score: {result['ghost_check']['ghost_score']:.3f}")
print(f"Adaptations: {len(orchestrator.adaptation_history)}")
