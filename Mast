# mirror_cast_enhanced.py — Advanced Self-Compiling Reflective Engine with Extended CLI

import copy
import uuid
import inspect
import argparse
import json
import math
import time
import hashlib
from typing import Any, Callable, Dict, List, Optional, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import sys

# --- Enhanced Entropy Engine ---
class EntropyAnalyzer:
    """Advanced entropy measurement with multiple algorithms"""
    
    def measure(self, data: Any) -> float:
        """Shannon entropy approximation"""
        s = str(data)
        if not s:
            return 0.0
        
        # Character frequency analysis
        char_counts = {}
        for char in s:
            char_counts[char] = char_counts.get(char, 0) + 1
        
        length = len(s)
        entropy = 0.0
        for count in char_counts.values():
            probability = count / length
            if probability > 0:
                entropy -= probability * math.log2(probability)
        
        return entropy
    
    def complexity_score(self, data: Any) -> float:
        """Kolmogorov complexity approximation"""
        s = str(data)
        compressed_len = len(json.dumps(data).encode('utf-8'))
        original_len = len(s.encode('utf-8'))
        return compressed_len / max(original_len, 1)

# --- Enhanced Dianne Reflector ---
class DianneReflector:
    """Symbolic insight generator with pattern recognition"""
    
    def reflect(self, data: Any) -> Dict[str, str]:
        patterns = self._detect_patterns(data)
        insight = self._generate_insight(data, patterns)
        
        return {
            "insight": insight,
            "patterns": patterns,
            "symbolic_depth": self._calculate_depth(data)
        }
    
    def _detect_patterns(self, data: Any) -> List[str]:
        patterns = []
        s = str(data)
        
        # Repetition patterns
        if len(set(s)) < len(s) * 0.5:
            patterns.append("high_repetition")
        
        # Numeric patterns
        if any(c.isdigit() for c in s):
            patterns.append("numeric_elements")
        
        # Structure patterns
        if isinstance(data, dict):
            patterns.append("hierarchical_structure")
        elif isinstance(data, list):
            patterns.append("sequential_structure")
        
        return patterns
    
    def _generate_insight(self, data: Any, patterns: List[str]) -> str:
        base_insight = f"Reflecting essence of: {str(data)[:40]}..."
        
        if "high_repetition" in patterns:
            return f"Cyclical resonance detected in {base_insight}"
        elif "hierarchical_structure" in patterns:
            return f"Nested reality layers within {base_insight}"
        else:
            return f"Linear transformation potential in {base_insight}"
    
    def _calculate_depth(self, data: Any) -> str:
        if isinstance(data, dict):
            max_depth = self._dict_depth(data)
            return f"depth_{max_depth}"
        elif isinstance(data, (list, tuple)):
            return f"sequence_length_{len(data)}"
        else:
            return f"scalar_complexity_{len(str(data))}"
    
    def _dict_depth(self, d: Dict, depth: int = 0) -> int:
        if not isinstance(d, dict) or not d:
            return depth
        return max(self._dict_depth(v, depth + 1) if isinstance(v, dict) else depth + 1 
                  for v in d.values())

# --- Enhanced Matrix Processor ---
class MatrixTransformer:
    """Advanced matrix operations with dimensional analysis"""
    
    def project(self, data: Any) -> Dict[str, Any]:
        dimensions = self._analyze_dimensions(data)
        transformation = self._compute_transformation(data, dimensions)
        
        return {
            "projected_rank": dimensions["rank"],
            "structure": dimensions["structure"],
            "eigenvalues": transformation["eigenvalues"],
            "determinant": transformation["determinant"],
            "trace": transformation["trace"]
        }
    
    def _analyze_dimensions(self, data: Any) -> Dict[str, Any]:
        if isinstance(data, dict):
            return {
                "rank": len(data),
                "structure": "sparse_matrix",
                "dimensionality": self._calculate_dict_dimensionality(data)
            }
        elif isinstance(data, (list, tuple)):
            return {
                "rank": len(data),
                "structure": "vector_space",
                "dimensionality": len(data)
            }
        else:
            return {
                "rank": 1,
                "structure": "scalar_field",
                "dimensionality": 1
            }
    
    def _compute_transformation(self, data: Any, dimensions: Dict) -> Dict[str, float]:
        # Pseudo-mathematical transformations based on data characteristics
        data_hash = hash(str(data))
        
        return {
            "eigenvalues": [math.sin(data_hash * 0.001 * i) for i in range(min(3, dimensions["rank"]))],
            "determinant": math.cos(data_hash * 0.0001),
            "trace": math.tan(data_hash * 0.00001) if data_hash % 100 != 0 else 0.0
        }
    
    def _calculate_dict_dimensionality(self, d: Dict) -> int:
        total = 0
        for v in d.values():
            if isinstance(v, dict):
                total += self._calculate_dict_dimensionality(v)
            else:
                total += 1
        return total

# --- Enhanced Julia Bridge ---
class JuliaSymbolEngine:
    """Mathematical symbol processing with polynomial generation"""
    
    def analyze(self, data: Any) -> Dict[str, Any]:
        coefficients = self._extract_coefficients(data)
        polynomial = self._generate_polynomial(coefficients)
        derivatives = self._compute_derivatives(coefficients)
        
        return {
            "chebyshev_polynomial": polynomial,
            "coefficients": coefficients,
            "derivatives": derivatives,
            "critical_points": self._find_critical_points(coefficients)
        }
    
    def _extract_coefficients(self, data: Any) -> List[float]:
        """Extract numerical coefficients from data"""
        s = str(data)
        coeffs = []
        
        # Extract numbers from string representation
        current_num = ""
        for char in s:
            if char.isdigit() or char == '.' or char == '-':
                current_num += char
            else:
                if current_num:
                    try:
                        coeffs.append(float(current_num))
                    except ValueError:
                        pass
                    current_num = ""
        
        if current_num:
            try:
                coeffs.append(float(current_num))
            except ValueError:
                pass
        
        # If no numbers found, use hash-based coefficients
        if not coeffs:
            data_hash = hash(str(data))
            coeffs = [(data_hash >> (i * 4)) % 10 - 5 for i in range(4)]
        
        return coeffs[:6]  # Limit to 6 coefficients
    
    def _generate_polynomial(self, coefficients: List[float]) -> str:
        if not coefficients:
            return "T0(x) = 1"
        
        terms = []
        for i, coeff in enumerate(coefficients):
            if abs(coeff) < 1e-10:
                continue
            
            if i == 0:
                terms.append(f"{coeff:.3f}")
            elif i == 1:
                terms.append(f"{coeff:.3f}x")
            else:
                terms.append(f"{coeff:.3f}x^{i}")
        
        if not terms:
            return "T0(x) = 0"
        
        return f"T(x) = {' + '.join(terms).replace('+ -', '- ')}"
    
    def _compute_derivatives(self, coefficients: List[float]) -> List[str]:
        if len(coefficients) <= 1:
            return ["T'(x) = 0"]
        
        derivatives = []
        current_coeffs = coefficients[:]
        
        for order in range(1, min(4, len(coefficients))):
            # Compute derivative coefficients
            deriv_coeffs = []
            for i in range(1, len(current_coeffs)):
                deriv_coeffs.append(current_coeffs[i] * i)
            
            if not deriv_coeffs:
                derivatives.append(f"T^({order})(x) = 0")
                break
            
            # Format derivative
            terms = []
            for i, coeff in enumerate(deriv_coeffs):
                if abs(coeff) < 1e-10:
                    continue
                
                if i == 0:
                    terms.append(f"{coeff:.3f}")
                elif i == 1:
                    terms.append(f"{coeff:.3f}x")
                else:
                    terms.append(f"{coeff:.3f}x^{i}")
            
            if terms:
                derivatives.append(f"T^({order})(x) = {' + '.join(terms).replace('+ -', '- ')}")
            else:
                derivatives.append(f"T^({order})(x) = 0")
            
            current_coeffs = deriv_coeffs
        
        return derivatives
    
    def _find_critical_points(self, coefficients: List[float]) -> List[float]:
        """Find approximate critical points using derivative"""
        if len(coefficients) <= 1:
            return []
        
        # Simple critical point approximation for demonstration
        critical_points = []
        for i in range(-10, 11):
            x = i * 0.5
            # Evaluate derivative at point
            deriv_value = sum(coeff * j * (x ** (j-1)) if j > 0 else 0 
                            for j, coeff in enumerate(coefficients))
            if abs(deriv_value) < 0.1:  # Approximate zero
                critical_points.append(x)
        
        return critical_points[:3]  # Return first 3 critical points

# --- Enhanced Choppy Chunker ---
class ChoppyProcessor:
    """Advanced data chunking with overlap and context preservation"""
    
    def chunk(self, data: Any, chunk_size: int = 10, overlap: int = 2) -> Dict[str, Any]:
        s = str(data)
        
        # Standard chunking
        standard_chunks = [s[i:i+chunk_size] for i in range(0, len(s), chunk_size - overlap)]
        
        # Semantic chunking (by words if possible)
        words = s.split()
        word_chunks = []
        for i in range(0, len(words), max(1, chunk_size // 3)):
            chunk_words = words[i:i + chunk_size // 3]
            word_chunks.append(' '.join(chunk_words))
        
        # Fibonacci chunking
        fib_chunks = self._fibonacci_chunk(s)
        
        return {
            "standard": standard_chunks,
            "semantic": word_chunks,
            "fibonacci": fib_chunks,
            "statistics": {
                "total_length": len(s),
                "chunk_count": len(standard_chunks),
                "average_chunk_size": len(s) / max(len(standard_chunks), 1)
            }
        }
    
    def _fibonacci_chunk(self, s: str) -> List[str]:
        """Chunk string using fibonacci sequence lengths"""
        def fib_gen():
            a, b = 1, 1
            while True:
                yield a
                a, b = b, a + b
        
        chunks = []
        fib = fib_gen()
        start = 0
        
        while start < len(s):
            chunk_size = next(fib)
            if chunk_size > 50:  # Prevent huge chunks
                chunk_size = 10
            
            end = min(start + chunk_size, len(s))
            chunks.append(s[start:end])
            start = end
        
        return chunks

# --- Enhanced Endpoint Caster ---
class EndpointCaster:
    """RESTful endpoint generation with versioning and metadata"""
    
    def generate(self, data: Any) -> Dict[str, Any]:
        data_signature = self._generate_signature(data)
        endpoints = self._create_endpoints(data_signature)
        
        return {
            "primary_endpoint": endpoints["primary"],
            "versioned_endpoints": endpoints["versions"],
            "artifact_id": f"art-{uuid.uuid4().hex[:8]}",
            "metadata": {
                "content_type": self._detect_content_type(data),
                "estimated_size": len(str(data)),
                "complexity": self._assess_complexity(data)
            }
        }
    
    def _generate_signature(self, data: Any) -> str:
        """Generate a unique signature for the data"""
        data_str = json.dumps(data, default=str, sort_keys=True)
        return hashlib.sha256(data_str.encode()).hexdigest()[:12]
    
    def _create_endpoints(self, signature: str) -> Dict[str, Any]:
        base_id = uuid.uuid4().hex[:6]
        
        return {
            "primary": f"/api/v1/cast/{base_id}",
            "versions": [
                f"/api/v1/cast/{base_id}/reflect",
                f"/api/v1/cast/{base_id}/transform",
                f"/api/v1/cast/{base_id}/metadata",
                f"/api/v2/mirror/{signature}"
            ]
        }
    
    def _detect_content_type(self, data: Any) -> str:
        if isinstance(data, dict):
            return "application/json"
        elif isinstance(data, str):
            return "text/plain"
        elif isinstance(data, (list, tuple)):
            return "application/array"
        else:
            return "application/octet-stream"
    
    def _assess_complexity(self, data: Any) -> str:
        size = len(str(data))
        if size < 100:
            return "low"
        elif size < 1000:
            return "medium"
        else:
            return "high"

# --- Enhanced Memory Manager ---
class CarryOnManager:
    """Advanced state management with persistence and history"""
    
    def __init__(self, max_history: int = 100):
        self.memory: Dict[str, Any] = {}
        self.history: List[Dict[str, Any]] = []
        self.max_history = max_history
        self.access_counts: Dict[str, int] = {}
    
    def store_state(self, key: str, state: Any, metadata: Optional[Dict] = None):
        """Store state with optional metadata and history tracking"""
        timestamp = time.time()
        
        self.memory[key] = {
            "data": state,
            "timestamp": timestamp,
            "metadata": metadata or {}
        }
        
        # Update access count
        self.access_counts[key] = self.access_counts.get(key, 0) + 1
        
        # Add to history
        self.history.append({
            "operation": "store",
            "key": key,
            "timestamp": timestamp,
            "data_preview": str(state)[:50]
        })
        
        # Maintain history size
        if len(self.history) > self.max_history:
            self.history.pop(0)
    
    def resume_state(self, key: str) -> Optional[Dict[str, Any]]:
        """Resume state with access tracking"""
        if key in self.memory:
            self.access_counts[key] = self.access_counts.get(key, 0) + 1
            
            self.history.append({
                "operation": "resume",
                "key": key,
                "timestamp": time.time(),
                "access_count": self.access_counts[key]
            })
            
            return self.memory[key]
        return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get memory usage statistics"""
        return {
            "total_keys": len(self.memory),
            "history_length": len(self.history),
            "most_accessed": max(self.access_counts.items(), key=lambda x: x[1]) if self.access_counts else None,
            "memory_size_estimate": sum(len(str(v)) for v in self.memory.values())
        }

# --- Enhanced Semantic Mapper ---
class SemanticMapper:
    """Advanced semantic expansion with contextual relationships"""
    
    def __init__(self):
        self.semantic_networks = {
            "reflection": ["mirror", "echo", "reverberation", "contemplation", "introspection"],
            "transformation": ["metamorphosis", "mutation", "evolution", "adaptation", "transmutation"],
            "analysis": ["examination", "scrutiny", "dissection", "investigation", "exploration"],
            "synthesis": ["combination", "fusion", "amalgamation", "integration", "unification"]
        }
    
    def expand(self, data: Any) -> Dict[str, List[str]]:
        """Generate semantic expansions with contextual groupings"""
        text = str(data).lower()
        
        expansions = {
            "direct_synonyms": self._find_direct_synonyms(text),
            "contextual_relations": self._find_contextual_relations(text),
            "conceptual_clusters": self._generate_conceptual_clusters(text),
            "semantic_distance": self._calculate_semantic_distances(text)
        }
        
        return expansions
    
    def _find_direct_synonyms(self, text: str) -> List[str]:
        """Find direct synonyms based on keyword matching"""
        synonyms = []
        for keyword, synonym_list in self.semantic_networks.items():
            if keyword in text:
                synonyms.extend(synonym_list)
        
        # Add some general reflection-related terms
        base_synonyms = ["echo", "mirror", "reflection", "resonance", "harmony"]
        return list(set(synonyms + base_synonyms))[:8]  # Limit to 8 synonyms
    
    def _find_contextual_relations(self, text: str) -> List[str]:
        """Find contextually related terms"""
        relations = []
        
        # Analyze text characteristics
        if any(char.isdigit() for char in text):
            relations.extend(["numerical", "quantitative", "measured"])
        
        if len(text) > 100:
            relations.extend(["complex", "detailed", "elaborate"])
        
        if '{' in text or '[' in text:
            relations.extend(["structured", "hierarchical", "organized"])
        
        return relations
    
    def _generate_conceptual_clusters(self, text: str) -> Dict[str, List[str]]:
        """Generate clusters of related concepts"""
        return {
            "mathematical": ["function", "transformation", "mapping", "projection"],
            "philosophical": ["essence", "truth", "reality", "consciousness"],
            "computational": ["algorithm", "process", "calculation", "iteration"],
            "artistic": ["creativity", "expression", "beauty", "harmony"]
        }
    
    def _calculate_semantic_distances(self, text: str) -> Dict[str, float]:
        """Calculate semantic distances to key concepts"""
        distances = {}
        
        key_concepts = ["mirror", "transform", "analyze", "reflect"]
        
        for concept in key_concepts:
            # Simple distance based on character overlap and length difference
            common_chars = set(text) & set(concept)
            distance = 1.0 - (len(common_chars) / max(len(set(text)), len(set(concept))))
            distances[concept] = round(distance, 3)
        
        return distances

# --- Enhanced Love Reflector ---
class LoveReflector:
    """Poetic and emotional dimension analysis"""
    
    def infuse(self, data: Any) -> Dict[str, Any]:
        emotional_resonance = self._analyze_emotional_resonance(data)
        poetic_transformation = self._create_poetic_transformation(data)
        
        return {
            "poetic": poetic_transformation,
            "emotional_resonance": emotional_resonance,
            "love_quotient": self._calculate_love_quotient(data),
            "harmony_index": self._measure_harmony(data)
        }
    
    def _analyze_emotional_resonance(self, data: Any) -> Dict[str, Any]:
        """Analyze emotional characteristics of data"""
        text = str(data)
        
        return {
            "warmth": min(1.0, len([c for c in text if c in 'warm']) / max(len(text), 1) * 100),
            "complexity": min(1.0, len(set(text)) / max(len(text), 1) * 2),
            "rhythm": self._calculate_rhythm(text),
            "sentiment": "positive" if hash(text) % 3 == 0 else "neutral" if hash(text) % 3 == 1 else "contemplative"
        }
    
    def _create_poetic_transformation(self, data: Any) -> str:
        """Create poetic representation of data"""
        text = str(data)[:20]
        
        poetic_templates = [
            f"In mirrors deep, {text} whispers truth...",
            f"Through love's lens, {text} transforms to light...",
            f"Echoing softly, {text} finds its voice...",
            f"Dancing shadows of {text} embrace the void...",
            f"Golden threads weave {text} into dreams..."
        ]
        
        return poetic_templates[hash(text) % len(poetic_templates)]
    
    def _calculate_love_quotient(self, data: Any) -> float:
        """Calculate a whimsical 'love quotient' based on data characteristics"""
        text = str(data)
        
        # Factors that increase love quotient
        vowel_count = len([c for c in text.lower() if c in 'aeiou'])
        symmetry = self._measure_symmetry(text)
        golden_ratio_proximity = abs(len(text) / max(len(set(text)), 1) - 1.618)
        
        love_quotient = (vowel_count * 0.1 + symmetry * 0.5 - golden_ratio_proximity * 0.1)
        return max(0.0, min(1.0, love_quotient))
    
    def _measure_harmony(self, data: Any) -> float:
        """Measure the 'harmony' of data structure"""
        if isinstance(data, dict):
            # Dictionary harmony based on key-value balance
            keys_len = sum(len(str(k)) for k in data.keys())
            values_len = sum(len(str(v)) for v in data.values())
            return 1.0 - abs(keys_len - values_len) / max(keys_len + values_len, 1)
        
        elif isinstance(data, (list, tuple)):
            # List harmony based on element size consistency
            if not data:
                return 1.0
            lengths = [len(str(item)) for item in data]
            avg_length = sum(lengths) / len(lengths)
            variance = sum((l - avg_length) ** 2 for l in lengths) / len(lengths)
            return 1.0 / (1.0 + variance * 0.1)
        
        else:
            # String harmony based on character distribution
            text = str(data)
            if not text:
                return 1.0
            
            char_counts = {}
            for char in text:
                char_counts[char] = char_counts.get(char, 0) + 1
            
            # Calculate evenness of distribution
            max_count = max(char_counts.values())
            min_count = min(char_counts.values())
            return min_count / max_count if max_count > 0 else 1.0
    
    def _calculate_rhythm(self, text: str) -> float:
        """Calculate rhythmic quality of text"""
        if len(text) < 2:
            return 0.5
        
        # Measure alternation between different character types
        rhythm_score = 0.0
        for i in range(len(text) - 1):
            if text[i].isalpha() != text[i+1].isalpha():
                rhythm_score += 1
            if text[i].isdigit() != text[i+1].isdigit():
                rhythm_score += 0.5
        
        return min(1.0, rhythm_score / len(text))
    
    def _measure_symmetry(self, text: str) -> float:
        """Measure textual symmetry"""
        if len(text) <= 1:
            return 1.0
        
        # Check palindromic properties
        cleaned = ''.join(c.lower() for c in text if c.isalnum())
        if not cleaned:
            return 0.5
        
        matches = sum(1 for i in range(len(cleaned)) if i < len(cleaned) - 1 - i and 
                     cleaned[i] == cleaned[-(i+1)])
        
        return matches / (len(cleaned) // 2) if len(cleaned) > 1 else 1.0

# --- Enhanced Configuration ---
@dataclass
class MirrorCastConfig:
    """Enhanced configuration with comprehensive settings"""
    name: str
    transform: Callable[[Any], Any]
    max_depth: int = 3
    enable_love_reflection: bool = True
    enable_julia_analysis: bool = True
    enable_matrix_projection: bool = True
    chunk_size: int = 10
    chunk_overlap: int = 2
    max_memory_entries: int = 100
    enable_statistics: bool = True
    
    def __post_init__(self):
        # Initialize all components
        self.entropy_analyzer = EntropyAnalyzer()
        self.dianne_reflector = DianneReflector()
        self.matrix_transformer = MatrixTransformer()
        self.julia_engine = JuliaSymbolEngine()
        self.choppy = ChoppyProcessor()
        self.eopiez = EndpointCaster()
        self.semantic = SemanticMapper()
        self.carryon = CarryOnManager(self.max_memory_entries)
        self.love_reflector = LoveReflector()

# --- Enhanced Mirror Node ---
class MirrorNode:
    """Enhanced mirror node with comprehensive analysis"""
    
    def __init__(self, config: MirrorCastConfig, depth: int = 0):
        self.config = config
        self.depth = depth
        self.id = f"node-{uuid.uuid4().hex[:8]}"
        self.children: List[MirrorNode] = []
        self.creation_time = time.time()
        
        # Create children based on depth
        if self.depth < self.config.max_depth:
            child_count = max(1, min(3, 2 + self.depth))
            for _ in range(child_count):
                self.children.append(MirrorNode(config, self.depth + 1))
    
    def reflect(self, data: Any, metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """Enhanced reflection with comprehensive analysis"""
        start_time = time.time()
        
        # Core transformations
        original = copy.deepcopy(data)
        mirrored = self.config.transform(original)
        
        # Advanced chunking
        chunks = self.config.choppy.chunk(data, self.config.chunk_size, self.config.chunk_overlap)
        
        # Enhanced entropy analysis
        entropy_score = self.config.entropy_analyzer.measure(data)
        complexity_score = self.config.entropy_analyzer.complexity_score(data)
        
        # Mathematical analysis
        harmonic_resonance = math.sin(self.depth * math.pi / (self.config.max_depth + 1))
        fractal_power = (self.depth + 1) ** min(self.depth + 1, 3)  # Prevent excessive growth
        harmonic_potential = harmonic_resonance * fractal_power
        
        # Logic deconstruction with safer bounds
        logic_steps = min(20, max(3, int(abs(harmonic_potential)) % 15 + 3))
        logic_deconstruction = [
            f"logic[{i}]: {str(data)[:10]} → transform({i}) = {(i ** 0.5):.4f}"
            for i in range(1, logic_steps)
        ]
        
        # Singularity detection
        singularity_threshold = 1000
        singularity_state = "STUTTER_DETECTED" if harmonic_potential > singularity_threshold else "STABLE"
        
        # Build comprehensive result
        result = {
            "node_info": {
                "id": self.id,
                "depth": self.depth,
                "creation_time": self.creation_time,
                "processing_time": time.time() - start_time
            },
            "data_analysis": {
                "input": original,
                "mirrored": mirrored,
                "chunks": chunks,
                "entropy": {
                    "shannon_entropy": entropy_score,
                    "complexity_score": complexity_score,
                    "adjusted_entropy": entropy_score + harmonic_potential * 0.001
                }
            },
            "mathematical_analysis": {
                "harmonic_resonance": harmonic_resonance,
                "fractal_power": fractal_power,
                "harmonic_potential": harmonic_potential,
                "singularity_state": singularity_state,
                "logic_deconstruction": logic_deconstruction
            }
        }
        
        # Optional enhanced analyses
        if self.config.enable_julia_analysis:
            result["julia_analysis"] = self.config.julia_engine.analyze(mirrored)
        
        if self.config.enable_matrix_projection:
            result["matrix_projection"] = self.config.matrix_transformer.project(mirrored)
        
        # Always include these analyses
        result["symbolic_reflection"] = self.config.dianne_reflector.reflect(mirrored)
        result["endpoint_generation"] = self.config.eopiez.generate(mirrored)
        result["semantic_expansion"] = self.config.semantic.expand(mirrored)
        
        if self.config.enable_love_reflection:
            result["love_infusion"] = self.config.love_reflector.infuse(mirrored)
        
        # Store state and process children
        self.config.carryon.store_state(self.id, mirrored, metadata)
        
        # Process children
        result["children"] = []
        for child in self.children:
            child_result = child.reflect(mirrored, {"parent_id": self.id, "depth": self.depth + 1})
            result["children"].append(child_result)
        
        # Add statistics if enabled
        if self.config.enable_statistics:
            result["statistics"] = {
                "total_nodes": 1 + len(result["children"]),
                "max_child_depth": max((child.get("node_info", {
