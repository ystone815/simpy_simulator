"""GNN access pattern definitions and implementations"""
from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

class AccessPattern(Enum):
    """Available GNN access patterns"""
    THREAD_0_LEADER = "thread_0_leader"
    MULTI_THREAD_PARALLEL = "multi_thread_parallel" 
    EDGE_CENTRIC_CUGRAPH = "edge_centric_cugraph"
    HYBRID_DEGREE_BASED = "hybrid_degree_based"

class GraphFormat(Enum):
    """Graph storage formats"""
    COORDINATE_LIST = "coordinate_list"  # COO format
    COMPRESSED_SPARSE_ROW = "compressed_sparse_row"  # CSR format
    HYBRID_ADAPTIVE = "hybrid_adaptive"

class ExecutionMode(Enum):
    """GNN execution optimization modes"""
    STORAGE_OPTIMIZED = "storage_optimized"
    COMPUTE_OPTIMIZED = "compute_optimized"
    ADAPTIVE_HYBRID = "adaptive_hybrid"

@dataclass
class AccessPatternMetrics:
    """Performance metrics for access patterns"""
    storage_accesses: int = 0
    total_threads: int = 0
    bandwidth_utilization: float = 0.0
    latency_cycles: int = 0
    sq_contention_rate: float = 0.0
    
    @property
    def storage_efficiency(self) -> float:
        """Calculate storage efficiency (1 - access_ratio)"""
        if self.total_threads == 0:
            return 0.0
        return 1.0 - (self.storage_accesses / self.total_threads)
    
    @property
    def bandwidth_savings(self) -> float:
        """Calculate bandwidth savings percentage"""
        return self.storage_efficiency * 100.0

class AccessPatternSelector:
    """Selects optimal access pattern based on graph characteristics"""
    
    def __init__(self, config=None):
        self.config = config or {}
        self.degree_threshold = self.config.get('degree_threshold', 32)
        self.sparse_threshold = self.config.get('sparse_threshold', 0.005)
        self.dense_threshold = self.config.get('dense_threshold', 0.02)
        self.hub_ratio_threshold = self.config.get('hub_ratio_threshold', 0.1)
    
    def select_pattern(self, graph_characteristics: Dict[str, Any]) -> AccessPattern:
        """Select optimal access pattern based on graph characteristics"""
        avg_degree = graph_characteristics.get('avg_degree', 0)
        max_degree = graph_characteristics.get('max_degree', 0)
        hub_ratio = graph_characteristics.get('hub_ratio', 0)
        sparsity = graph_characteristics.get('sparsity', 0)
        
        # Thread 0 leader pattern for high-degree nodes or storage optimization
        if (max_degree > self.degree_threshold * 2 or 
            hub_ratio > self.hub_ratio_threshold or
            graph_characteristics.get('execution_mode') == 'storage_optimized'):
            return AccessPattern.THREAD_0_LEADER
        
        # Edge-centric pattern for sparse, uniform graphs
        elif sparsity < self.sparse_threshold and avg_degree < self.degree_threshold / 2:
            return AccessPattern.EDGE_CENTRIC_CUGRAPH
        
        # Multi-thread parallel for compute-optimized scenarios
        elif graph_characteristics.get('execution_mode') == 'compute_optimized':
            return AccessPattern.MULTI_THREAD_PARALLEL
        
        # Default to hybrid approach
        else:
            return AccessPattern.HYBRID_DEGREE_BASED
    
    def get_selection_reasoning(self, characteristics: Dict[str, Any]) -> str:
        """Provide reasoning for pattern selection"""
        pattern = self.select_pattern(characteristics)
        
        if pattern == AccessPattern.THREAD_0_LEADER:
            if characteristics.get('max_degree', 0) > self.degree_threshold * 2:
                return f"high_degree_nodes (max_degree: {characteristics['max_degree']})"
            elif characteristics.get('hub_ratio', 0) > self.hub_ratio_threshold:
                return f"hub_nodes_present (hub_ratio: {characteristics['hub_ratio']:.3f})"
            else:
                return "storage_optimization_mode"
        
        elif pattern == AccessPattern.EDGE_CENTRIC_CUGRAPH:
            return f"sparse_uniform_graph (sparsity: {characteristics.get('sparsity', 0):.4f})"
        
        elif pattern == AccessPattern.MULTI_THREAD_PARALLEL:
            return "compute_optimization_mode"
        
        else:
            return "balanced_hybrid_approach"

class PatternImplementation:
    """Base class for access pattern implementations"""
    
    def __init__(self, pattern: AccessPattern, config: Dict[str, Any] = None):
        self.pattern = pattern
        self.config = config or {}
        self.metrics = AccessPatternMetrics()
    
    def execute(self, warp_id: int, num_threads: int, message_data: Any) -> Dict[str, Any]:
        """Execute the access pattern for given warp and data"""
        raise NotImplementedError
    
    def get_metrics(self) -> AccessPatternMetrics:
        """Get current performance metrics"""
        return self.metrics
    
    def reset_metrics(self):
        """Reset performance metrics"""
        self.metrics = AccessPatternMetrics()

class Thread0LeaderImplementation(PatternImplementation):
    """Thread 0 leader access pattern implementation"""
    
    def execute(self, warp_id: int, num_threads: int, message_data: Any) -> Dict[str, Any]:
        # Only thread 0 accesses storage, broadcasts to others
        self.metrics.storage_accesses += 1
        self.metrics.total_threads += num_threads
        self.metrics.latency_cycles += self.config.get('storage_latency', 64)
        self.metrics.latency_cycles += self.config.get('broadcast_latency', 2)
        
        return {
            'storage_accesses': 1,
            'broadcast_operations': 1,
            'total_latency': self.metrics.latency_cycles
        }

class MultiThreadImplementation(PatternImplementation):
    """Multi-thread parallel access pattern implementation"""
    
    def execute(self, warp_id: int, num_threads: int, message_data: Any) -> Dict[str, Any]:
        # All threads access storage in parallel
        self.metrics.storage_accesses += num_threads
        self.metrics.total_threads += num_threads
        self.metrics.latency_cycles += self.config.get('storage_latency', 64)
        # Potential SQ contention
        if num_threads > self.config.get('sq_count', 1024):
            self.metrics.sq_contention_rate += 0.1
        
        return {
            'storage_accesses': num_threads,
            'parallel_operations': num_threads,
            'total_latency': self.metrics.latency_cycles
        }

def create_pattern_implementation(pattern: AccessPattern, config: Dict[str, Any] = None) -> PatternImplementation:
    """Factory function to create pattern implementation"""
    if pattern == AccessPattern.THREAD_0_LEADER:
        return Thread0LeaderImplementation(pattern, config)
    elif pattern == AccessPattern.MULTI_THREAD_PARALLEL:
        return MultiThreadImplementation(pattern, config)
    else:
        # Default to Thread 0 for now
        return Thread0LeaderImplementation(pattern, config)