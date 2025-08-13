"""GNN workload configuration classes"""
from dataclasses import dataclass
from typing import List, Dict, Optional
from enum import Enum
import yaml

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
class GraphConfig:
    """Graph structure configuration"""
    num_nodes: int = 1000
    num_edges: int = 5000
    graph_type: str = "random"  # random, power_law, small_world, etc.
    average_degree: float = 10.0
    degree_distribution: str = "uniform"  # uniform, power_law, normal
    hub_threshold: int = 32
    sparsity_ratio: float = 0.01
    
    @property
    def expected_edges(self) -> int:
        """Expected number of edges based on nodes and average degree"""
        return int(self.num_nodes * self.average_degree / 2)

@dataclass
class AccessPatternConfig:
    """Access pattern specific configuration"""
    pattern: AccessPattern = AccessPattern.HYBRID_DEGREE_BASED
    degree_threshold: int = 32
    enable_thread0_optimization: bool = True
    enable_warp_shuffle: bool = True
    sq_lock_optimization: bool = True
    
    # Hybrid pattern specific settings
    sparse_threshold: float = 0.005
    dense_threshold: float = 0.02
    hub_ratio_threshold: float = 0.1

@dataclass
class GNNConfig:
    """Complete GNN workload configuration"""
    execution_mode: ExecutionMode = ExecutionMode.ADAPTIVE_HYBRID
    graph_format: GraphFormat = GraphFormat.HYBRID_ADAPTIVE
    access_pattern: AccessPatternConfig = None
    graph: GraphConfig = None
    
    # Layer configuration
    num_layers: int = 3
    hidden_dimensions: List[int] = None
    message_size_bytes: int = 256
    
    # Performance settings
    batch_size: int = 1
    num_warps: int = 8
    enable_memory_coalescing: bool = True
    
    def __post_init__(self):
        if self.access_pattern is None:
            self.access_pattern = AccessPatternConfig()
        if self.graph is None:
            self.graph = GraphConfig()
        if self.hidden_dimensions is None:
            self.hidden_dimensions = [256, 128, 64]

@dataclass
class StorageConfig:
    """GNN storage system configuration"""
    kv_cache_size_gb: float = 8.0
    vector_db_dimension: int = 512
    graph_storage_format: str = "adjacency_list"
    compression_enabled: bool = True
    cache_policy: str = "lru"  # lru, lfu, adaptive
    
    # Thread 0 optimization settings
    broadcast_latency_cycles: int = 2
    storage_access_latency_cycles: int = 64
    bandwidth_reduction_factor: float = 0.969  # 96.9% reduction

def load_gnn_config(config_path: str) -> GNNConfig:
    """Load GNN configuration from YAML file"""
    with open(config_path, 'r') as f:
        data = yaml.safe_load(f)
    
    gnn_data = data.get('gnn', {})
    
    # Parse enums
    if 'execution_mode' in gnn_data:
        gnn_data['execution_mode'] = ExecutionMode(gnn_data['execution_mode'])
    if 'graph_format' in gnn_data:
        gnn_data['graph_format'] = GraphFormat(gnn_data['graph_format'])
    
    # Parse nested configurations
    if 'access_pattern' in gnn_data:
        pattern_data = gnn_data['access_pattern']
        if 'pattern' in pattern_data:
            pattern_data['pattern'] = AccessPattern(pattern_data['pattern'])
        gnn_data['access_pattern'] = AccessPatternConfig(**pattern_data)
    
    if 'graph' in gnn_data:
        gnn_data['graph'] = GraphConfig(**gnn_data['graph'])
    
    return GNNConfig(**gnn_data)