"""Benchmark configuration classes"""
from dataclasses import dataclass
from typing import List, Dict, Optional, Any
import yaml

@dataclass
class TestConfig:
    """Individual test configuration"""
    name: str = "default_test"
    description: str = ""
    enabled: bool = True
    timeout_seconds: int = 300
    expected_results: Optional[Dict[str, Any]] = None
    
@dataclass
class BenchmarkConfig:
    """Benchmark suite configuration"""
    name: str = "gpu_simulation_benchmark"
    version: str = "1.0"
    description: str = "Comprehensive GPU simulation benchmarking"
    
    # Test categories
    run_unit_tests: bool = True
    run_integration_tests: bool = True
    run_performance_tests: bool = True
    
    # Performance test parameters
    workload_sizes: List[int] = None
    sq_counts: List[int] = None
    graph_types: List[str] = None
    access_patterns: List[str] = None
    
    # Output configuration  
    save_results: bool = True
    results_directory: str = "results"
    generate_plots: bool = False
    verbose_output: bool = True
    
    # Resource limits
    max_memory_gb: int = 16
    max_cpu_cores: int = 8
    parallel_tests: bool = False
    
    def __post_init__(self):
        if self.workload_sizes is None:
            self.workload_sizes = [5, 10, 20, 40]
            
        if self.sq_counts is None:
            self.sq_counts = [32, 64, 128, 256, 512, 1024]
            
        if self.graph_types is None:
            self.graph_types = [
                "sparse_uniform", 
                "dense_uniform", 
                "power_law_hubs", 
                "random_mixed"
            ]
            
        if self.access_patterns is None:
            self.access_patterns = [
                "thread_0_leader",
                "multi_thread_parallel", 
                "edge_centric_cugraph",
                "hybrid_degree_based"
            ]

@dataclass
class GNNBenchmarkConfig(BenchmarkConfig):
    """GNN-specific benchmark configuration"""
    name: str = "gnn_benchmark"
    
    # GNN-specific parameters
    node_counts: List[int] = None
    edge_ratios: List[float] = None
    layer_counts: List[int] = None
    
    # Performance thresholds
    min_storage_efficiency: float = 0.95
    max_sq_contention_rate: float = 0.05
    min_throughput_ops_per_sec: int = 1000
    
    def __post_init__(self):
        super().__post_init__()
        if self.node_counts is None:
            self.node_counts = [160, 320, 640, 1280]
        if self.edge_ratios is None:
            self.edge_ratios = [2.5, 5.0, 10.0, 20.0]  # edges per node
        if self.layer_counts is None:
            self.layer_counts = [1, 3, 5]

@dataclass
class ComparisonBenchmarkConfig(BenchmarkConfig):
    """Configuration for comparing different approaches"""
    name: str = "comparison_benchmark"
    
    # Comparison targets
    gpu_architectures: List[str] = None
    optimization_levels: List[str] = None
    baseline_config: str = "default"
    
    # Statistical analysis
    num_runs_per_config: int = 5
    confidence_interval: float = 0.95
    statistical_significance_threshold: float = 0.05
    
    def __post_init__(self):
        super().__post_init__()
        if self.gpu_architectures is None:
            self.gpu_architectures = ["H100", "B200"]
        if self.optimization_levels is None:
            self.optimization_levels = ["none", "thread_0", "hybrid"]

def load_benchmark_config(config_path: str, benchmark_type: str = "default") -> BenchmarkConfig:
    """Load benchmark configuration from YAML file"""
    with open(config_path, 'r') as f:
        data = yaml.safe_load(f)
    
    benchmark_data = data.get('benchmark', {})
    
    if benchmark_type == "gnn":
        return GNNBenchmarkConfig(**benchmark_data)
    elif benchmark_type == "comparison":
        return ComparisonBenchmarkConfig(**benchmark_data)
    else:
        return BenchmarkConfig(**benchmark_data)