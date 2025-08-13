"""Unified GNN engine combining configurable and cuGraph approaches"""
import simpy
import random
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from config.gnn_config import GNNConfig, AccessPattern, GraphFormat, ExecutionMode
from .access_patterns import AccessPatternSelector, create_pattern_implementation
from .cugraph_integration import CuGraphInspiredGNN

@dataclass
class GraphCharacteristics:
    """Graph analysis results"""
    num_nodes: int
    num_edges: int
    avg_degree: float
    max_degree: int
    min_degree: int
    degree_variance: float
    hub_ratio: float
    sparsity: float

class UnifiedGNNEngine:
    """Unified GNN engine with configurable patterns and cuGraph integration"""
    
    def __init__(self, env: simpy.Environment, config: GNNConfig):
        self.env = env
        self.config = config
        self.current_pattern = config.access_pattern.pattern
        self.pattern_selector = AccessPatternSelector(config.access_pattern.__dict__)
        self.cugraph_engine = CuGraphInspiredGNN(env, config)
        
        # Graph data structures
        self.adjacency_list = {}
        self.edge_list = []
        self.node_degrees = {}
        
        # Performance tracking
        self.performance_metrics = {}
        self.decision_history = []
        
        # Initialize graph
        self._initialize_graph()
        self.graph_characteristics = self._analyze_graph()
    
    def _initialize_graph(self):
        """Initialize graph structure based on configuration"""
        num_nodes = self.config.graph.num_nodes
        num_edges = self.config.graph.num_edges
        
        # Generate random graph (simplified for demo)
        random.seed(self.config.graph.__dict__.get('seed', 42))
        
        for i in range(num_nodes):
            self.adjacency_list[i] = []
            self.node_degrees[i] = 0
        
        edges_created = 0
        while edges_created < num_edges:
            src = random.randint(0, num_nodes - 1)
            dst = random.randint(0, num_nodes - 1)
            
            if src != dst and dst not in self.adjacency_list[src]:
                self.adjacency_list[src].append(dst)
                self.adjacency_list[dst].append(src)  # Undirected graph
                self.edge_list.append((src, dst))
                self.node_degrees[src] += 1
                self.node_degrees[dst] += 1
                edges_created += 1
    
    def _analyze_graph(self) -> GraphCharacteristics:
        """Analyze graph characteristics for pattern selection"""
        degrees = list(self.node_degrees.values())
        non_zero_degrees = [d for d in degrees if d > 0]
        
        if not non_zero_degrees:
            return GraphCharacteristics(0, 0, 0, 0, 0, 0, 0, 0)
        
        avg_degree = sum(non_zero_degrees) / len(non_zero_degrees)
        max_degree = max(non_zero_degrees)
        min_degree = min(non_zero_degrees)
        
        # Calculate variance
        variance = sum((d - avg_degree) ** 2 for d in non_zero_degrees) / len(non_zero_degrees)
        
        # Hub nodes (high degree nodes)
        hub_threshold = self.config.access_pattern.degree_threshold
        hub_nodes = sum(1 for d in non_zero_degrees if d > hub_threshold)
        hub_ratio = hub_nodes / len(non_zero_degrees) if non_zero_degrees else 0
        
        # Sparsity
        max_edges = self.config.graph.num_nodes * (self.config.graph.num_nodes - 1) / 2
        sparsity = len(self.edge_list) / max_edges if max_edges > 0 else 0
        
        return GraphCharacteristics(
            num_nodes=len([d for d in degrees if d > 0]),
            num_edges=len(self.edge_list),
            avg_degree=avg_degree,
            max_degree=max_degree,
            min_degree=min_degree,
            degree_variance=variance,
            hub_ratio=hub_ratio,
            sparsity=sparsity
        )
    
    def switch_access_pattern(self, new_pattern: AccessPattern, reason: str = ""):
        """Switch to a different access pattern"""
        if new_pattern != self.current_pattern:
            print(f"Access pattern switched: {self.current_pattern.value} → {new_pattern.value} ({reason})")
            self.current_pattern = new_pattern
            self.decision_history.append({
                'from_pattern': self.current_pattern.value,
                'to_pattern': new_pattern.value,
                'reason': reason,
                'timestamp': self.env.now
            })
    
    def execute_gnn_layer(self, layer_config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a GNN layer with current access pattern"""
        num_warps = layer_config.get('num_warps', self.config.num_warps)
        
        # Adaptive pattern selection if enabled
        if self.config.execution_mode == ExecutionMode.ADAPTIVE_HYBRID:
            characteristics = {
                **self.graph_characteristics.__dict__,
                'execution_mode': self.config.execution_mode.value
            }
            
            optimal_pattern = self.pattern_selector.select_pattern(characteristics)
            if optimal_pattern != self.current_pattern:
                reason = self.pattern_selector.get_selection_reasoning(characteristics)
                self.switch_access_pattern(optimal_pattern, reason)
        
        # Execute with current pattern
        if self.current_pattern in [AccessPattern.EDGE_CENTRIC_CUGRAPH]:
            # Use cuGraph-inspired implementation
            return self.cugraph_engine.execute_layer(num_warps, self.current_pattern.value)
        else:
            # Use standard implementation
            return self._execute_standard_pattern(num_warps, layer_config)
    
    def _execute_standard_pattern(self, num_warps: int, layer_config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute using standard access patterns"""
        start_time = self.env.now
        successful_warps = 0
        total_latency = 0
        storage_accesses = 0
        
        pattern_impl = create_pattern_implementation(
            self.current_pattern,
            {
                'storage_latency': 64,
                'broadcast_latency': 2,
                'sq_count': 1024
            }
        )
        
        for warp_id in range(num_warps):
            # Simulate warp execution
            try:
                result = pattern_impl.execute(warp_id, 32, None)  # 32 threads per warp
                successful_warps += 1
                total_latency += result.get('total_latency', 0)
                storage_accesses += result.get('storage_accesses', 0)
                
                # Simulate execution time
                yield self.env.timeout(random.uniform(1, 5))
                
            except Exception as e:
                print(f"Warp {warp_id} execution failed: {e}")
        
        execution_time = (self.env.now - start_time) / 1000.0  # Convert to seconds
        
        # Calculate metrics
        metrics = pattern_impl.get_metrics()
        
        return {
            'execution_time': execution_time,
            'successful_warps': successful_warps,
            'total_warps': num_warps,
            'avg_latency_per_warp': total_latency / max(successful_warps, 1),
            'storage_accesses': storage_accesses,
            'storage_efficiency': metrics.storage_efficiency * 100,
            'pattern_used': self.current_pattern.value
        }
    
    def run_benchmark_suite(self, benchmark_configs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run comprehensive benchmark suite"""
        results = {}
        
        for config in benchmark_configs:
            test_name = config.get('name', f'test_{len(results)}')
            print(f"\n🔬 Benchmarking {test_name}...")
            
            # Temporarily switch pattern if specified
            original_pattern = self.current_pattern
            if 'pattern' in config:
                pattern = AccessPattern(config['pattern'])
                self.switch_access_pattern(pattern, f"benchmark_{test_name}")
            
            # Execute test
            try:
                result = yield from self.execute_gnn_layer(config)
                results[test_name] = result
                
                print(f"   Avg latency/thread: {result['avg_latency_per_warp']:.2f}")
                print(f"   Storage accesses: {result['storage_accesses']}/{config.get('num_warps', 8) * 32}")
                print(f"   Storage efficiency: {result['storage_efficiency']:.1f}%")
                
            except Exception as e:
                print(f"   Benchmark failed: {e}")
                results[test_name] = {'error': str(e)}
            
            # Restore original pattern
            if 'pattern' in config:
                self.current_pattern = original_pattern
        
        return results
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        return {
            'graph_characteristics': self.graph_characteristics.__dict__,
            'current_pattern': self.current_pattern.value,
            'decision_history': self.decision_history,
            'performance_metrics': self.performance_metrics,
            'config_summary': {
                'execution_mode': self.config.execution_mode.value,
                'graph_format': self.config.graph_format.value,
                'num_layers': self.config.num_layers,
                'num_warps': self.config.num_warps
            }
        }