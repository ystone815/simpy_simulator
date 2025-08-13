#!/usr/bin/env python3
import simpy
import json
from enum import Enum
from src.workloads.gnn.cugraph_integration import CuGraphInspiredGNN, GraphStorageConfig, AccessPattern, GraphFormat
from src.components.gpu.gpu_warp import Warp
from src.components.storage.nvme_doorbell_system import NVMeDoorbellController, DoorbellAccessPattern

class GNNEngineMode(Enum):
    """GNN execution engine modes"""
    STORAGE_OPTIMIZED = "storage_optimized"      # Prioritize storage bandwidth (Thread 0)
    COMPUTE_OPTIMIZED = "compute_optimized"      # Prioritize compute balance (Multi-thread)
    MEMORY_OPTIMIZED = "memory_optimized"        # Prioritize memory coalescing (Edge-centric)
    ADAPTIVE_HYBRID = "adaptive_hybrid"          # Dynamic selection based on workload

class ConfigurableGNNEngine:
    """
    Configurable GNN execution engine supporting multiple access patterns
    with runtime switching between Thread 0 and Multi-thread approaches
    """
    def __init__(self, env, engine_mode=GNNEngineMode.ADAPTIVE_HYBRID):
        self.env = env
        self.engine_mode = engine_mode
        
        # Initialize components
        self.gnn_processor = None
        self.nvme_controller = None
        self.active_warps = {}  # warp_id -> Warp object
        
        # Configuration presets
        self.config_presets = self._create_configuration_presets()
        self.current_config = self.config_presets[engine_mode]
        
        # Performance monitoring
        self.execution_history = []
        self.workload_characteristics = {}
        self.adaptive_decisions = []
        
        # Runtime switching statistics
        self.pattern_switches = {
            'thread_0_to_multi': 0,
            'multi_to_thread_0': 0,
            'to_edge_centric': 0,
            'hybrid_activations': 0
        }
        
    def _create_configuration_presets(self):
        """Create predefined configuration presets for different modes"""
        return {
            GNNEngineMode.STORAGE_OPTIMIZED: GraphStorageConfig(
                format=GraphFormat.CSR,
                access_pattern=AccessPattern.THREAD_0_LEADER,
                degree_threshold=64,
                enable_sq_lock_optimization=False
            ),
            GNNEngineMode.COMPUTE_OPTIMIZED: GraphStorageConfig(
                format=GraphFormat.COO,
                access_pattern=AccessPattern.MULTI_THREAD,
                degree_threshold=16,
                enable_sq_lock_optimization=True
            ),
            GNNEngineMode.MEMORY_OPTIMIZED: GraphStorageConfig(
                format=GraphFormat.COO,
                access_pattern=AccessPattern.EDGE_CENTRIC,
                degree_threshold=32,
                enable_sq_lock_optimization=True
            ),
            GNNEngineMode.ADAPTIVE_HYBRID: GraphStorageConfig(
                format=GraphFormat.HYBRID,
                access_pattern=AccessPattern.HYBRID_ADAPTIVE,
                degree_threshold=32,
                enable_sq_lock_optimization=True
            )
        }
    
    def initialize_engine(self, graph_config=None, nvme_config=None):
        """Initialize the GNN engine with specified configurations"""
        # Use custom config or default preset
        if graph_config:
            self.current_config = graph_config
        
        # Initialize GNN processor
        self.gnn_processor = CuGraphInspiredGNN(self.env, self.current_config)
        
        # Initialize NVMe controller
        nvme_sqs = nvme_config.get('num_sqs', 1024) if nvme_config else 1024
        self.nvme_controller = NVMeDoorbellController(
            env=self.env,
            num_submission_queues=nvme_sqs,
            sq_depth=64,
            access_pattern=DoorbellAccessPattern.THREAD_0_LEADER
        )
        
        # Connect NVMe controller to GNN processor
        self.gnn_processor.nvme_controller = self.nvme_controller
        
        print(f"GNN Engine initialized in {self.engine_mode.value} mode")
        print(f"  Graph format: {self.current_config.format.value}")
        print(f"  Access pattern: {self.current_config.access_pattern.value}")
        print(f"  Degree threshold: {self.current_config.degree_threshold}")
        print(f"  SQ lock optimization: {self.current_config.enable_sq_lock_optimization}")
    
    def load_graph(self, graph_data=None, **kwargs):
        """Load graph data into the engine"""
        if graph_data:
            # Load from provided data
            self.gnn_processor.coo_edges = graph_data['edges']
            self.gnn_processor.node_degrees = graph_data['degrees']
        else:
            # Generate synthetic graph
            num_nodes = kwargs.get('num_nodes', 1000)
            num_edges = kwargs.get('num_edges', 5000)
            distribution = kwargs.get('distribution', 'power_law')
            
            self.gnn_processor.initialize_graph(num_nodes, num_edges, distribution)
        
        # Analyze workload characteristics for adaptive decisions
        self._analyze_workload_characteristics()
    
    def _analyze_workload_characteristics(self):
        """Analyze graph characteristics to guide adaptive decisions"""
        if not self.gnn_processor.node_degrees:
            return
        
        degrees = list(self.gnn_processor.node_degrees.values())
        
        self.workload_characteristics = {
            'num_nodes': len(degrees),
            'num_edges': len(self.gnn_processor.coo_edges),
            'avg_degree': sum(degrees) / len(degrees),
            'max_degree': max(degrees),
            'min_degree': min(degrees),
            'degree_variance': self._calculate_variance(degrees),
            'hub_ratio': sum(1 for d in degrees if d > self.current_config.degree_threshold) / len(degrees),
            'sparsity': len(self.gnn_processor.coo_edges) / (len(degrees) ** 2)
        }
        
        print(f"Workload analysis:")
        for key, value in self.workload_characteristics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.3f}")
            else:
                print(f"  {key}: {value}")
    
    def _calculate_variance(self, values):
        """Calculate variance of degree distribution"""
        mean = sum(values) / len(values)
        return sum((x - mean) ** 2 for x in values) / len(values)
    
    def execute_gnn_layer(self, layer_config=None, force_pattern=None):
        """
        Execute GNN layer with configurable access patterns
        """
        layer_config = layer_config or {'num_warps': 10, 'message_type': 'aggregation'}
        
        layer_results = []
        
        # Determine access pattern for this layer
        if force_pattern:
            access_pattern = force_pattern
        else:
            access_pattern = self._decide_access_pattern(layer_config)
        
        print(f"\nExecuting GNN layer with {access_pattern.value} pattern")
        
        # Execute warps with chosen pattern
        for warp_id in range(layer_config['num_warps']):
            # Create or get warp
            if warp_id not in self.active_warps:
                warp = Warp(self.env, warp_id=warp_id, sm_id=0, num_threads=32)
                warp.enable_nvme_doorbell_optimization()
                self.active_warps[warp_id] = warp
            
            # Execute message passing with selected pattern
            warp_result = self._execute_warp_with_pattern(
                warp_id, access_pattern, layer_config
            )
            
            layer_results.append(warp_result)
        
        # Record execution for adaptive learning
        self._record_execution(access_pattern, layer_results, layer_config)
        
        return layer_results
    
    def _decide_access_pattern(self, layer_config):
        """Decide access pattern based on current engine mode and workload"""
        if self.engine_mode == GNNEngineMode.ADAPTIVE_HYBRID:
            return self._adaptive_pattern_selection(layer_config)
        else:
            return self.current_config.access_pattern
    
    def _adaptive_pattern_selection(self, layer_config):
        """Adaptive pattern selection based on workload characteristics"""
        characteristics = self.workload_characteristics
        
        # Decision tree based on graph characteristics
        if characteristics['max_degree'] > self.current_config.degree_threshold * 2:
            # Very high degree nodes - use Thread 0 for storage efficiency
            decision = AccessPattern.THREAD_0_LEADER
            reason = "high_degree_nodes"
        elif characteristics['hub_ratio'] > 0.1:
            # Many hub nodes - use Thread 0 pattern
            decision = AccessPattern.THREAD_0_LEADER
            reason = "many_hub_nodes"
        elif characteristics['degree_variance'] < 10:
            # Uniform degree distribution - use edge-centric for load balance
            decision = AccessPattern.EDGE_CENTRIC
            reason = "uniform_distribution"
        elif characteristics['sparsity'] > 0.01:
            # Dense graph - use multi-thread for parallelism
            decision = AccessPattern.MULTI_THREAD
            reason = "dense_graph"
        else:
            # Default to hybrid approach
            decision = AccessPattern.HYBRID_ADAPTIVE
            reason = "default_hybrid"
        
        # Record adaptive decision
        self.adaptive_decisions.append({
            'pattern': decision.value,
            'reason': reason,
            'characteristics': characteristics.copy(),
            'timestamp': self.env.now
        })
        
        return decision
    
    def _execute_warp_with_pattern(self, warp_id, pattern, layer_config):
        """Execute warp with specified access pattern"""
        # Temporarily set the pattern
        original_pattern = self.gnn_processor.config.access_pattern
        self.gnn_processor.config.access_pattern = pattern
        
        try:
            # Execute message passing
            results = self.gnn_processor.execute_message_passing(
                warp_id=warp_id,
                num_threads=32,
                message_data=layer_config.get('message_data')
            )
            
            # Add pattern information to results
            for result in results:
                result['selected_pattern'] = pattern.value
                result['warp_id'] = warp_id
                result['engine_mode'] = self.engine_mode.value
            
            return {
                'warp_id': warp_id,
                'pattern_used': pattern.value,
                'thread_results': results,
                'success': True
            }
            
        except Exception as e:
            return {
                'warp_id': warp_id,
                'pattern_used': pattern.value,
                'error': str(e),
                'success': False
            }
        finally:
            # Restore original pattern
            self.gnn_processor.config.access_pattern = original_pattern
    
    def _record_execution(self, pattern, results, layer_config):
        """Record execution for performance analysis and adaptive learning"""
        # Calculate performance metrics
        total_latency = 0
        successful_warps = 0
        thread_0_accesses = 0
        multi_thread_accesses = 0
        
        for warp_result in results:
            if warp_result['success']:
                successful_warps += 1
                for thread_result in warp_result['thread_results']:
                    total_latency += thread_result.get('storage_latency', 0)
                    if thread_result['access_pattern'] == 'thread_0_leader' and thread_result['thread_id'] == 0:
                        thread_0_accesses += 1
                    elif thread_result['access_pattern'] in ['multi_thread', 'edge_centric']:
                        multi_thread_accesses += 1
        
        # Record execution history
        execution_record = {
            'timestamp': self.env.now,
            'pattern': pattern.value,
            'layer_config': layer_config,
            'performance': {
                'total_latency': total_latency,
                'avg_latency_per_warp': total_latency / max(1, successful_warps),
                'successful_warps': successful_warps,
                'thread_0_accesses': thread_0_accesses,
                'multi_thread_accesses': multi_thread_accesses
            },
            'workload_state': self.workload_characteristics.copy()
        }
        
        self.execution_history.append(execution_record)
    
    def switch_access_pattern(self, new_pattern, reason="manual"):
        """Runtime switching of access patterns"""
        old_pattern = self.current_config.access_pattern
        
        if old_pattern != new_pattern:
            # Track pattern switch
            switch_key = f"{old_pattern.value}_to_{new_pattern.value}"
            if switch_key in self.pattern_switches:
                self.pattern_switches[switch_key] += 1
            
            # Update configuration
            self.current_config.access_pattern = new_pattern
            self.gnn_processor.config.access_pattern = new_pattern
            
            print(f"Access pattern switched: {old_pattern.value} → {new_pattern.value} ({reason})")
            
            return True
        return False
    
    def get_performance_report(self):
        """Generate comprehensive performance report"""
        if not self.execution_history:
            return {"error": "No execution history available"}
        
        # Aggregate performance metrics
        pattern_performance = {}
        for record in self.execution_history:
            pattern = record['pattern']
            if pattern not in pattern_performance:
                pattern_performance[pattern] = {
                    'executions': 0,
                    'total_latency': 0,
                    'total_warps': 0,
                    'thread_0_accesses': 0,
                    'multi_thread_accesses': 0
                }
            
            perf = record['performance']
            pattern_performance[pattern]['executions'] += 1
            pattern_performance[pattern]['total_latency'] += perf['total_latency']
            pattern_performance[pattern]['total_warps'] += perf['successful_warps']
            pattern_performance[pattern]['thread_0_accesses'] += perf['thread_0_accesses']
            pattern_performance[pattern]['multi_thread_accesses'] += perf['multi_thread_accesses']
        
        # Calculate efficiency metrics
        for pattern in pattern_performance:
            perf = pattern_performance[pattern]
            if perf['total_warps'] > 0:
                perf['avg_latency_per_warp'] = perf['total_latency'] / perf['total_warps']
                perf['thread_0_efficiency'] = perf['thread_0_accesses'] / (perf['thread_0_accesses'] + perf['multi_thread_accesses']) if (perf['thread_0_accesses'] + perf['multi_thread_accesses']) > 0 else 0
        
        # GNN processor stats
        gnn_stats = self.gnn_processor.get_performance_stats() if self.gnn_processor else {}
        
        # NVMe controller stats
        nvme_stats = self.nvme_controller.get_system_stats() if self.nvme_controller else {}
        
        return {
            'engine_mode': self.engine_mode.value,
            'workload_characteristics': self.workload_characteristics,
            'pattern_performance': pattern_performance,
            'pattern_switches': self.pattern_switches,
            'adaptive_decisions': len(self.adaptive_decisions),
            'gnn_processor_stats': gnn_stats,
            'nvme_controller_stats': nvme_stats,
            'execution_summary': {
                'total_executions': len(self.execution_history),
                'total_warps_processed': sum(r['performance']['successful_warps'] for r in self.execution_history)
            }
        }
    
    def benchmark_all_patterns(self, layer_config=None):
        """Benchmark all access patterns for comparison"""
        layer_config = layer_config or {'num_warps': 5, 'message_type': 'test'}
        
        patterns_to_test = [
            AccessPattern.THREAD_0_LEADER,
            AccessPattern.MULTI_THREAD,
            AccessPattern.EDGE_CENTRIC,
            AccessPattern.HYBRID_ADAPTIVE
        ]
        
        benchmark_results = {}
        
        for pattern in patterns_to_test:
            print(f"\n🔬 Benchmarking {pattern.value} pattern...")
            
            # Execute layer with this pattern
            results = self.execute_gnn_layer(layer_config, force_pattern=pattern)
            
            # Calculate metrics
            total_latency = 0
            successful_threads = 0
            storage_accesses = 0
            
            for warp_result in results:
                if warp_result['success']:
                    for thread_result in warp_result['thread_results']:
                        total_latency += thread_result.get('storage_latency', 0)
                        total_latency += thread_result.get('broadcast_latency', 0)
                        if thread_result.get('storage_latency', 0) > 0:
                            storage_accesses += 1
                        successful_threads += 1
            
            benchmark_results[pattern.value] = {
                'total_latency': total_latency,
                'avg_latency_per_thread': total_latency / max(1, successful_threads),
                'storage_accesses': storage_accesses,
                'threads_processed': successful_threads,
                'storage_efficiency': (successful_threads - storage_accesses) / max(1, successful_threads)
            }
            
            print(f"   Avg latency/thread: {benchmark_results[pattern.value]['avg_latency_per_thread']:.2f}")
            print(f"   Storage accesses: {storage_accesses}/{successful_threads}")
            print(f"   Storage efficiency: {benchmark_results[pattern.value]['storage_efficiency']:.1%}")
        
        return benchmark_results

# Factory function for easy engine creation
def create_gnn_engine(mode="adaptive", graph_config=None, nvme_config=None):
    """Factory function to create configured GNN engine"""
    env = simpy.Environment()
    
    mode_map = {
        "storage": GNNEngineMode.STORAGE_OPTIMIZED,
        "compute": GNNEngineMode.COMPUTE_OPTIMIZED, 
        "memory": GNNEngineMode.MEMORY_OPTIMIZED,
        "adaptive": GNNEngineMode.ADAPTIVE_HYBRID
    }
    
    engine_mode = mode_map.get(mode, GNNEngineMode.ADAPTIVE_HYBRID)
    engine = ConfigurableGNNEngine(env, engine_mode)
    engine.initialize_engine(graph_config, nvme_config)
    
    return engine, env