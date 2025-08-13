import simpy
import random
from enum import Enum
from src.components.gpu.streaming_multiprocessor import StreamingMultiprocessor, SMCluster
from src.components.gpu.gpu_memory_hierarchy import MemoryHierarchy
from src.components.gpu.h100_gpu import TensorPrecision, TransformerEngine
from src.components.gpu.gpu_warp import WarpInstruction
from src.base.packet import Packet

class ChipletInterconnect:
    """
    B200 dual chiplet interconnect - 10 TB/s custom link for cache coherency.
    """
    def __init__(self, env, gpu_id):
        self.env = env
        self.gpu_id = gpu_id
        self.bandwidth_tbps = 10.0  # 10 TB/s interconnect
        self.latency_cycles = 5  # Inter-chiplet latency
        
        # Traffic tracking
        self.data_transferred = 0
        self.total_transfers = 0
        self.congestion_events = 0
        
        # Interconnect channels
        self.chiplet0_to_chiplet1 = simpy.Store(env, capacity=16)
        self.chiplet1_to_chiplet0 = simpy.Store(env, capacity=16)
        
        self.action = env.process(self.run())
    
    def run(self):
        """Handle inter-chiplet communication"""
        while True:
            # Process transfers in both directions
            if self.chiplet0_to_chiplet1.items:
                packet = yield self.chiplet0_to_chiplet1.get()
                yield self.env.process(self.handle_transfer(packet, "0->1"))
            
            if self.chiplet1_to_chiplet0.items:
                packet = yield self.chiplet1_to_chiplet0.get()
                yield self.env.process(self.handle_transfer(packet, "1->0"))
            
            yield self.env.timeout(1)  # Check every cycle
    
    def handle_transfer(self, packet, direction):
        """Handle data transfer between chiplets"""
        transfer_size = packet.size
        transfer_cycles = max(1, transfer_size / (self.bandwidth_tbps * 1024**4 / 8 / 1000))  # Convert to cycles
        
        # Add base latency
        yield self.env.timeout(self.latency_cycles + transfer_cycles)
        
        self.data_transferred += transfer_size
        self.total_transfers += 1
        
        # Check for congestion
        if (len(self.chiplet0_to_chiplet1.items) + len(self.chiplet1_to_chiplet0.items)) > 12:
            self.congestion_events += 1
    
    def send_to_chiplet1(self, packet):
        """Send packet to chiplet 1"""
        try:
            self.chiplet0_to_chiplet1.put(packet)
            return True
        except:
            return False  # Channel full
    
    def send_to_chiplet0(self, packet):
        """Send packet to chiplet 0"""
        try:
            self.chiplet1_to_chiplet0.put(packet)
            return True
        except:
            return False  # Channel full
    
    def get_stats(self):
        """Get interconnect performance statistics"""
        bandwidth_utilization = 0
        if self.env.now > 0:
            actual_bandwidth = self.data_transferred / self.env.now
            max_bandwidth = self.bandwidth_tbps * 1024**4 / 8  # Bytes per ns
            bandwidth_utilization = actual_bandwidth / max_bandwidth
        
        return {
            'bandwidth_tbps': self.bandwidth_tbps,
            'data_transferred': self.data_transferred,
            'total_transfers': self.total_transfers,
            'bandwidth_utilization': bandwidth_utilization,
            'congestion_events': self.congestion_events,
            'latency_cycles': self.latency_cycles
        }


class ShaderExecutionReordering2:
    """
    SER 2.0 (Shader Execution Reordering) - optimizes how warps are arranged and executed.
    Improves coherency and reduces divergence in ray tracing and AI workloads.
    """
    def __init__(self, env, sm_id):
        self.env = env
        self.sm_id = sm_id
        
        # Reordering queues
        self.coherent_warps = []     # Warps with similar execution paths
        self.divergent_warps = []    # Warps with divergent execution
        self.reordered_warps = []    # Optimally reordered warps
        
        # Reordering policies
        self.coherence_threshold = 0.7  # Minimum coherence for grouping
        self.reorder_window_size = 16   # Number of warps to consider
        
        # Performance tracking
        self.reordering_operations = 0
        self.coherence_improvements = 0
        self.divergence_reductions = 0
        
    def analyze_warp_coherence(self, warp):
        """Analyze coherence of warp execution paths"""
        active_threads = warp.thread_context.get_active_threads()
        if not active_threads:
            return 0.0
        
        # Check program counter coherence
        pcs = [thread.program_counter for thread in active_threads]
        unique_pcs = len(set(pcs))
        coherence = 1.0 - (unique_pcs - 1) / len(active_threads)
        
        return coherence
    
    def reorder_warps_for_coherence(self, warp_list):
        """Reorder warps to improve execution coherence"""
        if len(warp_list) < 2:
            return warp_list
        
        self.reordering_operations += 1
        
        # Group warps by coherence characteristics
        coherence_groups = {}
        
        for warp in warp_list:
            coherence = self.analyze_warp_coherence(warp)
            
            # Quantize coherence into groups
            coherence_bucket = int(coherence * 10) / 10
            
            if coherence_bucket not in coherence_groups:
                coherence_groups[coherence_bucket] = []
            coherence_groups[coherence_bucket].append(warp)
        
        # Reorder: high coherence groups first
        reordered = []
        for coherence_level in sorted(coherence_groups.keys(), reverse=True):
            # Further sort within group by program counter similarity
            group_warps = coherence_groups[coherence_level]
            group_warps.sort(key=lambda w: w.program_counter)
            reordered.extend(group_warps)
        
        # Track improvement
        if len(coherence_groups) > 1:
            self.coherence_improvements += 1
        
        return reordered
    
    def reduce_divergence(self, warp):
        """Apply divergence reduction optimizations"""
        if warp.divergence_stack:
            # Try to merge divergent paths
            original_divergence = len(warp.divergence_stack)
            
            # Simplified divergence reduction
            if original_divergence > 1:
                # Merge similar divergence paths
                merged_stack = []
                current_divergence = warp.divergence_stack[0]
                
                for divergence in warp.divergence_stack[1:]:
                    if abs(divergence['pc'] - current_divergence['pc']) < 4:
                        # Merge similar program counters
                        current_divergence['true_threads'].extend(divergence['true_threads'])
                        current_divergence['false_threads'].extend(divergence['false_threads'])
                    else:
                        merged_stack.append(current_divergence)
                        current_divergence = divergence
                
                merged_stack.append(current_divergence)
                warp.divergence_stack = merged_stack
                
                if len(merged_stack) < original_divergence:
                    self.divergence_reductions += 1
    
    def optimize_warp_scheduling(self, ready_warps):
        """Optimize warp scheduling using SER 2.0"""
        if len(ready_warps) <= 1:
            return ready_warps
        
        # Apply reordering for coherence
        reordered_warps = self.reorder_warps_for_coherence(ready_warps)
        
        # Apply divergence reduction
        for warp in reordered_warps:
            self.reduce_divergence(warp)
        
        return reordered_warps
    
    def get_stats(self):
        """Get SER 2.0 performance statistics"""
        return {
            'sm_id': self.sm_id,
            'reordering_operations': self.reordering_operations,
            'coherence_improvements': self.coherence_improvements,
            'divergence_reductions': self.divergence_reductions,
            'coherence_threshold': self.coherence_threshold
        }


class AdvancedTensorCore:
    """
    B200 Advanced Tensor Core with FP4 support and enhanced throughput.
    Provides ~5x improvement over H100 for AI inference.
    """
    def __init__(self, env, core_id, sm_id):
        self.env = env
        self.core_id = core_id
        self.sm_id = sm_id
        
        # Enhanced precision support including FP4
        self.precision_configs = {
            (TensorPrecision.FP32, TensorPrecision.FP32): {"throughput": 624, "latency": 1},
            (TensorPrecision.FP16, TensorPrecision.FP16): {"throughput": 2500, "latency": 1},  # 5x improvement
            (TensorPrecision.BF16, TensorPrecision.BF16): {"throughput": 2500, "latency": 1},
            (TensorPrecision.FP8, TensorPrecision.FP8): {"throughput": 5000, "latency": 1},   # 5x improvement
            (TensorPrecision.INT8, TensorPrecision.INT8): {"throughput": 5000, "latency": 1},
            (TensorPrecision.FP4, TensorPrecision.FP4): {"throughput": 10000, "latency": 1},  # New FP4 support
            (TensorPrecision.INT4, TensorPrecision.INT4): {"throughput": 10000, "latency": 1},
        }
        
        # Enhanced sparsity support
        self.sparsity_patterns = {
            "2:4": {"speedup": 2.0, "accuracy_retention": 0.95},
            "4:8": {"speedup": 1.8, "accuracy_retention": 0.98},
            "8:16": {"speedup": 1.5, "accuracy_retention": 0.99}
        }
        
        # Performance tracking
        self.operations_completed = 0
        self.sparse_operations = 0
        self.total_cycles = 0
        self.utilization_time = 0
        
        # Supported matrix shapes (enhanced for B200)
        self.supported_shapes = [
            (16, 16, 16), (32, 8, 16), (8, 32, 16),  # H100 shapes
            (64, 8, 16), (8, 64, 16), (32, 16, 32),  # Enhanced B200 shapes
            (64, 32, 64), (128, 64, 128), (256, 128, 256)  # Large shapes for big models
        ]
    
    def execute_sparse_mma(self, shape, precision_a, precision_b, sparsity_pattern=None, precision_c=TensorPrecision.FP32):
        """Execute sparse Matrix Multiply-Accumulate with sparsity optimization"""
        if shape not in self.supported_shapes:
            raise ValueError(f"Unsupported tensor shape: {shape}")
        
        precision_key = (precision_a, precision_b)
        if precision_key not in self.precision_configs:
            raise ValueError(f"Unsupported precision combination: {precision_key}")
        
        config = self.precision_configs[precision_key]
        base_execution_cycles = config["latency"]
        
        # Apply sparsity optimization
        if sparsity_pattern and sparsity_pattern in self.sparsity_patterns:
            sparsity_info = self.sparsity_patterns[sparsity_pattern]
            execution_cycles = base_execution_cycles / sparsity_info["speedup"]
            self.sparse_operations += 1
        else:
            execution_cycles = base_execution_cycles
        
        # Simulate execution
        start_time = self.env.now
        yield self.env.timeout(execution_cycles)
        
        self.operations_completed += 1
        self.utilization_time += execution_cycles
        self.total_cycles += execution_cycles
        
        # Calculate enhanced throughput
        m, n, k = shape
        operations = 2 * m * n * k
        effective_throughput = operations / execution_cycles
        
        return {
            'operations': operations,
            'cycles': execution_cycles,
            'throughput_ops_per_cycle': effective_throughput,
            'precision_config': precision_key,
            'sparsity_pattern': sparsity_pattern,
            'sparse_optimized': sparsity_pattern is not None
        }
    
    def get_utilization(self):
        """Get advanced tensor core utilization"""
        if self.total_cycles == 0:
            return 0
        return self.utilization_time / self.total_cycles
    
    def get_sparse_ratio(self):
        """Get ratio of sparse operations"""
        if self.operations_completed == 0:
            return 0
        return self.sparse_operations / self.operations_completed
    
    def get_stats(self):
        """Get advanced tensor core statistics"""
        return {
            'core_id': self.core_id,
            'sm_id': self.sm_id,
            'operations_completed': self.operations_completed,
            'sparse_operations': self.sparse_operations,
            'sparse_ratio': self.get_sparse_ratio(),
            'utilization': self.get_utilization(),
            'total_cycles': self.total_cycles,
            'fp4_supported': True
        }


class B200StreamingMultiprocessor(StreamingMultiprocessor):
    """
    B200-specific Streaming Multiprocessor with dual chiplet, SER 2.0, and advanced tensor cores.
    """
    def __init__(self, env, sm_id, chiplet_id=0):
        super().__init__(env, sm_id, gpu_type="B200")
        
        self.chiplet_id = chiplet_id
        
        # B200-specific components
        self.advanced_tensor_cores = []
        for i in range(4):  # 4 enhanced Tensor Cores per SM
            tensor_core = AdvancedTensorCore(env, core_id=i, sm_id=sm_id)
            self.advanced_tensor_cores.append(tensor_core)
        
        # SER 2.0 - Shader Execution Reordering
        self.ser_2 = ShaderExecutionReordering2(env, sm_id)
        
        # Enhanced Transformer Engine (inherited from H100 but improved)
        self.transformer_engine = TransformerEngine(env, engine_id=0, sm_id=sm_id)
        
        # Override warp schedulers with SER 2.0 optimization
        self._enhance_warp_schedulers()
        
        # Enhanced specifications
        self.max_warps = 64  # Same as H100 for compute capability 10.0
        self.shared_memory_size_kb = 228  # Same as H100
        self.register_file_size = 65536
        
        # Inter-chiplet communication capability
        self.chiplet_interconnect = None  # Will be set by GPU
    
    def _enhance_warp_schedulers(self):
        """Enhance warp schedulers with SER 2.0"""
        for scheduler in self.warp_schedulers:
            original_select = scheduler._select_next_warp
            
            def enhanced_select(self=scheduler):
                ready_warps = list(self.ready_warps)
                if len(ready_warps) > 1:
                    # Apply SER 2.0 optimization
                    optimized_warps = self.sm_id.ser_2.optimize_warp_scheduling(ready_warps)
                    self.ready_warps.clear()
                    self.ready_warps.extend(optimized_warps)
                
                return original_select()
            
            scheduler._select_next_warp = enhanced_select
    
    def execute_advanced_tensor_operation(self, shape, precision_a, precision_b, sparsity_pattern=None):
        """Execute tensor operation using Advanced Tensor Core with sparsity"""
        # Select least utilized tensor core
        tensor_core = min(self.advanced_tensor_cores, key=lambda tc: tc.utilization_time)
        
        # Execute sparse MMA operation
        result = yield self.env.process(
            tensor_core.execute_sparse_mma(shape, precision_a, precision_b, sparsity_pattern)
        )
        
        return result
    
    def process_inference_workload(self, model_config):
        """Process AI inference workload optimized for B200"""
        batch_size = model_config.get('batch_size', 1)
        seq_length = model_config['seq_length']
        hidden_dim = model_config['hidden_dim']
        num_layers = model_config.get('num_layers', 12)
        
        total_cycles = 0
        inference_results = []
        
        # Process each layer with B200 optimizations
        for layer_id in range(num_layers):
            layer_config = {
                'layer_id': layer_id,
                'seq_length': seq_length,
                'hidden_dim': hidden_dim,
                'batch_size': batch_size
            }
            
            # Use FP8 or FP4 for inference optimization
            if hidden_dim > 4096:
                precision = TensorPrecision.FP4  # Ultra-low precision for large models
                sparsity = "2:4"  # 2:4 sparsity pattern
            else:
                precision = TensorPrecision.FP8
                sparsity = "4:8"
            
            # Execute optimized tensor operations with supported shapes
            # Use smaller tile sizes for large matrices
            if hidden_dim <= 256:
                shape = (min(seq_length, 256), min(hidden_dim, 128), min(hidden_dim, 256))
            else:
                shape = (64, 32, 64)  # Use supported large shape
            
            tensor_result = yield self.env.process(
                self.execute_advanced_tensor_operation(
                    shape=shape,
                    precision_a=precision,
                    precision_b=precision,
                    sparsity_pattern=sparsity
                )
            )
            
            inference_results.append(tensor_result)
            total_cycles += tensor_result['cycles']
        
        # Calculate inference performance metrics
        tokens_per_second = (batch_size * seq_length * 1000) / max(total_cycles, 1)
        
        return {
            'model_config': model_config,
            'total_cycles': total_cycles,
            'tokens_per_second': tokens_per_second,
            'avg_precision': 'FP4/FP8',
            'sparsity_utilized': True,
            'inference_results': inference_results
        }
    
    def get_b200_specific_stats(self):
        """Get B200-specific performance statistics"""
        base_stats = self.get_performance_stats()
        
        b200_stats = {
            'chiplet_id': self.chiplet_id,
            'advanced_tensor_cores': [tc.get_stats() for tc in self.advanced_tensor_cores],
            'ser_2_stats': self.ser_2.get_stats(),
            'transformer_engine': self.transformer_engine.get_stats(),
            'fp4_capable': True,
            'sparsity_support': True
        }
        
        base_stats.update(b200_stats)
        return base_stats


class B200GPU:
    """
    Complete NVIDIA B200 GPU model with dual chiplet design, 192GB HBM3E, and 8TB/s bandwidth.
    """
    def __init__(self, env, gpu_id=0):
        self.env = env
        self.gpu_id = gpu_id
        self.gpu_type = "B200"
        
        # B200 Dual Chiplet Architecture
        self.num_chiplets = 2
        self.sms_per_chiplet = 72  # Estimated, 144 total SMs across 2 chiplets
        self.total_sms = 144
        self.hbm3e_memory_gb = 192
        self.hbm3e_bandwidth_tbps = 8.0
        self.transistor_count = 208_000_000_000  # 208 billion transistors
        
        # Inter-chiplet communication
        self.chiplet_interconnect = ChipletInterconnect(env, gpu_id)
        
        # Create dual chiplet design
        self.chiplet_0_sms = []
        self.chiplet_1_sms = []
        
        # Chiplet 0 SMs
        for i in range(self.sms_per_chiplet):
            sm = B200StreamingMultiprocessor(env, sm_id=i, chiplet_id=0)
            sm.chiplet_interconnect = self.chiplet_interconnect
            self.chiplet_0_sms.append(sm)
        
        # Chiplet 1 SMs
        for i in range(self.sms_per_chiplet, self.total_sms):
            sm = B200StreamingMultiprocessor(env, sm_id=i, chiplet_id=1)
            sm.chiplet_interconnect = self.chiplet_interconnect
            self.chiplet_1_sms.append(sm)
        
        # All SMs combined
        self.all_sms = self.chiplet_0_sms + self.chiplet_1_sms
        
        # Memory hierarchy (enhanced for B200)
        self.memory_hierarchy = MemoryHierarchy(env, gpu_type="B200")
        
        # Create memory subsystems
        for sm in self.all_sms:
            l1_cache, shared_mem, register_file = self.memory_hierarchy.create_sm_memory_subsystem(sm.sm_id)
            sm.connect_memory_subsystem(l1_cache, shared_mem, register_file)
        
        # Enhanced L2 cache
        self.memory_hierarchy.create_l2_cache(size_mb=100)  # Larger L2 for B200
        
        # NVLink 5th generation
        self.nvlink_bandwidth_gbps = 1800  # 1.8 TB/s per direction
        self.nvlink_ports = 18
        
        # Performance tracking
        self.total_operations = 0
        self.total_runtime = 0
        self.chiplet_load_balance = {'chiplet_0': 0, 'chiplet_1': 0}
    
    def launch_distributed_kernel(self, grid_size, block_size, kernel_instructions, chiplet_preference=None):
        """Launch CUDA kernel with dual chiplet load balancing"""
        start_time = self.env.now
        
        total_blocks = grid_size[0] * grid_size[1] * grid_size[2]
        threads_per_block = block_size[0] * block_size[1] * block_size[2]
        warps_per_block = (threads_per_block + 31) // 32
        
        blocks_launched = 0
        
        # Distribute blocks across chiplets
        chiplet_0_blocks = 0
        chiplet_1_blocks = 0
        
        for block_id in range(total_blocks):
            # Load balancing between chiplets
            if chiplet_preference == 0:
                target_chiplet = 0
            elif chiplet_preference == 1:
                target_chiplet = 1
            else:
                # Automatic load balancing
                chiplet_0_load = sum(len(sm.active_warps) for sm in self.chiplet_0_sms)
                chiplet_1_load = sum(len(sm.active_warps) for sm in self.chiplet_1_sms)
                target_chiplet = 0 if chiplet_0_load <= chiplet_1_load else 1
            
            # Select target SMs
            target_sms = self.chiplet_0_sms if target_chiplet == 0 else self.chiplet_1_sms
            
            # Launch warps on least loaded SM in target chiplet
            sm = min(target_sms, key=lambda s: len(s.active_warps))
            
            warps_launched = 0
            for warp_id in range(warps_per_block):
                if sm.launch_warp(kernel_instructions):
                    warps_launched += 1
                else:
                    break
            
            if warps_launched == warps_per_block:
                blocks_launched += 1
                if target_chiplet == 0:
                    chiplet_0_blocks += 1
                else:
                    chiplet_1_blocks += 1
                
                self.chiplet_load_balance[f'chiplet_{target_chiplet}'] += 1
            else:
                break
        
        execution_time = self.env.now - start_time
        self.total_runtime += execution_time
        
        return {
            'blocks_launched': blocks_launched,
            'chiplet_0_blocks': chiplet_0_blocks,
            'chiplet_1_blocks': chiplet_1_blocks,
            'execution_time': execution_time,
            'chiplet_balance': chiplet_0_blocks / max(chiplet_1_blocks, 1)
        }
    
    def process_large_language_model(self, model_config):
        """Process Large Language Model optimized for B200's dual chiplet architecture"""
        num_layers = model_config['num_layers']
        seq_length = model_config['seq_length']
        hidden_dim = model_config['hidden_dim']
        batch_size = model_config.get('batch_size', 1)
        
        # Distribute layers across chiplets for optimal performance
        chiplet_0_layers = list(range(0, num_layers, 2))  # Even layers
        chiplet_1_layers = list(range(1, num_layers, 2))  # Odd layers
        
        chiplet_0_results = []
        chiplet_1_results = []
        
        # Process layers in parallel across chiplets
        chiplet_0_process = self.env.process(
            self._process_layers_on_chiplet(chiplet_0_layers, model_config, 0)
        )
        chiplet_1_process = self.env.process(
            self._process_layers_on_chiplet(chiplet_1_layers, model_config, 1)
        )
        
        # Wait for both chiplets to complete
        chiplet_0_result = yield chiplet_0_process
        chiplet_1_result = yield chiplet_1_process
        
        # Combine results
        total_cycles = max(chiplet_0_result['total_cycles'], chiplet_1_result['total_cycles'])
        total_tokens_per_second = (chiplet_0_result['tokens_per_second'] + 
                                 chiplet_1_result['tokens_per_second'])
        
        return {
            'model_config': model_config,
            'total_cycles': total_cycles,
            'tokens_per_second': total_tokens_per_second,
            'chiplet_0_performance': chiplet_0_result,
            'chiplet_1_performance': chiplet_1_result,
            'dual_chiplet_speedup': total_tokens_per_second / max(
                chiplet_0_result['tokens_per_second'], 
                chiplet_1_result['tokens_per_second']
            ),
            'interconnect_stats': self.chiplet_interconnect.get_stats()
        }
    
    def _process_layers_on_chiplet(self, layer_ids, model_config, chiplet_id):
        """Process specific layers on a chiplet"""
        target_sms = self.chiplet_0_sms if chiplet_id == 0 else self.chiplet_1_sms
        
        total_cycles = 0
        layer_results = []
        
        for layer_id in layer_ids:
            # Select SM with lowest utilization
            sm = min(target_sms, key=lambda s: len(s.active_warps))
            
            # Process inference workload
            result = yield self.env.process(
                sm.process_inference_workload(model_config)
            )
            
            layer_results.append(result)
            total_cycles += result['total_cycles']
        
        tokens_per_second = (len(layer_ids) * model_config['seq_length'] * 1000) / max(total_cycles, 1)
        
        return {
            'chiplet_id': chiplet_id,
            'layers_processed': layer_ids,
            'total_cycles': total_cycles,
            'tokens_per_second': tokens_per_second,
            'layer_results': layer_results
        }
    
    def get_gpu_stats(self):
        """Get comprehensive B200 GPU statistics"""
        # Separate statistics for each chiplet
        chiplet_0_stats = [sm.get_b200_specific_stats() for sm in self.chiplet_0_sms]
        chiplet_1_stats = [sm.get_b200_specific_stats() for sm in self.chiplet_1_sms]
        
        # Aggregate statistics
        total_instructions = sum(stats['total_instructions'] for stats in chiplet_0_stats + chiplet_1_stats)
        total_cycles = max(
            max([stats['total_cycles'] for stats in chiplet_0_stats] + [0]),
            max([stats['total_cycles'] for stats in chiplet_1_stats] + [0])
        )
        
        avg_occupancy = sum(stats['occupancy'] for stats in chiplet_0_stats + chiplet_1_stats) / len(chiplet_0_stats + chiplet_1_stats)
        
        return {
            'gpu_id': self.gpu_id,
            'gpu_type': self.gpu_type,
            'dual_chiplet_design': True,
            'total_sms': self.total_sms,
            'transistor_count': self.transistor_count,
            'hbm3e_memory_gb': self.hbm3e_memory_gb,
            'hbm3e_bandwidth_tbps': self.hbm3e_bandwidth_tbps,
            'total_instructions': total_instructions,
            'total_cycles': total_cycles,
            'gpu_ipc': total_instructions / max(total_cycles, 1),
            'average_occupancy': avg_occupancy,
            'chiplet_0_stats': chiplet_0_stats,
            'chiplet_1_stats': chiplet_1_stats,
            'chiplet_load_balance': self.chiplet_load_balance,
            'interconnect_stats': self.chiplet_interconnect.get_stats(),
            'memory_stats': self.memory_hierarchy.get_memory_stats(),
            'nvlink_bandwidth_gbps': self.nvlink_bandwidth_gbps
        }