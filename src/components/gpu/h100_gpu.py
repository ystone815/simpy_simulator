import simpy
import random
from enum import Enum
from src.components.gpu.streaming_multiprocessor import StreamingMultiprocessor, SMCluster
from src.components.gpu.gpu_memory_hierarchy import MemoryHierarchy
from src.components.gpu.gpu_warp import WarpInstruction
from src.base.packet import Packet

class TensorPrecision(Enum):
    FP32 = "fp32"
    FP16 = "fp16" 
    BF16 = "bf16"
    FP8 = "fp8"
    FP4 = "fp4"
    INT8 = "int8"
    INT4 = "int4"

class TensorCoreGen4:
    """
    4th Generation Tensor Core for H100 - supports mixed precision FP8/FP16.
    Implements matrix multiplication acceleration for AI workloads.
    """
    def __init__(self, env, core_id, sm_id):
        self.env = env
        self.core_id = core_id
        self.sm_id = sm_id
        
        # Supported precision combinations
        self.precision_configs = {
            (TensorPrecision.FP8, TensorPrecision.FP8): {"throughput": 1980, "latency": 1},
            (TensorPrecision.FP16, TensorPrecision.FP16): {"throughput": 989, "latency": 1},
            (TensorPrecision.BF16, TensorPrecision.BF16): {"throughput": 989, "latency": 1},
            (TensorPrecision.FP8, TensorPrecision.FP16): {"throughput": 1485, "latency": 1},  # Mixed precision
        }
        
        # Performance tracking
        self.operations_completed = 0
        self.total_cycles = 0
        self.utilization_time = 0
        
        # Supported matrix shapes (M, N, K)
        self.supported_shapes = [(16, 16, 16), (32, 8, 16), (8, 32, 16)]
        
    def execute_mma(self, shape, precision_a, precision_b, precision_c=TensorPrecision.FP32):
        """Execute Matrix Multiply-Accumulate operation"""
        if shape not in self.supported_shapes:
            raise ValueError(f"Unsupported tensor shape: {shape}")
        
        precision_key = (precision_a, precision_b)
        if precision_key not in self.precision_configs:
            raise ValueError(f"Unsupported precision combination: {precision_key}")
        
        config = self.precision_configs[precision_key]
        execution_cycles = config["latency"]
        
        # Simulate execution
        start_time = self.env.now
        yield self.env.timeout(execution_cycles)
        
        self.operations_completed += 1
        self.utilization_time += execution_cycles
        self.total_cycles += execution_cycles
        
        # Calculate theoretical throughput (TOPS)
        m, n, k = shape
        operations = 2 * m * n * k  # Multiply-accumulate
        
        return {
            'operations': operations,
            'cycles': execution_cycles,
            'throughput_ops_per_cycle': operations / execution_cycles,
            'precision_config': precision_key
        }
    
    def get_utilization(self):
        """Get tensor core utilization"""
        if self.total_cycles == 0:
            return 0
        return self.utilization_time / self.total_cycles
    
    def get_stats(self):
        """Get tensor core performance statistics"""
        return {
            'core_id': self.core_id,
            'sm_id': self.sm_id,
            'operations_completed': self.operations_completed,
            'utilization': self.get_utilization(),
            'total_cycles': self.total_cycles
        }


class TransformerEngine:
    """
    H100 Transformer Engine - dynamically switches between FP8 and FP16 precision
    to optimize transformer model training and inference.
    """
    def __init__(self, env, engine_id, sm_id):
        self.env = env
        self.engine_id = engine_id
        self.sm_id = sm_id
        
        # Dynamic precision switching
        self.current_precision = TensorPrecision.FP16
        self.precision_history = []
        self.switching_overhead_cycles = 2
        
        # Layer-specific precision tracking
        self.layer_precisions = {}  # layer_id -> precision
        
        # Performance counters
        self.attention_operations = 0
        self.ffn_operations = 0
        self.precision_switches = 0
        self.total_cycles = 0
        
    def process_attention_layer(self, layer_id, seq_length, hidden_dim, num_heads):
        """Process transformer attention mechanism with dynamic precision"""
        # Determine optimal precision for this layer
        optimal_precision = self._determine_precision(layer_id, "attention", seq_length, hidden_dim)
        
        # Switch precision if needed
        if optimal_precision != self.current_precision:
            yield self.env.timeout(self.switching_overhead_cycles)
            self.current_precision = optimal_precision
            self.precision_switches += 1
        
        # Record precision decision
        self.layer_precisions[layer_id] = optimal_precision
        self.precision_history.append((self.env.now, layer_id, optimal_precision))
        
        # Calculate computation cycles for attention
        head_dim = hidden_dim // num_heads
        
        # Q, K, V projections
        qkv_cycles = 3 * (seq_length * hidden_dim * hidden_dim) // self._get_throughput(optimal_precision)
        
        # Attention scores computation
        attn_cycles = num_heads * (seq_length * seq_length * head_dim) // self._get_throughput(optimal_precision)
        
        # Output projection
        out_cycles = (seq_length * hidden_dim * hidden_dim) // self._get_throughput(optimal_precision)
        
        total_cycles = qkv_cycles + attn_cycles + out_cycles
        
        yield self.env.timeout(total_cycles)
        
        self.attention_operations += 1
        self.total_cycles += total_cycles
        
        return {
            'layer_id': layer_id,
            'precision': optimal_precision.value,
            'cycles': total_cycles,
            'seq_length': seq_length,
            'hidden_dim': hidden_dim
        }
    
    def process_ffn_layer(self, layer_id, seq_length, hidden_dim, intermediate_dim):
        """Process transformer feed-forward network with dynamic precision"""
        optimal_precision = self._determine_precision(layer_id, "ffn", seq_length, hidden_dim)
        
        # Switch precision if needed
        if optimal_precision != self.current_precision:
            yield self.env.timeout(self.switching_overhead_cycles)
            self.current_precision = optimal_precision
            self.precision_switches += 1
        
        # Record precision decision
        self.layer_precisions[f"{layer_id}_ffn"] = optimal_precision
        
        # Calculate FFN computation cycles
        # First linear layer
        linear1_cycles = (seq_length * hidden_dim * intermediate_dim) // self._get_throughput(optimal_precision)
        
        # Activation function (GELU/ReLU)
        activation_cycles = seq_length * intermediate_dim // 1000  # Simplified
        
        # Second linear layer
        linear2_cycles = (seq_length * intermediate_dim * hidden_dim) // self._get_throughput(optimal_precision)
        
        total_cycles = linear1_cycles + activation_cycles + linear2_cycles
        
        yield self.env.timeout(total_cycles)
        
        self.ffn_operations += 1
        self.total_cycles += total_cycles
        
        return {
            'layer_id': layer_id,
            'precision': optimal_precision.value,
            'cycles': total_cycles,
            'intermediate_dim': intermediate_dim
        }
    
    def _determine_precision(self, layer_id, layer_type, seq_length, hidden_dim):
        """Determine optimal precision for layer based on characteristics"""
        # Simplified heuristic for precision selection
        # In practice, this would use learned policies or gradient sensitivity analysis
        
        if layer_type == "attention":
            # Use higher precision for attention computation due to softmax sensitivity
            if seq_length > 2048 or hidden_dim > 4096:
                return TensorPrecision.FP8  # Use FP8 for very large sequences
            else:
                return TensorPrecision.FP16
        
        elif layer_type == "ffn":
            # FFN layers can often use lower precision
            if hidden_dim > 8192:
                return TensorPrecision.FP8
            else:
                return TensorPrecision.FP16
        
        return TensorPrecision.FP16  # Default
    
    def _get_throughput(self, precision):
        """Get throughput for given precision (operations per cycle)"""
        throughput_map = {
            TensorPrecision.FP32: 312,
            TensorPrecision.FP16: 624,
            TensorPrecision.BF16: 624,
            TensorPrecision.FP8: 1248
        }
        return throughput_map.get(precision, 624)
    
    def get_precision_distribution(self):
        """Get distribution of precision usage"""
        if not self.precision_history:
            return {}
        
        precision_counts = {}
        for _, _, precision in self.precision_history:
            precision_counts[precision.value] = precision_counts.get(precision.value, 0) + 1
        
        total = len(self.precision_history)
        return {p: count/total for p, count in precision_counts.items()}
    
    def get_stats(self):
        """Get transformer engine statistics"""
        return {
            'engine_id': self.engine_id,
            'sm_id': self.sm_id,
            'attention_operations': self.attention_operations,
            'ffn_operations': self.ffn_operations,
            'precision_switches': self.precision_switches,
            'total_cycles': self.total_cycles,
            'precision_distribution': self.get_precision_distribution(),
            'current_precision': self.current_precision.value
        }


class H100StreamingMultiprocessor(StreamingMultiprocessor):
    """
    H100-specific Streaming Multiprocessor with 4th gen Tensor Cores and Transformer Engine.
    Extends base SM with H100-specific features.
    """
    def __init__(self, env, sm_id):
        super().__init__(env, sm_id, gpu_type="H100")
        
        # H100-specific components
        self.tensor_cores = []
        for i in range(4):  # 4 Tensor Cores per SM in H100
            tensor_core = TensorCoreGen4(env, core_id=i, sm_id=sm_id)
            self.tensor_cores.append(tensor_core)
        
        # Transformer Engine
        self.transformer_engine = TransformerEngine(env, engine_id=0, sm_id=sm_id)
        
        # Thread Block Clusters support (H100 feature)
        self.max_cluster_size = 16  # Non-portable max cluster size
        self.portable_cluster_size = 8
        self.active_clusters = []
        
        # Enhanced memory subsystem
        self.l1_cache_size_kb = 256  # Combined L1/Texture/Shared
        self.shared_memory_size_kb = 228
        self.register_file_size = 65536
        
    def execute_tensor_operation(self, shape, precision_a, precision_b):
        """Execute tensor operation using Tensor Core"""
        # Select least utilized tensor core
        tensor_core = min(self.tensor_cores, key=lambda tc: tc.utilization_time)
        
        # Execute MMA operation
        result = yield self.env.process(
            tensor_core.execute_mma(shape, precision_a, precision_b)
        )
        
        return result
    
    def process_transformer_layer(self, layer_config):
        """Process transformer layer using Transformer Engine"""
        layer_id = layer_config['layer_id']
        seq_length = layer_config['seq_length']
        hidden_dim = layer_config['hidden_dim']
        num_heads = layer_config.get('num_heads', 32)
        intermediate_dim = layer_config.get('intermediate_dim', hidden_dim * 4)
        
        # Process attention
        attention_result = yield self.env.process(
            self.transformer_engine.process_attention_layer(
                layer_id, seq_length, hidden_dim, num_heads
            )
        )
        
        # Process FFN
        ffn_result = yield self.env.process(
            self.transformer_engine.process_ffn_layer(
                layer_id, seq_length, hidden_dim, intermediate_dim
            )
        )
        
        return {
            'attention': attention_result,
            'ffn': ffn_result,
            'total_cycles': attention_result['cycles'] + ffn_result['cycles']
        }
    
    def launch_thread_block_cluster(self, cluster_size, block_configs):
        """Launch thread block cluster (H100 feature)"""
        if cluster_size > self.max_cluster_size:
            return False
        
        if len(self.active_clusters) * 8 + cluster_size > self.max_warps:
            return False  # Not enough resources
        
        cluster = {
            'cluster_id': len(self.active_clusters),
            'size': cluster_size,
            'blocks': block_configs,
            'start_time': self.env.now
        }
        
        self.active_clusters.append(cluster)
        
        # Launch blocks with cluster synchronization support
        for block_config in block_configs[:cluster_size]:
            # Each block becomes warps
            warps_per_block = (block_config['threads'] + 31) // 32
            for _ in range(warps_per_block):
                instructions = block_config.get('instructions', [])
                self.launch_warp(instructions)
        
        return True
    
    def get_h100_specific_stats(self):
        """Get H100-specific performance statistics"""
        base_stats = self.get_performance_stats()
        
        h100_stats = {
            'tensor_cores': [tc.get_stats() for tc in self.tensor_cores],
            'transformer_engine': self.transformer_engine.get_stats(),
            'active_clusters': len(self.active_clusters),
            'cluster_utilization': len(self.active_clusters) / 8,  # Max 8 clusters typically
        }
        
        base_stats.update(h100_stats)
        return base_stats


class H100GPU:
    """
    Complete NVIDIA H100 GPU model with 144 SMs, NVLink, and memory hierarchy.
    """
    def __init__(self, env, gpu_id=0):
        self.env = env
        self.gpu_id = gpu_id
        self.gpu_type = "H100"
        
        # H100 Architecture Configuration
        self.num_sms = 144  # Up to 144 SMs in H100
        self.total_fp32_cores = 18432  # 144 SMs * 128 cores
        self.total_tensor_cores = 576  # 144 SMs * 4 cores
        self.hbm3_memory_gb = 80
        self.hbm3_bandwidth_tbps = 2.0
        
        # Create SM cluster
        self.sm_cluster = SMCluster(env, cluster_id=0, num_sms=self.num_sms, gpu_type="H100")
        
        # Replace base SMs with H100-specific SMs
        self.sms = []
        for i in range(self.num_sms):
            h100_sm = H100StreamingMultiprocessor(env, sm_id=i)
            self.sms.append(h100_sm)
        self.sm_cluster.sms = self.sms
        
        # Memory hierarchy
        self.memory_hierarchy = MemoryHierarchy(env, gpu_type="H100")
        
        # Create memory subsystems for all SMs
        for sm in self.sms:
            l1_cache, shared_mem, register_file = self.memory_hierarchy.create_sm_memory_subsystem(sm.sm_id)
            sm.connect_memory_subsystem(l1_cache, shared_mem, register_file)
        
        # Create L2 cache
        self.memory_hierarchy.create_l2_cache(size_mb=50)
        
        # NVLink connectivity (simplified)
        self.nvlink_bandwidth_gbps = 900  # 900 GB/s per direction
        self.nvlink_ports = 18
        
        # Performance tracking
        self.total_operations = 0
        self.total_runtime = 0
        
    def launch_cuda_kernel(self, grid_size, block_size, kernel_instructions):
        """Launch CUDA kernel across H100 SMs"""
        start_time = self.env.now
        
        blocks_launched = self.sm_cluster.launch_kernel(grid_size, block_size, kernel_instructions)
        
        # Wait for completion (simplified)
        max_sm_cycles = 0
        for sm in self.sms:
            if sm.total_cycles > max_sm_cycles:
                max_sm_cycles = sm.total_cycles
        
        self.total_runtime += (self.env.now - start_time)
        
        return {
            'blocks_launched': blocks_launched,
            'execution_time': self.env.now - start_time,
            'sm_utilization': [sm.get_occupancy() for sm in self.sms]
        }
    
    def process_transformer_model(self, model_config):
        """Process entire transformer model on H100"""
        num_layers = model_config['num_layers']
        seq_length = model_config['seq_length']
        hidden_dim = model_config['hidden_dim']
        num_heads = model_config.get('num_heads', 32)
        
        layer_results = []
        total_cycles = 0
        
        # Distribute layers across SMs
        for layer_id in range(num_layers):
            sm_id = layer_id % len(self.sms)
            sm = self.sms[sm_id]
            
            layer_config = {
                'layer_id': layer_id,
                'seq_length': seq_length,
                'hidden_dim': hidden_dim,
                'num_heads': num_heads
            }
            
            result = yield self.env.process(sm.process_transformer_layer(layer_config))
            layer_results.append(result)
            total_cycles += result['total_cycles']
        
        return {
            'model_config': model_config,
            'layer_results': layer_results,
            'total_cycles': total_cycles,
            'estimated_tokens_per_second': seq_length * 1000 / max(total_cycles, 1)
        }
    
    def get_gpu_stats(self):
        """Get comprehensive H100 GPU statistics"""
        sm_stats = [sm.get_h100_specific_stats() for sm in self.sms]
        memory_stats = self.memory_hierarchy.get_memory_stats()
        
        # Aggregate statistics
        total_instructions = sum(stats['total_instructions'] for stats in sm_stats)
        total_cycles = max(stats['total_cycles'] for stats in sm_stats) if sm_stats else 0
        avg_occupancy = sum(stats['occupancy'] for stats in sm_stats) / len(sm_stats)
        
        return {
            'gpu_id': self.gpu_id,
            'gpu_type': self.gpu_type,
            'total_sms': self.num_sms,
            'total_instructions': total_instructions,
            'total_cycles': total_cycles,
            'gpu_ipc': total_instructions / max(total_cycles, 1),
            'average_occupancy': avg_occupancy,
            'sm_statistics': sm_stats,
            'memory_statistics': memory_stats,
            'nvlink_bandwidth_gbps': self.nvlink_bandwidth_gbps
        }