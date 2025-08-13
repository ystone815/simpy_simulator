import simpy
import random
from enum import Enum
from src.components.gpu.gpu_warp import WarpInstruction
from src.components.gpu.h100_gpu import TensorPrecision
from src.base.packet import Packet

class KernelType(Enum):
    MATMUL = "matrix_multiply"
    ATTENTION = "attention"
    CONVOLUTION = "convolution"
    ELEMENTWISE = "elementwise"
    REDUCTION = "reduction"
    GNN_MESSAGE_PASSING = "gnn_message_passing"

class CUDAKernel:
    """
    CUDA Kernel representation with grid/block configuration, instruction generation,
    and thread 0 storage access optimization patterns.
    """
    def __init__(self, kernel_type, grid_size, block_size, kernel_params=None):
        self.kernel_type = kernel_type
        self.grid_size = grid_size  # (grid_x, grid_y, grid_z)
        self.block_size = block_size  # (block_x, block_y, block_z)
        self.kernel_params = kernel_params or {}
        
        # Calculate total work
        self.total_blocks = grid_size[0] * grid_size[1] * grid_size[2]
        self.threads_per_block = block_size[0] * block_size[1] * block_size[2]
        self.warps_per_block = (self.threads_per_block + 31) // 32
        self.total_warps = self.total_blocks * self.warps_per_block
        
        # Storage access optimization settings
        self.uses_storage_access = self._kernel_uses_storage()
        self.thread_0_leadership = True  # Enable thread 0 leadership for storage
        self.storage_access_patterns = self._identify_storage_patterns()
        
        # Generate instruction sequence based on kernel type
        self.instructions = self._generate_instructions()
        
    def _generate_instructions(self):
        """Generate instruction sequence based on kernel type"""
        if self.kernel_type == KernelType.MATMUL:
            return self._generate_matmul_instructions()
        elif self.kernel_type == KernelType.ATTENTION:
            return self._generate_attention_instructions()
        elif self.kernel_type == KernelType.CONVOLUTION:
            return self._generate_conv_instructions()
        elif self.kernel_type == KernelType.ELEMENTWISE:
            return self._generate_elementwise_instructions()
        elif self.kernel_type == KernelType.REDUCTION:
            return self._generate_reduction_instructions()
        elif self.kernel_type == KernelType.GNN_MESSAGE_PASSING:
            return self._generate_gnn_instructions()
        else:
            return self._generate_generic_instructions()
    
    def _generate_matmul_instructions(self):
        """Generate matrix multiplication kernel instructions"""
        instructions = []
        
        # Load matrix tiles into shared memory
        instructions.extend([
            WarpInstruction("LOAD", ["GMEM_A", "SHMEM_A"], latency=100, throughput=1),
            WarpInstruction("LOAD", ["GMEM_B", "SHMEM_B"], latency=100, throughput=1),
            WarpInstruction("SYNC", latency=1),  # Block sync
        ])
        
        # Matrix multiply-accumulate loop
        tile_size = self.kernel_params.get('tile_size', 16)
        for k in range(tile_size):
            instructions.extend([
                WarpInstruction("LOAD", ["SHMEM_A", "REG_A"], latency=1),
                WarpInstruction("LOAD", ["SHMEM_B", "REG_B"], latency=1),
                WarpInstruction("HMMA", ["REG_A", "REG_B", "REG_C"], latency=1),  # Tensor Core operation
            ])
        
        # Store result
        instructions.extend([
            WarpInstruction("SYNC", latency=1),
            WarpInstruction("STORE", ["REG_C", "GMEM_C"], latency=100, throughput=1)
        ])
        
        return instructions
    
    def _generate_attention_instructions(self):
        """Generate transformer attention kernel instructions"""
        instructions = []
        seq_len = self.kernel_params.get('seq_length', 512)
        head_dim = self.kernel_params.get('head_dim', 64)
        
        # Q, K, V computation
        instructions.extend([
            WarpInstruction("LOAD", ["GMEM_INPUT", "REG_INPUT"], latency=100),
            WarpInstruction("HMMA", ["REG_INPUT", "REG_WQ", "REG_Q"], latency=1),  # Q projection
            WarpInstruction("HMMA", ["REG_INPUT", "REG_WK", "REG_K"], latency=1),  # K projection
            WarpInstruction("HMMA", ["REG_INPUT", "REG_WV", "REG_V"], latency=1),  # V projection
        ])
        
        # Attention scores computation with KV Cache access
        for i in range(seq_len // 32):  # Process in chunks
            instructions.extend([
                WarpInstruction("HMMA", ["REG_Q", "REG_K", "REG_SCORES"], latency=1),
                WarpInstruction("EXP", ["REG_SCORES"], latency=4),  # Softmax exp
                WarpInstruction("ADD", ["REG_SCORES", "REG_SUM"], latency=1),  # Sum for normalization
            ])
            
            # Add KV Cache access pattern (thread 0 leader)
            if self.uses_storage_access:
                instructions.extend(self._add_kv_cache_access_pattern())
        
        # Softmax normalization and output
        instructions.extend([
            WarpInstruction("DIV", ["REG_SCORES", "REG_SUM"], latency=8),  # Softmax normalization
            WarpInstruction("HMMA", ["REG_SCORES", "REG_V", "REG_OUT"], latency=1),  # Weighted sum
            WarpInstruction("STORE", ["REG_OUT", "GMEM_OUTPUT"], latency=100)
        ])
        
        return instructions
    
    def _generate_conv_instructions(self):
        """Generate convolution kernel instructions"""
        instructions = []
        filter_size = self.kernel_params.get('filter_size', 3)
        
        # Load input tile and filters
        instructions.extend([
            WarpInstruction("LOAD", ["GMEM_INPUT", "SHMEM_INPUT"], latency=100),
            WarpInstruction("LOAD", ["GMEM_FILTER", "REG_FILTER"], latency=50),
            WarpInstruction("SYNC", latency=1),
        ])
        
        # Convolution computation
        for fy in range(filter_size):
            for fx in range(filter_size):
                instructions.extend([
                    WarpInstruction("LOAD", ["SHMEM_INPUT", "REG_INPUT"], latency=1),
                    WarpInstruction("MUL", ["REG_INPUT", "REG_FILTER", "REG_TEMP"], latency=1),
                    WarpInstruction("ADD", ["REG_TEMP", "REG_ACC"], latency=1),
                ])
        
        # Activation and store
        instructions.extend([
            WarpInstruction("MAX", ["REG_ACC", "ZERO", "REG_RELU"], latency=1),  # ReLU
            WarpInstruction("STORE", ["REG_RELU", "GMEM_OUTPUT"], latency=100)
        ])
        
        return instructions
    
    def _generate_elementwise_instructions(self):
        """Generate elementwise operation instructions"""
        instructions = []
        vector_size = self.kernel_params.get('vector_size', 1024)
        
        # Simple elementwise operations
        elements_per_thread = vector_size // self.threads_per_block
        for i in range(elements_per_thread):
            instructions.extend([
                WarpInstruction("LOAD", ["GMEM_A", "REG_A"], latency=100),
                WarpInstruction("LOAD", ["GMEM_B", "REG_B"], latency=100),
                WarpInstruction("ADD", ["REG_A", "REG_B", "REG_C"], latency=1),
                WarpInstruction("STORE", ["REG_C", "GMEM_C"], latency=100)
            ])
        
        return instructions
    
    def _generate_reduction_instructions(self):
        """Generate reduction kernel instructions"""
        instructions = []
        
        # Load data
        instructions.append(WarpInstruction("LOAD", ["GMEM_INPUT", "REG_DATA"], latency=100))
        
        # Tree reduction within warp
        for stride in [16, 8, 4, 2, 1]:
            instructions.extend([
                WarpInstruction("SHFL", ["REG_DATA", f"stride_{stride}"], latency=1),
                WarpInstruction("ADD", ["REG_DATA", "REG_TEMP"], latency=1),
            ])
        
        # Block-level reduction
        instructions.extend([
            WarpInstruction("SYNC", latency=1),
            WarpInstruction("STORE", ["REG_DATA", "SHMEM_PARTIAL"], latency=1),
            WarpInstruction("SYNC", latency=1),
            WarpInstruction("LOAD", ["SHMEM_PARTIAL", "REG_FINAL"], latency=1),
            WarpInstruction("ADD", ["REG_FINAL", "REG_RESULT"], latency=1),
            WarpInstruction("STORE", ["REG_RESULT", "GMEM_OUTPUT"], latency=100)
        ])
        
        return instructions
    
    def _generate_gnn_instructions(self):
        """Generate GNN message passing instructions"""
        instructions = []
        feature_dim = self.kernel_params.get('feature_dim', 128)
        max_neighbors = self.kernel_params.get('max_neighbors', 16)
        
        # Load node features
        instructions.append(WarpInstruction("LOAD", ["GMEM_FEATURES", "REG_FEATURES"], latency=100))
        
        # Message passing loop with GNN storage access
        for neighbor in range(max_neighbors):
            instructions.extend([
                WarpInstruction("BRANCH", ["neighbor_valid"], latency=1, divergent=True),
                WarpInstruction("HMMA", ["REG_FEATURES", "REG_WEIGHT", "REG_MESSAGE"], latency=1),
                WarpInstruction("ADD", ["REG_MESSAGE", "REG_AGGREGATION"], latency=1),
            ])
            
            # GNN storage access using thread 0 leadership
            if self.uses_storage_access:
                instructions.extend(self._add_gnn_storage_access_pattern())
        
        # Activation and update
        instructions.extend([
            WarpInstruction("DIV", ["REG_AGGREGATION", "neighbor_count"], latency=8),  # Mean aggregation
            WarpInstruction("HMMA", ["REG_AGGREGATION", "REG_UPDATE_WEIGHT", "REG_UPDATED"], latency=1),
            WarpInstruction("ADD", ["REG_FEATURES", "REG_UPDATED"], latency=1),  # Residual connection
            WarpInstruction("STORE", ["REG_UPDATED", "GMEM_OUTPUT"], latency=100)
        ])
        
        return instructions
    
    def _generate_generic_instructions(self):
        """Generate generic instruction sequence"""
        instructions = [
            WarpInstruction("LOAD", latency=100),
            WarpInstruction("ADD", latency=1),
            WarpInstruction("MUL", latency=1),
            WarpInstruction("STORE", latency=100)
        ]
        return instructions * 10  # Repeat pattern
    
    def _kernel_uses_storage(self):
        """Determine if kernel type uses storage systems"""
        storage_kernels = [KernelType.ATTENTION, KernelType.GNN_MESSAGE_PASSING]
        return self.kernel_type in storage_kernels
    
    def _identify_storage_patterns(self):
        """Identify which storage systems this kernel uses"""
        patterns = []
        if self.kernel_type == KernelType.ATTENTION:
            patterns.extend(['kv_cache', 'vector_db'])  # For RAG-style attention
        elif self.kernel_type == KernelType.GNN_MESSAGE_PASSING:
            patterns.append('gnn_storage')
        return patterns
    
    def _add_kv_cache_access_pattern(self):
        """Add KV Cache access instructions with thread 0 leadership"""
        return [
            WarpInstruction("THREAD_LEADER_CHECK", ["thread_id", "0"], latency=1),
            WarpInstruction("BRANCH", ["is_leader"], latency=1, divergent=True),
            WarpInstruction("STORAGE_ACCESS", ["KV_CACHE", "token_positions"], latency=5),
            WarpInstruction("WARP_BROADCAST", ["kv_data", "all_threads"], latency=2),
            WarpInstruction("SYNC", latency=1),
        ]
    
    def _add_vector_db_access_pattern(self):
        """Add Vector Database access instructions with thread 0 leadership"""
        return [
            WarpInstruction("THREAD_LEADER_CHECK", ["thread_id", "0"], latency=1),
            WarpInstruction("BRANCH", ["is_leader"], latency=1, divergent=True),
            WarpInstruction("STORAGE_ACCESS", ["VECTOR_DB", "query_vector"], latency=10),
            WarpInstruction("WARP_BROADCAST", ["search_results", "all_threads"], latency=3),
            WarpInstruction("SYNC", latency=1),
        ]
    
    def _add_gnn_storage_access_pattern(self):
        """Add GNN Storage access instructions with thread 0 leadership"""
        return [
            WarpInstruction("THREAD_LEADER_CHECK", ["thread_id", "0"], latency=1),
            WarpInstruction("BRANCH", ["is_leader"], latency=1, divergent=True),
            WarpInstruction("STORAGE_ACCESS", ["GNN_STORAGE", "neighbor_query"], latency=15),
            WarpInstruction("LOAD", ["GMEM_NEIGHBORS", "REG_NEIGHBOR"], latency=100),
            WarpInstruction("WARP_BROADCAST", ["neighbor_data", "all_threads"], latency=4),
            WarpInstruction("SYNC", latency=1),
        ]
    
    def get_storage_optimization_stats(self):
        """Get statistics about storage access optimizations"""
        if not self.uses_storage_access:
            return {'uses_storage': False}
        
        # Estimate savings from thread 0 leadership
        total_threads = self.total_warps * 32
        threads_per_warp = 32
        storage_instructions = len([i for i in self.instructions if 'STORAGE_ACCESS' in i.opcode])
        
        # Thread 0 leadership saves (31/32) of storage accesses per warp
        bandwidth_savings = (threads_per_warp - 1) / threads_per_warp * storage_instructions
        latency_reduction = storage_instructions * 0.3  # 30% latency reduction
        
        return {
            'uses_storage': True,
            'storage_patterns': self.storage_access_patterns,
            'thread_0_leadership': self.thread_0_leadership,
            'total_warps': self.total_warps,
            'storage_instructions': storage_instructions,
            'estimated_bandwidth_savings': bandwidth_savings,
            'estimated_latency_reduction': latency_reduction
        }
    
    def get_kernel_info(self):
        """Get kernel configuration information including storage optimizations"""
        base_info = {
            'kernel_type': self.kernel_type.value,
            'grid_size': self.grid_size,
            'block_size': self.block_size,
            'total_blocks': self.total_blocks,
            'threads_per_block': self.threads_per_block,
            'warps_per_block': self.warps_per_block,
            'total_warps': self.total_warps,
            'instruction_count': len(self.instructions),
            'kernel_params': self.kernel_params
        }
        
        # Add storage optimization info
        storage_info = self.get_storage_optimization_stats()
        base_info.update(storage_info)
        
        return base_info


class WorkloadGenerator:
    """
    Generates various AI workloads for testing GPU performance.
    """
    def __init__(self, env):
        self.env = env
        self.workloads_generated = 0
    
    def generate_llm_inference_workload(self, model_size="7B", batch_size=1, seq_length=2048):
        """Generate Large Language Model inference workload"""
        if model_size == "7B":
            hidden_dim = 4096
            num_heads = 32
            num_layers = 32
        elif model_size == "13B":
            hidden_dim = 5120
            num_heads = 40
            num_layers = 40
        elif model_size == "70B":
            hidden_dim = 8192
            num_heads = 64
            num_layers = 80
        else:
            hidden_dim = 4096
            num_heads = 32
            num_layers = 32
        
        kernels = []
        
        for layer in range(num_layers):
            # Attention kernel
            attention_kernel = CUDAKernel(
                kernel_type=KernelType.ATTENTION,
                grid_size=(seq_length // 128, batch_size, num_heads),
                block_size=(128, 1, 1),
                kernel_params={
                    'seq_length': seq_length,
                    'hidden_dim': hidden_dim,
                    'head_dim': hidden_dim // num_heads,
                    'layer_id': layer
                }
            )
            kernels.append(attention_kernel)
            
            # FFN kernels (2 matrix multiplications)
            for ffn_layer in range(2):
                intermediate_dim = hidden_dim * 4 if ffn_layer == 0 else hidden_dim
                input_dim = hidden_dim if ffn_layer == 0 else hidden_dim * 4
                
                ffn_kernel = CUDAKernel(
                    kernel_type=KernelType.MATMUL,
                    grid_size=(seq_length // 16, batch_size, 1),
                    block_size=(256, 1, 1),
                    kernel_params={
                        'input_dim': input_dim,
                        'output_dim': intermediate_dim,
                        'tile_size': 16
                    }
                )
                kernels.append(ffn_kernel)
        
        self.workloads_generated += 1
        
        return {
            'workload_type': 'llm_inference',
            'model_size': model_size,
            'batch_size': batch_size,
            'seq_length': seq_length,
            'num_layers': num_layers,
            'kernels': kernels,
            'total_kernels': len(kernels)
        }
    
    def generate_training_workload(self, model_type="transformer", batch_size=32):
        """Generate training workload with forward and backward passes"""
        kernels = []
        
        if model_type == "transformer":
            seq_length = 512
            hidden_dim = 768
            num_layers = 12
            
            for layer in range(num_layers):
                # Forward pass
                forward_kernels = [
                    CUDAKernel(KernelType.ATTENTION, (seq_length//64, batch_size, 12), (64, 1, 1)),
                    CUDAKernel(KernelType.MATMUL, (seq_length//16, batch_size, 1), (256, 1, 1)),
                    CUDAKernel(KernelType.MATMUL, (seq_length//16, batch_size, 1), (256, 1, 1)),
                ]
                kernels.extend(forward_kernels)
                
                # Backward pass (gradient computation)
                backward_kernels = [
                    CUDAKernel(KernelType.MATMUL, (seq_length//16, batch_size, 1), (256, 1, 1)),
                    CUDAKernel(KernelType.MATMUL, (seq_length//16, batch_size, 1), (256, 1, 1)),
                    CUDAKernel(KernelType.ATTENTION, (seq_length//64, batch_size, 12), (64, 1, 1)),
                ]
                kernels.extend(backward_kernels)
        
        elif model_type == "cnn":
            # CNN training workload
            feature_maps = [64, 128, 256, 512]
            for layer, channels in enumerate(feature_maps):
                # Forward convolution
                conv_kernel = CUDAKernel(
                    KernelType.CONVOLUTION,
                    grid_size=(224//16, 224//16, channels),
                    block_size=(16, 16, 1),
                    kernel_params={'filter_size': 3, 'channels': channels}
                )
                kernels.append(conv_kernel)
                
                # Backward convolution
                kernels.append(conv_kernel)  # Simplified - same as forward
        
        self.workloads_generated += 1
        
        return {
            'workload_type': 'training',
            'model_type': model_type,
            'batch_size': batch_size,
            'kernels': kernels,
            'total_kernels': len(kernels)
        }
    
    def generate_gnn_workload(self, graph_size="medium", batch_size=1024):
        """Generate Graph Neural Network workload"""
        if graph_size == "small":
            num_nodes = 10000
            avg_degree = 10
            feature_dim = 64
        elif graph_size == "medium":
            num_nodes = 100000
            avg_degree = 20
            feature_dim = 128
        elif graph_size == "large":
            num_nodes = 1000000
            avg_degree = 50
            feature_dim = 256
        
        kernels = []
        
        # Graph sampling kernel
        sampling_kernel = CUDAKernel(
            kernel_type=KernelType.GNN_MESSAGE_PASSING,
            grid_size=(num_nodes // 256, 1, 1),
            block_size=(256, 1, 1),
            kernel_params={
                'num_nodes': num_nodes,
                'feature_dim': feature_dim,
                'max_neighbors': avg_degree * 2
            }
        )
        kernels.append(sampling_kernel)
        
        # Message passing layers
        for layer in range(3):  # 3-layer GNN
            message_kernel = CUDAKernel(
                kernel_type=KernelType.GNN_MESSAGE_PASSING,
                grid_size=(num_nodes // 128, 1, 1),
                block_size=(128, 1, 1),
                kernel_params={
                    'feature_dim': feature_dim,
                    'max_neighbors': avg_degree,
                    'layer_id': layer
                }
            )
            kernels.append(message_kernel)
        
        self.workloads_generated += 1
        
        return {
            'workload_type': 'gnn',
            'graph_size': graph_size,
            'num_nodes': num_nodes,
            'feature_dim': feature_dim,
            'batch_size': batch_size,
            'kernels': kernels,
            'total_kernels': len(kernels)
        }
    
    def generate_mixed_workload(self):
        """Generate mixed AI workload combining different types"""
        workloads = []
        
        # Add some inference
        workloads.append(self.generate_llm_inference_workload("7B", batch_size=2))
        
        # Add some training
        workloads.append(self.generate_training_workload("transformer", batch_size=16))
        
        # Add some GNN processing
        workloads.append(self.generate_gnn_workload("medium"))
        
        return {
            'workload_type': 'mixed',
            'workloads': workloads,
            'total_workloads': len(workloads)
        }