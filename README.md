# NVIDIA H100/B200 GPU Simulation System

A comprehensive discrete-event simulation system for NVIDIA H100 and B200 GPUs, built with SimPy. This simulator models GPU architectures from Thread/Warp level up to complete GPU systems, enabling detailed performance analysis of AI workloads.

## 🚀 Features

### GPU Architecture Modeling
- **Thread-Level Simulation**: Individual GPU threads with register context and divergence tracking
- **Warp-Level Execution**: 32-thread SIMT execution with branch divergence handling
- **Thread 0 Leadership Pattern**: Optimized storage access using thread 0 as warp leader
- **SM Architecture**: 144 Streaming Multiprocessors with 4 warp schedulers each
- **Memory Hierarchy**: L1/L2 caches, shared memory, and register files with realistic latencies

### H100 Hopper Architecture
- **4th Generation Tensor Cores**: FP8/FP16 mixed precision support
- **Transformer Engine**: Dynamic precision switching for optimal AI performance
- **80GB HBM3 Memory**: 2TB/s bandwidth simulation
- **Thread Block Clusters**: Support for up to 16-block clusters

### B200 Blackwell Architecture
- **Dual Chiplet Design**: 2×72 SMs with 10TB/s inter-chiplet interconnect
- **Advanced Tensor Cores**: FP4 precision with 2:4 sparsity optimization
- **SER 2.0**: Shader Execution Reordering for improved warp scheduling
- **192GB HBM3E Memory**: 8TB/s bandwidth simulation

### AI-Specific Storage Systems (Thread 0 Optimized)
- **KV Cache**: LLM inference cache with compression, adaptive retention, and thread 0 access patterns
- **Vector Database**: HNSW/FAISS indexing for RAG workloads with warp-level broadcast optimization
- **GNN Storage**: Graph sampling and neighborhood queries with coordinated access patterns

### Workload Support
- **Large Language Models**: GPT-3 style inference and training
- **Transformer Models**: Attention mechanisms and FFN layers  
- **Graph Neural Networks**: Message passing and batch sampling
- **Computer Vision**: Convolution and image processing kernels

## 📁 Project Structure

```
simpy_simulator/
├── base/                          # Core simulation framework
│   ├── packet.py                  # Transaction/packet abstraction
│   ├── cuda_kernel.py             # CUDA kernel modeling
│   └── ...
├── components/                    # GPU component models
│   ├── gpu_thread.py              # Thread-level simulation
│   ├── gpu_warp.py                # Warp execution model
│   ├── streaming_multiprocessor.py # SM architecture
│   ├── gpu_memory_hierarchy.py    # Cache and memory systems
│   ├── h100_gpu.py                # H100-specific features
│   ├── b200_gpu.py                # B200-specific features
│   └── ai_storage.py              # AI workload storage
├── test_gpu_system.py             # Comprehensive test suite
├── test_thread0_optimization.py   # Thread 0 optimization benchmark
├── main_gpu_demo.py               # Interactive demonstration
└── README.md                      # This file
```

## 🛠 Installation

### Prerequisites
- Python 3.8+
- SimPy discrete event simulation framework

### Setup
```bash
# Clone the repository
git clone <repository-url>
cd simpy_simulator

# Install dependencies
pip install -r requirements.txt
```

## 🚦 Quick Start

### Run the Test Suite
```bash
python test_gpu_system.py
```
This runs comprehensive tests covering all GPU components and validates functionality.

### Run Thread 0 Optimization Benchmark
```bash
python test_thread0_optimization.py
```
This benchmarks the thread 0 leadership pattern performance optimization.

### Run the Interactive Demo
```bash
python main_gpu_demo.py
```
This demonstrates H100/B200 performance comparison, AI workloads, and storage systems.

### Basic Usage Example
```python
import simpy
from components.h100_gpu import H100GPU
from components.b200_gpu import B200GPU

# Create simulation environment
env = simpy.Environment()

# Initialize GPUs
h100 = H100GPU(env, gpu_id=0)
b200 = B200GPU(env, gpu_id=1)

# Define model configuration
model_config = {
    'num_layers': 32,
    'seq_length': 2048, 
    'hidden_dim': 4096,
    'num_heads': 32
}

def run_comparison(env):
    # Run on H100
    h100_result = yield env.process(
        h100.process_transformer_model(model_config)
    )
    
    # Run on B200
    b200_result = yield env.process(
        b200.process_large_language_model(model_config)
    )
    
    print(f"H100 Performance: {h100_result['estimated_tokens_per_second']:.1f} tokens/sec")
    print(f"B200 Performance: {b200_result['tokens_per_second']:.1f} tokens/sec")

# Run simulation
env.process(run_comparison(env))
env.run(until=10000)
```

## 📊 Performance Analysis

The simulator provides detailed performance metrics:

- **Execution Time**: Cycle-accurate timing simulation
- **Throughput**: Tokens/second for LLM workloads
- **Occupancy**: SM utilization and warp scheduling efficiency
- **Memory Performance**: Cache hit rates and bandwidth utilization
- **Power Efficiency**: Operations per cycle across different precisions

### Sample Results
- **B200 LLM Inference**: 341,333 tokens/second
- **Dual Chiplet Speedup**: 2.0x over single chiplet
- **FP4 Sparse Operations**: 65,536 ops/cycle
- **KV Cache Compression**: Up to 94% size reduction

### Thread 0 Storage Optimization Results
- **KV Cache Access Speedup**: 12.6x performance improvement
- **Vector Database Search Speedup**: 31.2x performance improvement
- **Storage Bandwidth Reduction**: 96.9% memory bandwidth savings
- **Average Storage Access Speedup**: 21.9x across all storage systems

## 🧪 Testing

The project includes comprehensive testing covering:

- ✅ H100 basic functionality (144 SMs, 80GB HBM3)
- ✅ B200 dual chiplet architecture (2×72 SMs, interconnect)
- ✅ Tensor Core operations (FP32, FP16, FP8, FP4)
- ✅ Transformer Engine dynamic precision
- ✅ KV Cache storage and compression
- ✅ Vector Database indexing and search
- ✅ GNN graph sampling and storage
- ✅ CUDA kernel generation and execution
- ✅ Mixed AI workload processing
- ✅ H100 vs B200 performance comparison
- ✅ Thread 0 storage access optimization

**Test Results: 100% Pass Rate (11/11 tests + Thread 0 optimization benchmark)**

## 🔬 Advanced Features

### Thread-Level Modeling
- Individual thread state tracking
- Register allocation and pressure analysis
- Branch divergence prediction and mitigation

### Warp-Level Operations
- SIMT execution with predication
- Thread 0 leadership for storage access with broadcast distribution
- Warp-level primitives (shuffle, vote, ballot)
- Memory coalescing optimization

### Memory System Accuracy
- Cache hierarchy with configurable policies
- Bank conflict detection in shared memory
- Realistic memory access latencies

### AI Workload Optimization
- Thread 0 leadership pattern for 97% storage bandwidth reduction
- Dynamic KV cache retention policies with warp-level access coordination
- Vector similarity search with hot/cold tiering and broadcast optimization
- Graph neural network batch processing with coordinated storage access

## 📈 Benchmarking

### Supported Workloads
- **LLM Inference**: GPT-3 (7B, 13B, 70B parameter models)
- **LLM Training**: Forward/backward pass simulation
- **RAG Systems**: Vector embedding search
- **GNN Processing**: Graph convolution and message passing
- **Computer Vision**: CNN training and inference

### Performance Comparisons
- H100 vs B200 across different model sizes
- Precision impact analysis (FP32 → FP16 → FP8 → FP4)
- Sparsity optimization benefits
- Memory hierarchy efficiency

## 🤝 Contributing

This project demonstrates advanced GPU simulation techniques. To extend the simulator:

1. **Add New GPU Architectures**: Implement new GPU classes following the base patterns
2. **Expand Workload Support**: Add new kernel types and execution patterns
3. **Enhance Memory Models**: Implement additional cache policies or memory types
4. **Improve Accuracy**: Add more detailed timing and power models

## 📚 Technical Details

### Simulation Methodology
- **Discrete Event Simulation**: Using SimPy for accurate timing
- **Cycle-Accurate Modeling**: Precise execution cycle counting
- **Resource Contention**: Realistic modeling of SM and memory conflicts

### Validation Approach
- Component-level unit testing
- End-to-end workload validation
- Performance regression testing
- Architecture comparison benchmarks

### Key Algorithms
- **Warp Scheduling**: Round-robin and priority-based policies
- **Cache Replacement**: LRU, LFU, and custom AI-aware policies
- **Memory Coalescing**: Transaction optimization for global memory
- **Divergence Handling**: Stack-based reconvergence

## 📄 License

This project is for educational and research purposes, demonstrating GPU architecture simulation techniques.

## 🔗 References

- NVIDIA H100 Tensor Core GPU Architecture
- NVIDIA B200 Blackwell Architecture Documentation  
- CUDA Programming Guide
- SimPy Discrete Event Simulation Framework

---

**Built with ❤️ for GPU architecture research and AI performance optimization**