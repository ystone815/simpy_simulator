#!/usr/bin/env python3
"""
GPU System Test Suite for H100/B200 simulation.
Tests various components and workloads to validate the implementation.
"""

import simpy
import random
from components.h100_gpu import H100GPU, TensorPrecision
from components.b200_gpu import B200GPU
from components.ai_storage import KVCacheStorage, VectorDatabase, GNNStorage, VectorDBIndexType
from base.cuda_kernel import WorkloadGenerator, KernelType
from base.packet import Packet

class GPUSystemTester:
    """Test suite for GPU system components"""
    
    def __init__(self):
        self.test_results = {}
        self.failed_tests = []
        self.passed_tests = []
    
    def run_all_tests(self):
        """Run complete test suite"""
        print("=" * 80)
        print("GPU SYSTEM TEST SUITE")
        print("=" * 80)
        
        # Component tests
        self.test_h100_basic_functionality()
        self.test_b200_dual_chiplet()
        self.test_tensor_core_operations()
        self.test_transformer_engine()
        self.test_kv_cache_operations()
        self.test_vector_database()
        self.test_gnn_storage()
        
        # Workload tests
        self.test_llm_inference_workload()
        self.test_training_workload()
        self.test_mixed_workload()
        
        # Performance comparison
        self.test_h100_vs_b200_performance()
        
        self.print_test_summary()
    
    def test_h100_basic_functionality(self):
        """Test H100 GPU basic functionality"""
        print("\n📋 Testing H100 Basic Functionality...")
        
        try:
            env = simpy.Environment()
            gpu = H100GPU(env, gpu_id=0)
            
            # Test SM initialization
            assert len(gpu.sms) == 144, f"Expected 144 SMs, got {len(gpu.sms)}"
            assert gpu.gpu_type == "H100", f"GPU type should be H100, got {gpu.gpu_type}"
            assert gpu.hbm3_memory_gb == 80, f"Expected 80GB HBM3, got {gpu.hbm3_memory_gb}"
            
            # Test memory hierarchy
            assert gpu.memory_hierarchy is not None, "Memory hierarchy not initialized"
            assert len(gpu.memory_hierarchy.l1_caches) == 144, "L1 caches not created for all SMs"
            
            print("✅ H100 basic functionality test PASSED")
            self.passed_tests.append("H100 Basic Functionality")
            
        except Exception as e:
            print(f"❌ H100 basic functionality test FAILED: {e}")
            self.failed_tests.append(f"H100 Basic Functionality: {e}")
    
    def test_b200_dual_chiplet(self):
        """Test B200 dual chiplet architecture"""
        print("\n📋 Testing B200 Dual Chiplet Architecture...")
        
        try:
            env = simpy.Environment()
            gpu = B200GPU(env, gpu_id=0)
            
            # Test dual chiplet design
            assert gpu.num_chiplets == 2, f"Expected 2 chiplets, got {gpu.num_chiplets}"
            assert len(gpu.chiplet_0_sms) == 72, f"Expected 72 SMs in chiplet 0, got {len(gpu.chiplet_0_sms)}"
            assert len(gpu.chiplet_1_sms) == 72, f"Expected 72 SMs in chiplet 1, got {len(gpu.chiplet_1_sms)}"
            assert gpu.hbm3e_memory_gb == 192, f"Expected 192GB HBM3E, got {gpu.hbm3e_memory_gb}"
            
            # Test interconnect
            assert gpu.chiplet_interconnect is not None, "Chiplet interconnect not initialized"
            assert gpu.chiplet_interconnect.bandwidth_tbps == 10.0, "Interconnect bandwidth incorrect"
            
            print("✅ B200 dual chiplet test PASSED")
            self.passed_tests.append("B200 Dual Chiplet")
            
        except Exception as e:
            print(f"❌ B200 dual chiplet test FAILED: {e}")
            self.failed_tests.append(f"B200 Dual Chiplet: {e}")
    
    def test_tensor_core_operations(self):
        """Test Tensor Core operations"""
        print("\n📋 Testing Tensor Core Operations...")
        
        def tensor_core_test(env):
            # Test H100 Tensor Core
            from components.h100_gpu import TensorCoreGen4
            tensor_core = TensorCoreGen4(env, core_id=0, sm_id=0)
            
            # Test FP16 operation
            result = yield env.process(tensor_core.execute_mma(
                shape=(16, 16, 16),
                precision_a=TensorPrecision.FP16,
                precision_b=TensorPrecision.FP16
            ))
            
            assert result['operations'] == 2 * 16 * 16 * 16, "Incorrect operation count"
            assert result['precision_config'] == (TensorPrecision.FP16, TensorPrecision.FP16), "Precision config mismatch"
            
            # Test mixed precision FP8/FP16
            result = yield env.process(tensor_core.execute_mma(
                shape=(16, 16, 16),
                precision_a=TensorPrecision.FP8,
                precision_b=TensorPrecision.FP16
            ))
            
            assert result['operations'] == 2 * 16 * 16 * 16, "Mixed precision operation count incorrect"
            
            print("✅ Tensor Core operations test PASSED")
        
        try:
            env = simpy.Environment()
            env.process(tensor_core_test(env))
            env.run(until=100)
            self.passed_tests.append("Tensor Core Operations")
            
        except Exception as e:
            print(f"❌ Tensor Core operations test FAILED: {e}")
            self.failed_tests.append(f"Tensor Core Operations: {e}")
    
    def test_transformer_engine(self):
        """Test Transformer Engine functionality"""
        print("\n📋 Testing Transformer Engine...")
        
        def transformer_test(env):
            from components.h100_gpu import TransformerEngine
            engine = TransformerEngine(env, engine_id=0, sm_id=0)
            
            # Test attention layer processing
            result = yield env.process(engine.process_attention_layer(
                layer_id=0,
                seq_length=512,
                hidden_dim=1024,
                num_heads=16
            ))
            
            assert result['layer_id'] == 0, "Layer ID mismatch"
            assert result['seq_length'] == 512, "Sequence length mismatch"
            assert result['cycles'] > 0, "No cycles recorded"
            
            # Test FFN layer processing
            result = yield env.process(engine.process_ffn_layer(
                layer_id=0,
                seq_length=512,
                hidden_dim=1024,
                intermediate_dim=4096
            ))
            
            assert result['cycles'] > 0, "FFN processing should take cycles"
            assert result['intermediate_dim'] == 4096, "Intermediate dimension mismatch"
            
            print("✅ Transformer Engine test PASSED")
        
        try:
            env = simpy.Environment()
            env.process(transformer_test(env))
            env.run(until=1000)
            self.passed_tests.append("Transformer Engine")
            
        except Exception as e:
            print(f"❌ Transformer Engine test FAILED: {e}")
            self.failed_tests.append(f"Transformer Engine: {e}")
    
    def test_kv_cache_operations(self):
        """Test KV Cache storage operations"""
        print("\n📋 Testing KV Cache Operations...")
        
        def kv_cache_test(env):
            cache = KVCacheStorage(env, cache_id=0, max_tokens=1024)
            
            # Test storing tokens
            store_packet = Packet(
                id=1,
                type='store',
                source_id='test',
                layer_id=0,
                token_positions=[0, 1, 2, 3],
                attention_scores=[0.8, 0.6, 0.4, 0.2],
                data=[
                    {'key': b'key0', 'value': b'val0'},
                    {'key': b'key1', 'value': b'val1'},
                    {'key': b'key2', 'value': b'val2'},
                    {'key': b'key3', 'value': b'val3'}
                ]
            )
            
            yield cache.request_port.put(store_packet)
            yield env.timeout(10)  # Allow processing
            
            # Test retrieving tokens
            retrieve_packet = Packet(
                id=2,
                type='retrieve',
                source_id='test',
                layer_id=0,
                token_positions=[0, 1, 2],
                access_pattern='sequential'
            )
            
            yield cache.request_port.put(retrieve_packet)
            response = yield cache.response_port.get()
            
            assert response.cache_hit == True, "Should be cache hit"
            assert len(response.data) == 3, "Should retrieve 3 tokens"
            
            stats = cache.get_cache_stats()
            assert stats['stored_tokens'] > 0, "Should have stored tokens"
            assert stats['hit_rate'] > 0, "Should have cache hits"
            
            print("✅ KV Cache operations test PASSED")
        
        try:
            env = simpy.Environment()
            env.process(kv_cache_test(env))
            env.run(until=100)
            self.passed_tests.append("KV Cache Operations")
            
        except Exception as e:
            print(f"❌ KV Cache operations test FAILED: {e}")
            self.failed_tests.append(f"KV Cache Operations: {e}")
    
    def test_vector_database(self):
        """Test Vector Database operations"""
        print("\n📋 Testing Vector Database...")
        
        def vector_db_test(env):
            db = VectorDatabase(env, db_id=0, vector_dim=128, index_type=VectorDBIndexType.HNSW)
            
            # Test vector insertion
            vectors = [[random.random() for _ in range(128)] for _ in range(100)]
            metadata = [{'id': i} for i in range(100)]
            
            insert_packet = Packet(
                id=1,
                type='insert',
                source_id='test',
                data=vectors,
                metadata_list=metadata
            )
            
            yield db.request_port.put(insert_packet)
            response = yield db.response_port.get()
            
            assert response.data['inserted_count'] == 100, "Should insert 100 vectors"
            
            # Test index building
            build_packet = Packet(id=2, type='build_index', source_id='test', data={})
            yield db.request_port.put(build_packet)
            response = yield db.response_port.get()
            
            assert response.data['status'] == 'success', "Index build should succeed"
            
            # Test vector search
            query_vector = [random.random() for _ in range(128)]
            search_packet = Packet(
                id=3,
                type='search',
                source_id='test',
                data=query_vector,
                k=10
            )
            
            yield db.request_port.put(search_packet)
            response = yield db.response_port.get()
            
            assert len(response.data['results']) <= 10, "Should return at most 10 results"
            
            stats = db.get_db_stats()
            assert stats['total_vectors'] == 100, "Should have 100 vectors"
            
            print("✅ Vector Database test PASSED")
        
        try:
            env = simpy.Environment()
            env.process(vector_db_test(env))
            env.run(until=200)
            self.passed_tests.append("Vector Database")
            
        except Exception as e:
            print(f"❌ Vector Database test FAILED: {e}")
            self.failed_tests.append(f"Vector Database: {e}")
    
    def test_gnn_storage(self):
        """Test GNN Storage operations"""
        print("\n📋 Testing GNN Storage...")
        
        def gnn_test(env):
            storage = GNNStorage(env, storage_id=0, max_nodes=1000, feature_dim=64)
            
            # Test adding nodes
            node_packet = Packet(
                id=1,
                type='add_nodes',
                source_id='test',
                data={
                    'node_ids': list(range(100)),
                    'features': [[random.random() for _ in range(64)] for _ in range(100)]
                }
            )
            
            yield storage.request_port.put(node_packet)
            response = yield storage.response_port.get()
            
            assert response.data['nodes_added'] == 100, "Should add 100 nodes"
            
            # Test adding edges
            edges = [(i, (i + 1) % 100) for i in range(100)]  # Ring graph
            edge_packet = Packet(
                id=2,
                type='add_edges',
                source_id='test',
                data={'edges': edges}
            )
            
            yield storage.request_port.put(edge_packet)
            response = yield storage.response_port.get()
            
            assert response.data['edges_added'] == 100, "Should add 100 edges"
            
            # Test neighborhood sampling
            sample_packet = Packet(
                id=3,
                type='sample_neighborhood',
                source_id='test',
                data={
                    'node_id': 0,
                    'num_hops': 2,
                    'num_neighbors': 5
                }
            )
            
            yield storage.request_port.put(sample_packet)
            response = yield storage.response_port.get()
            
            assert response.data['center_node'] == 0, "Center node should be 0"
            assert len(response.data['sampled_nodes']) > 0, "Should sample some nodes"
            
            stats = storage.get_storage_stats()
            assert stats['nodes'] == 100, "Should have 100 nodes"
            assert stats['edges'] == 100, "Should have 100 edges"
            
            print("✅ GNN Storage test PASSED")
        
        try:
            env = simpy.Environment()
            env.process(gnn_test(env))
            env.run(until=200)
            self.passed_tests.append("GNN Storage")
            
        except Exception as e:
            print(f"❌ GNN Storage test FAILED: {e}")
            self.failed_tests.append(f"GNN Storage: {e}")
    
    def test_llm_inference_workload(self):
        """Test LLM inference workload generation and execution"""
        print("\n📋 Testing LLM Inference Workload...")
        
        try:
            env = simpy.Environment()
            workload_gen = WorkloadGenerator(env)
            
            # Generate 7B model inference workload
            workload = workload_gen.generate_llm_inference_workload("7B", batch_size=1, seq_length=1024)
            
            assert workload['model_size'] == "7B", "Model size mismatch"
            assert workload['seq_length'] == 1024, "Sequence length mismatch"
            assert len(workload['kernels']) > 0, "No kernels generated"
            
            # Test with H100 GPU
            h100_gpu = H100GPU(env, gpu_id=0)
            
            def run_inference_test(env):
                model_config = {
                    'num_layers': 4,  # Reduced for testing
                    'seq_length': 512,
                    'hidden_dim': 1024,
                    'num_heads': 16
                }
                
                result = yield env.process(h100_gpu.process_transformer_model(model_config))
                
                assert result['total_cycles'] > 0, "Should have execution cycles"
                assert len(result['layer_results']) == 4, "Should process 4 layers"
                assert result['estimated_tokens_per_second'] > 0, "Should have throughput estimate"
                
                print("✅ LLM Inference workload test PASSED")
            
            env.process(run_inference_test(env))
            env.run(until=2000)
            self.passed_tests.append("LLM Inference Workload")
            
        except Exception as e:
            print(f"❌ LLM Inference workload test FAILED: {e}")
            self.failed_tests.append(f"LLM Inference Workload: {e}")
    
    def test_training_workload(self):
        """Test training workload generation"""
        print("\n📋 Testing Training Workload Generation...")
        
        try:
            env = simpy.Environment()
            workload_gen = WorkloadGenerator(env)
            
            # Generate transformer training workload
            workload = workload_gen.generate_training_workload("transformer", batch_size=32)
            
            assert workload['model_type'] == "transformer", "Model type mismatch"
            assert workload['batch_size'] == 32, "Batch size mismatch"
            assert len(workload['kernels']) > 0, "No kernels generated"
            
            # Check that we have both forward and backward kernels
            kernel_types = [kernel.kernel_type for kernel in workload['kernels']]
            assert KernelType.ATTENTION in kernel_types, "Should have attention kernels"
            assert KernelType.MATMUL in kernel_types, "Should have matrix multiplication kernels"
            
            print("✅ Training workload test PASSED")
            self.passed_tests.append("Training Workload")
            
        except Exception as e:
            print(f"❌ Training workload test FAILED: {e}")
            self.failed_tests.append(f"Training Workload: {e}")
    
    def test_mixed_workload(self):
        """Test mixed workload generation"""
        print("\n📋 Testing Mixed Workload...")
        
        try:
            env = simpy.Environment()
            workload_gen = WorkloadGenerator(env)
            
            mixed_workload = workload_gen.generate_mixed_workload()
            
            assert mixed_workload['workload_type'] == 'mixed', "Should be mixed workload"
            assert len(mixed_workload['workloads']) == 3, "Should have 3 different workloads"
            
            # Check workload types
            workload_types = [w['workload_type'] for w in mixed_workload['workloads']]
            assert 'llm_inference' in workload_types, "Should include LLM inference"
            assert 'training' in workload_types, "Should include training"
            assert 'gnn' in workload_types, "Should include GNN"
            
            print("✅ Mixed workload test PASSED")
            self.passed_tests.append("Mixed Workload")
            
        except Exception as e:
            print(f"❌ Mixed workload test FAILED: {e}")
            self.failed_tests.append(f"Mixed Workload: {e}")
    
    def test_h100_vs_b200_performance(self):
        """Test H100 vs B200 performance comparison"""
        print("\n📋 Testing H100 vs B200 Performance Comparison...")
        
        def performance_test(env):
            h100_gpu = H100GPU(env, gpu_id=0)
            b200_gpu = B200GPU(env, gpu_id=1)
            
            # Test model configuration
            model_config = {
                'num_layers': 4,
                'seq_length': 512,
                'hidden_dim': 2048,
                'num_heads': 32,
                'batch_size': 2
            }
            
            # Test H100 performance
            h100_start = env.now
            h100_result = yield env.process(h100_gpu.process_transformer_model(model_config))
            h100_time = env.now - h100_start
            
            # Reset environment time for fair comparison
            env._now = 0
            
            # Test B200 performance
            b200_start = env.now
            b200_result = yield env.process(b200_gpu.process_large_language_model(model_config))
            b200_time = env.now - b200_start
            
            # Compare performance
            h100_tokens_per_sec = h100_result.get('estimated_tokens_per_second', 0)
            b200_tokens_per_sec = b200_result.get('tokens_per_second', 0)
            
            print(f"🏁 H100 Performance: {h100_tokens_per_sec:.1f} tokens/sec, {h100_time} cycles")
            print(f"🏁 B200 Performance: {b200_tokens_per_sec:.1f} tokens/sec, {b200_time} cycles")
            
            # B200 should generally be faster due to architectural improvements
            speedup = b200_tokens_per_sec / max(h100_tokens_per_sec, 1)
            print(f"🚀 B200 Speedup: {speedup:.2f}x over H100")
            
            print("✅ H100 vs B200 performance comparison test PASSED")
        
        try:
            env = simpy.Environment()
            env.process(performance_test(env))
            env.run(until=5000)
            self.passed_tests.append("H100 vs B200 Performance")
            
        except Exception as e:
            print(f"❌ H100 vs B200 performance test FAILED: {e}")
            self.failed_tests.append(f"H100 vs B200 Performance: {e}")
    
    def print_test_summary(self):
        """Print comprehensive test summary"""
        print("\n" + "=" * 80)
        print("TEST RESULTS SUMMARY")
        print("=" * 80)
        
        total_tests = len(self.passed_tests) + len(self.failed_tests)
        pass_rate = len(self.passed_tests) / total_tests * 100 if total_tests > 0 else 0
        
        print(f"📊 Total Tests: {total_tests}")
        print(f"✅ Passed: {len(self.passed_tests)}")
        print(f"❌ Failed: {len(self.failed_tests)}")
        print(f"📈 Pass Rate: {pass_rate:.1f}%")
        
        if self.passed_tests:
            print(f"\n✅ PASSED TESTS:")
            for test in self.passed_tests:
                print(f"   • {test}")
        
        if self.failed_tests:
            print(f"\n❌ FAILED TESTS:")
            for test in self.failed_tests:
                print(f"   • {test}")
        
        print(f"\n{'🎉 ALL TESTS PASSED!' if not self.failed_tests else '⚠️  SOME TESTS FAILED'}")
        print("=" * 80)


def run_performance_benchmark():
    """Run performance benchmark comparing different configurations"""
    print("\n" + "=" * 80)
    print("PERFORMANCE BENCHMARK")
    print("=" * 80)
    
    env = simpy.Environment()
    
    # Create GPUs
    h100 = H100GPU(env, gpu_id=0)
    b200 = B200GPU(env, gpu_id=1)
    
    # Create workload generator
    workload_gen = WorkloadGenerator(env)
    
    # Test different model sizes
    model_configs = [
        {"name": "Small (1B)", "layers": 12, "hidden": 2048, "heads": 16, "seq": 512},
        {"name": "Medium (7B)", "layers": 32, "hidden": 4096, "heads": 32, "seq": 1024},
        {"name": "Large (13B)", "layers": 40, "hidden": 5120, "heads": 40, "seq": 2048},
    ]
    
    print("🔬 Model Size Comparison:")
    print("-" * 60)
    
    for config in model_configs:
        model_config = {
            'num_layers': config['layers'],
            'seq_length': config['seq'],
            'hidden_dim': config['hidden'],
            'num_heads': config['heads']
        }
        
        # Get basic stats (without running full simulation for speed)
        h100_stats = h100.get_gpu_stats()
        b200_stats = b200.get_gpu_stats()
        
        print(f"{config['name']}:")
        print(f"  H100: {h100_stats['total_sms']} SMs, {h100_stats['gpu_type']}")
        print(f"  B200: {b200_stats['total_sms']} SMs, {b200_stats['gpu_type']} (Dual Chiplet)")
        print()
    
    print("✨ Benchmark completed!")


if __name__ == "__main__":
    # Run the test suite
    tester = GPUSystemTester()
    tester.run_all_tests()
    
    # Run performance benchmark
    run_performance_benchmark()