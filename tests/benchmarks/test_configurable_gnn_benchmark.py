#!/usr/bin/env python3
import simpy
import time
import json
# import matplotlib.pyplot as plt  # Removed for compatibility
# import numpy as np
from components.configurable_gnn_engine import create_gnn_engine, GNNEngineMode, AccessPattern
from components.cugraph_inspired_gnn import GraphStorageConfig, GraphFormat

def test_basic_pattern_switching():
    """Test basic functionality of pattern switching"""
    print("🔬 Basic Pattern Switching Test")
    print("=" * 60)
    
    # Create adaptive engine
    engine, env = create_gnn_engine("adaptive")
    
    # Load test graph
    engine.load_graph(
        num_nodes=500,
        num_edges=2500,
        distribution='power_law'
    )
    
    # Test each pattern
    patterns = [
        AccessPattern.THREAD_0_LEADER,
        AccessPattern.MULTI_THREAD,
        AccessPattern.EDGE_CENTRIC,
        AccessPattern.HYBRID_ADAPTIVE
    ]
    
    results = {}
    
    for pattern in patterns:
        print(f"\n📊 Testing {pattern.value} pattern...")
        
        # Switch to pattern
        engine.switch_access_pattern(pattern, f"test_{pattern.value}")
        
        # Execute GNN layer
        start_time = time.time()
        layer_results = engine.execute_gnn_layer({
            'num_warps': 8,
            'message_type': 'test_pattern'
        })
        execution_time = time.time() - start_time
        
        # Analyze results
        successful_warps = sum(1 for r in layer_results if r['success'])
        total_latency = sum(
            sum(tr.get('storage_latency', 0) + tr.get('broadcast_latency', 0) 
                for tr in r['thread_results']) 
            for r in layer_results if r['success']
        )
        
        results[pattern.value] = {
            'execution_time': execution_time,
            'successful_warps': successful_warps,
            'total_latency': total_latency,
            'avg_latency_per_warp': total_latency / max(1, successful_warps)
        }
        
        print(f"   Execution time: {execution_time:.4f}s")
        print(f"   Successful warps: {successful_warps}/8")
        print(f"   Avg latency/warp: {results[pattern.value]['avg_latency_per_warp']:.2f}")
    
    return results

def test_sq_lock_contention():
    """Test SQ doorbell lock contention with multi-thread access"""
    print("\n🔒 SQ Doorbell Lock Contention Analysis")
    print("=" * 60)
    
    contention_results = {}
    
    # Test different SQ counts
    sq_counts = [32, 64, 128, 256, 512, 1024]
    
    for num_sqs in sq_counts:
        print(f"\n📊 Testing with {num_sqs} SQs...")
        
        # Create engine with limited SQs
        engine, env = create_gnn_engine("compute", nvme_config={'num_sqs': num_sqs})
        
        # Load dense graph to create contention
        engine.load_graph(
            num_nodes=320,  # 10 warps * 32 threads
            num_edges=1600,  # Dense connectivity
            distribution='uniform'
        )
        
        # Force multi-thread pattern to create SQ contention
        engine.switch_access_pattern(AccessPattern.MULTI_THREAD, "contention_test")
        
        # Execute with high warp count
        start_time = time.time()
        layer_results = engine.execute_gnn_layer({
            'num_warps': 20,  # 640 threads competing for SQs
            'message_type': 'contention_test'
        })
        execution_time = time.time() - start_time
        
        # Analyze contention
        total_contention_cycles = 0
        successful_accesses = 0
        contention_events = 0
        
        for warp_result in layer_results:
            if warp_result['success']:
                for thread_result in warp_result['thread_results']:
                    if 'sq_contention' in thread_result:
                        contention_cycles = thread_result['sq_contention']
                        total_contention_cycles += contention_cycles
                        if contention_cycles > 0:
                            contention_events += 1
                    successful_accesses += 1
        
        contention_rate = contention_events / max(1, successful_accesses)
        avg_contention_delay = total_contention_cycles / max(1, contention_events)
        
        contention_results[num_sqs] = {
            'execution_time': execution_time,
            'contention_rate': contention_rate,
            'avg_contention_delay': avg_contention_delay,
            'total_contention_cycles': total_contention_cycles,
            'successful_accesses': successful_accesses
        }
        
        print(f"   Contention rate: {contention_rate:.1%}")
        print(f"   Avg contention delay: {avg_contention_delay:.2f} cycles")
        print(f"   Total successful accesses: {successful_accesses}")
    
    return contention_results

def test_adaptive_decision_making():
    """Test adaptive pattern selection based on graph characteristics"""
    print("\n🧠 Adaptive Decision Making Test")
    print("=" * 60)
    
    # Test different graph types
    graph_configs = [
        {
            'name': 'sparse_uniform',
            'num_nodes': 1000,
            'num_edges': 2000,
            'distribution': 'uniform'
        },
        {
            'name': 'dense_uniform', 
            'num_nodes': 500,
            'num_edges': 5000,
            'distribution': 'uniform'
        },
        {
            'name': 'power_law_hubs',
            'num_nodes': 1000,
            'num_edges': 4000,
            'distribution': 'power_law'
        },
        {
            'name': 'random_mixed',
            'num_nodes': 800,
            'num_edges': 3200,
            'distribution': 'random'
        }
    ]
    
    adaptive_results = {}
    
    for config in graph_configs:
        print(f"\n📊 Testing {config['name']} graph...")
        
        # Create adaptive engine
        engine, env = create_gnn_engine("adaptive")
        
        # Load graph
        engine.load_graph(
            num_nodes=config['num_nodes'],
            num_edges=config['num_edges'],
            distribution=config['distribution']
        )
        
        # Execute multiple layers to see adaptive decisions
        layer_results = []
        for layer_idx in range(5):
            results = engine.execute_gnn_layer({
                'num_warps': 6,
                'message_type': f'adaptive_layer_{layer_idx}'
            })
            layer_results.append(results)
        
        # Analyze adaptive decisions
        report = engine.get_performance_report()
        
        adaptive_results[config['name']] = {
            'workload_characteristics': report['workload_characteristics'],
            'adaptive_decisions': report['adaptive_decisions'],
            'pattern_performance': report['pattern_performance'],
            'layer_results': layer_results
        }
        
        # Print decision summary
        print(f"   Graph characteristics:")
        characteristics = report['workload_characteristics']
        print(f"     Avg degree: {characteristics['avg_degree']:.2f}")
        print(f"     Max degree: {characteristics['max_degree']}")
        print(f"     Hub ratio: {characteristics['hub_ratio']:.1%}")
        print(f"     Sparsity: {characteristics['sparsity']:.4f}")
        print(f"   Adaptive decisions made: {report['adaptive_decisions']}")
    
    return adaptive_results

def test_performance_scaling():
    """Test performance scaling with different workload sizes"""
    print("\n📈 Performance Scaling Test")
    print("=" * 60)
    
    # Test different workload sizes
    workload_sizes = [
        {'warps': 5, 'nodes': 160, 'edges': 800},
        {'warps': 10, 'nodes': 320, 'edges': 1600},
        {'warps': 20, 'nodes': 640, 'edges': 3200},
        {'warps': 40, 'nodes': 1280, 'edges': 6400},
    ]
    
    scaling_results = {}
    
    # Test each engine mode
    modes = ['storage', 'compute', 'memory', 'adaptive']
    
    for mode in modes:
        print(f"\n📊 Testing {mode} mode scaling...")
        mode_results = {}
        
        for workload in workload_sizes:
            print(f"   Workload: {workload['warps']} warps, {workload['nodes']} nodes")
            
            # Create engine
            engine, env = create_gnn_engine(mode)
            
            # Load graph
            engine.load_graph(
                num_nodes=workload['nodes'],
                num_edges=workload['edges'],
                distribution='power_law'
            )
            
            # Benchmark execution
            start_time = time.time()
            benchmark_results = engine.benchmark_all_patterns({
                'num_warps': workload['warps'],
                'message_type': 'scaling_test'
            })
            execution_time = time.time() - start_time
            
            # Get performance report
            report = engine.get_performance_report()
            
            mode_results[workload['warps']] = {
                'execution_time': execution_time,
                'benchmark_results': benchmark_results,
                'total_warps_processed': report['execution_summary']['total_warps_processed'],
                'workload_characteristics': report['workload_characteristics']
            }
        
        scaling_results[mode] = mode_results
    
    return scaling_results

def comprehensive_comparison_test():
    """Comprehensive comparison of all approaches"""
    print("\n🏆 Comprehensive Comparison Test")
    print("=" * 60)
    
    # Test scenarios
    test_scenarios = [
        {
            'name': 'small_sparse',
            'num_nodes': 200,
            'num_edges': 400,
            'num_warps': 4,
            'distribution': 'uniform'
        },
        {
            'name': 'medium_dense',
            'num_nodes': 500,
            'num_edges': 2500,
            'num_warps': 10,
            'distribution': 'power_law'
        },
        {
            'name': 'large_hub_heavy',
            'num_nodes': 1000,
            'num_edges': 3000,
            'num_warps': 15,
            'distribution': 'power_law'
        }
    ]
    
    comparison_results = {}
    
    for scenario in test_scenarios:
        print(f"\n📊 Scenario: {scenario['name']}")
        print(f"   Nodes: {scenario['num_nodes']}, Edges: {scenario['num_edges']}")
        
        scenario_results = {}
        
        # Test each engine mode
        for mode in ['storage', 'compute', 'memory', 'adaptive']:
            print(f"   Testing {mode} mode...")
            
            # Create engine
            engine, env = create_gnn_engine(mode)
            
            # Load graph
            engine.load_graph(
                num_nodes=scenario['num_nodes'],
                num_edges=scenario['num_edges'],
                distribution=scenario['distribution']
            )
            
            # Execute and benchmark
            start_time = time.time()
            
            # Execute 3 layers for better averaging
            all_results = []
            for layer in range(3):
                layer_results = engine.execute_gnn_layer({
                    'num_warps': scenario['num_warps'],
                    'message_type': f'comparison_layer_{layer}'
                })
                all_results.extend(layer_results)
            
            execution_time = time.time() - start_time
            
            # Get final performance report
            report = engine.get_performance_report()
            
            # Calculate metrics
            total_successful_warps = sum(1 for r in all_results if r['success'])
            total_latency = 0
            thread_0_accesses = 0
            multi_thread_accesses = 0
            
            for warp_result in all_results:
                if warp_result['success']:
                    for thread_result in warp_result['thread_results']:
                        total_latency += thread_result.get('storage_latency', 0)
                        total_latency += thread_result.get('broadcast_latency', 0)
                        
                        if (thread_result.get('access_pattern') == 'thread_0_leader' and 
                            thread_result.get('thread_id') == 0):
                            thread_0_accesses += 1
                        elif thread_result.get('access_pattern') in ['multi_thread', 'edge_centric']:
                            multi_thread_accesses += 1
            
            storage_efficiency = thread_0_accesses / max(1, thread_0_accesses + multi_thread_accesses)
            
            scenario_results[mode] = {
                'execution_time': execution_time,
                'total_latency': total_latency,
                'avg_latency_per_warp': total_latency / max(1, total_successful_warps),
                'storage_efficiency': storage_efficiency,
                'successful_warps': total_successful_warps,
                'workload_characteristics': report['workload_characteristics'],
                'pattern_performance': report.get('pattern_performance', {})
            }
        
        comparison_results[scenario['name']] = scenario_results
    
    return comparison_results

def generate_performance_summary(all_results):
    """Generate comprehensive performance summary"""
    print("\n📊 PERFORMANCE SUMMARY REPORT")
    print("=" * 80)
    
    # Basic pattern switching results
    if 'basic_patterns' in all_results:
        print("\n🔄 Basic Pattern Performance:")
        basic = all_results['basic_patterns']
        for pattern, metrics in basic.items():
            print(f"   {pattern:20}: {metrics['avg_latency_per_warp']:6.2f} avg latency")
    
    # SQ contention analysis
    if 'sq_contention' in all_results:
        print("\n🔒 SQ Contention Analysis:")
        contention = all_results['sq_contention']
        optimal_sqs = min(contention.items(), key=lambda x: x[1]['contention_rate'])
        print(f"   Optimal SQ count: {optimal_sqs[0]} (contention rate: {optimal_sqs[1]['contention_rate']:.1%})")
        
        # Show contention trend
        sq_counts = sorted(contention.keys())
        contention_rates = [contention[sq]['contention_rate'] for sq in sq_counts]
        print(f"   Contention rates: {' → '.join(f'{rate:.1%}' for rate in contention_rates)}")
    
    # Adaptive decision effectiveness
    if 'adaptive_decisions' in all_results:
        print("\n🧠 Adaptive Decision Effectiveness:")
        adaptive = all_results['adaptive_decisions']
        for graph_type, results in adaptive.items():
            characteristics = results['workload_characteristics']
            print(f"   {graph_type:15}: avg_degree={characteristics['avg_degree']:5.1f}, "
                  f"hub_ratio={characteristics['hub_ratio']:5.1%}, "
                  f"decisions={results['adaptive_decisions']}")
    
    # Performance scaling
    if 'performance_scaling' in all_results:
        print("\n📈 Performance Scaling Summary:")
        scaling = all_results['performance_scaling']
        for mode, mode_results in scaling.items():
            warp_counts = sorted(mode_results.keys())
            execution_times = [mode_results[warps]['execution_time'] for warps in warp_counts]
            print(f"   {mode:10} mode: {' → '.join(f'{time:.3f}s' for time in execution_times)}")
    
    # Overall comparison winner
    if 'comprehensive_comparison' in all_results:
        print("\n🏆 Overall Performance Winners:")
        comparison = all_results['comprehensive_comparison']
        
        for scenario, results in comparison.items():
            # Find best mode by lowest avg latency
            best_mode = min(results.items(), key=lambda x: x[1]['avg_latency_per_warp'])
            best_storage = max(results.items(), key=lambda x: x[1]['storage_efficiency'])
            
            print(f"   {scenario:15}:")
            print(f"     Best latency:    {best_mode[0]} ({best_mode[1]['avg_latency_per_warp']:.2f})")
            print(f"     Best storage:    {best_storage[0]} ({best_storage[1]['storage_efficiency']:.1%})")

def main():
    """Main benchmark execution"""
    print("🎯 Configurable GNN Engine Comprehensive Benchmark")
    print("=" * 80)
    
    all_results = {}
    
    # Run all benchmark tests
    try:
        all_results['basic_patterns'] = test_basic_pattern_switching()
        all_results['sq_contention'] = test_sq_lock_contention()
        all_results['adaptive_decisions'] = test_adaptive_decision_making()
        all_results['performance_scaling'] = test_performance_scaling()
        all_results['comprehensive_comparison'] = comprehensive_comparison_test()
        
        # Generate summary
        generate_performance_summary(all_results)
        
        # Save results
        with open('gnn_benchmark_results.json', 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        print(f"\n📄 Complete benchmark results saved to gnn_benchmark_results.json")
        
        return all_results
        
    except Exception as e:
        print(f"\n❌ Benchmark failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()