#!/usr/bin/env python3
import simpy
import time
import json
from components.gpu_warp import Warp
from components.nvme_doorbell_system import NVMeDoorbellController, DoorbellAccessPattern
from components.nvme_fallback_patterns import NVMeFallbackController, FallbackAccessPattern

def test_nvme_doorbell_optimization():
    """Test NVMe doorbell access optimization patterns"""
    print("🔬 NVMe Doorbell Access Optimization Test")
    print("=" * 80)
    
    env = simpy.Environment()
    results = {}
    
    # Test scenario 1: Thread 0 available (optimal case)
    print("📊 Testing Thread 0 Leadership Pattern...")
    
    # Create warp with NVMe optimization
    warp = Warp(env, warp_id=0, sm_id=0, num_threads=32)
    warp.enable_nvme_doorbell_optimization(num_sqs=1024)
    
    # Simulate storage requests from warp
    storage_requests = []
    for i in range(32):
        request = {
            'lba': i * 8,       # Logical block address
            'blocks': 8,        # Number of blocks
            'write': True,      # Write operation
            'data': b'x' * 4096  # 4KB data
        }
        storage_requests.append(request)
    
    # Execute with Thread 0 optimization
    start_time = time.time()
    nvme_results = warp.execute_nvme_storage_access(storage_requests, "thread_0_optimized")
    optimized_time = time.time() - start_time
    
    # Analyze results
    thread_0_accesses = sum(1 for r in nvme_results if r.get('thread_id') == 0 and r.get('success'))
    total_doorbell_writes = sum(1 for r in nvme_results if r.get('doorbell_latency', 0) > 0)
    broadcast_operations = sum(1 for r in nvme_results if r.get('broadcast_latency', 0) > 0)
    
    results['thread_0_optimal'] = {
        'execution_time': optimized_time,
        'thread_0_accesses': thread_0_accesses,
        'total_doorbell_writes': total_doorbell_writes,
        'broadcast_operations': broadcast_operations,
        'bandwidth_savings': (32 - total_doorbell_writes) / 32 * 100
    }
    
    print(f"   Thread 0 Accesses: {thread_0_accesses}")
    print(f"   Total Doorbell Writes: {total_doorbell_writes}")
    print(f"   Broadcast Operations: {broadcast_operations}")
    print(f"   Bandwidth Savings: {results['thread_0_optimal']['bandwidth_savings']:.1f}%")
    print()
    
    # Test scenario 2: Thread 0 unavailable (fallback patterns)
    print("📊 Testing Fallback Patterns when Thread 0 Unavailable...")
    
    # Create new environment and warp
    env = simpy.Environment()
    warp_fallback = Warp(env, warp_id=1, sm_id=0, num_threads=32)
    warp_fallback.enable_nvme_doorbell_optimization(
        num_sqs=1024, 
        fallback_pattern=FallbackAccessPattern.NEXT_ACTIVE_LEADER
    )
    
    # Simulate Thread 0 being inactive/diverged
    # Manually set thread states to simulate failure
    thread_states = {0: 'STALLED'}  # Thread 0 is stalled
    for i in range(1, 32):
        thread_states[i] = 'ACTIVE'
    
    # Execute with fallback pattern
    start_time = time.time()
    fallback_results = []
    
    for i, request in enumerate(storage_requests):
        # Simulate calling fallback controller directly
        if hasattr(warp_fallback, 'nvme_fallback_controller'):
            from base.packet import Packet
            nvme_command = Packet(
                id=f"fallback_test_{i}",
                type="nvme_write",
                address=request['lba'],
                size=request['blocks']
            )
            
            result = warp_fallback.nvme_fallback_controller.submit_storage_command_with_fallback(
                nvme_command, i, 1, thread_states
            )
            result['thread_id'] = i
            fallback_results.append(result)
    
    fallback_time = time.time() - start_time
    
    # Analyze fallback results
    fallback_accesses = len([r for r in fallback_results if r.get('success')])
    leadership_transfers = warp_fallback.nvme_fallback_controller.leadership_transfers
    multi_leader_ops = warp_fallback.nvme_fallback_controller.multi_leader_operations
    
    results['fallback_patterns'] = {
        'execution_time': fallback_time,
        'successful_accesses': fallback_accesses,
        'leadership_transfers': leadership_transfers,
        'multi_leader_operations': multi_leader_ops,
        'fallback_activations': warp_fallback.nvme_fallback_controller.fallback_activations
    }
    
    print(f"   Successful Accesses: {fallback_accesses}/32")
    print(f"   Leadership Transfers: {leadership_transfers}")
    print(f"   Fallback Activations: {results['fallback_patterns']['fallback_activations']}")
    print()
    
    # Test scenario 3: SQ contention analysis
    print("📊 Testing SQ Contention with Limited Queues...")
    
    env = simpy.Environment()
    
    # Test with limited SQs (128 instead of 1024)
    limited_controller = NVMeDoorbellController(
        env=env,
        num_submission_queues=128,  # Much smaller than thread count
        sq_depth=32,
        access_pattern=DoorbellAccessPattern.THREAD_0_LEADER
    )
    
    # Simulate multiple warps competing for limited SQs
    contention_results = []
    total_stalls = 0
    
    for warp_id in range(10):  # 10 warps = 320 threads competing
        for thread_id in range(32):
            from base.packet import Packet
            command = Packet(
                id=f"contention_warp_{warp_id}_thread_{thread_id}",
                type="nvme_write",
                address=thread_id * 4096,
                size=8
            )
            
            result = limited_controller.submit_command_with_thread0_optimization(
                command, thread_id, warp_id, is_thread_leader=(thread_id == 0)
            )
            
            if not result['success']:
                total_stalls += 1
            
            contention_results.append(result)
    
    contention_stats = limited_controller.get_system_stats()
    
    results['sq_contention'] = {
        'num_sqs': 128,
        'total_commands': len(contention_results),
        'successful_commands': len([r for r in contention_results if r['success']]),
        'total_stalls': contention_stats['contention_stalls'],
        'utilization': contention_stats['total_utilization']
    }
    
    print(f"   Total Commands: {results['sq_contention']['total_commands']}")
    print(f"   Successful Commands: {results['sq_contention']['successful_commands']}")
    print(f"   Contention Stalls: {results['sq_contention']['total_stalls']}")
    print(f"   SQ Utilization: {results['sq_contention']['utilization']:.1%}")
    print()
    
    return results

def test_sq_mapping_strategies():
    """Test different SQ mapping strategies for thread-to-queue assignment"""
    print("🎯 SQ Mapping Strategy Comparison")
    print("=" * 80)
    
    env = simpy.Environment()
    strategies = [
        DoorbellAccessPattern.THREAD_0_LEADER,
        DoorbellAccessPattern.ROUND_ROBIN,
        DoorbellAccessPattern.HASH_BASED,
        DoorbellAccessPattern.DYNAMIC_LOAD_BALANCE
    ]
    
    results = {}
    
    for strategy in strategies:
        print(f"📊 Testing {strategy.value.upper()} Strategy...")
        
        controller = NVMeDoorbellController(
            env=env,
            num_submission_queues=256,
            sq_depth=64,
            access_pattern=strategy
        )
        
        # Simulate workload
        start_time = time.time()
        commands_submitted = 0
        
        for warp_id in range(20):  # 20 warps
            for thread_id in range(32):  # 32 threads per warp
                from base.packet import Packet
                command = Packet(
                    id=f"strategy_test_{strategy.value}_w{warp_id}_t{thread_id}",
                    type="nvme_read",
                    address=thread_id * 4096,
                    size=8
                )
                
                is_leader = (thread_id == 0) if strategy == DoorbellAccessPattern.THREAD_0_LEADER else False
                result = controller.submit_command_with_thread0_optimization(
                    command, thread_id, warp_id, is_leader
                )
                
                if result['success']:
                    commands_submitted += 1
        
        execution_time = time.time() - start_time
        stats = controller.get_system_stats()
        
        results[strategy.value] = {
            'execution_time': execution_time,
            'commands_submitted': commands_submitted,
            'total_doorbell_accesses': stats['total_doorbell_accesses'],
            'thread_0_efficiency': stats['thread_0_efficiency'],
            'contention_stalls': stats['contention_stalls'],
            'total_utilization': stats['total_utilization']
        }
        
        print(f"   Commands Submitted: {commands_submitted}/640")
        print(f"   Doorbell Accesses: {stats['total_doorbell_accesses']}")
        print(f"   Thread 0 Efficiency: {stats['thread_0_efficiency']:.1%}")
        print(f"   Contention Stalls: {stats['contention_stalls']}")
        print()
    
    return results

def analyze_sq_scaling():
    """Analyze how performance scales with different numbers of SQs"""
    print("📈 SQ Scaling Analysis")
    print("=" * 80)
    
    env = simpy.Environment()
    sq_counts = [64, 128, 256, 512, 1024, 2048]
    scaling_results = {}
    
    for num_sqs in sq_counts:
        print(f"📊 Testing with {num_sqs} Submission Queues...")
        
        controller = NVMeDoorbellController(
            env=env,
            num_submission_queues=num_sqs,
            sq_depth=64,
            access_pattern=DoorbellAccessPattern.THREAD_0_LEADER
        )
        
        # Fixed workload: 50 warps = 1600 threads
        start_time = time.time()
        
        for warp_id in range(50):
            for thread_id in range(32):
                from base.packet import Packet
                command = Packet(
                    id=f"scaling_test_{num_sqs}_w{warp_id}_t{thread_id}",
                    type="nvme_write",
                    address=warp_id * 1024 + thread_id * 32,
                    size=4
                )
                
                controller.submit_command_with_thread0_optimization(
                    command, thread_id, warp_id, is_thread_leader=(thread_id == 0)
                )
        
        execution_time = time.time() - start_time
        stats = controller.get_system_stats()
        
        # Calculate efficiency metrics
        threads_per_sq = 1600 / num_sqs
        theoretical_optimal_doorbell_accesses = 50  # One per warp (thread 0)
        actual_doorbell_accesses = stats['total_doorbell_accesses']
        efficiency = theoretical_optimal_doorbell_accesses / max(1, actual_doorbell_accesses)
        
        scaling_results[num_sqs] = {
            'execution_time': execution_time,
            'threads_per_sq': threads_per_sq,
            'doorbell_accesses': actual_doorbell_accesses,
            'efficiency': efficiency,
            'contention_stalls': stats['contention_stalls'],
            'utilization': stats['total_utilization']
        }
        
        print(f"   Threads per SQ: {threads_per_sq:.1f}")
        print(f"   Doorbell Accesses: {actual_doorbell_accesses}")
        print(f"   Efficiency: {efficiency:.1%}")
        print(f"   Stalls: {stats['contention_stalls']}")
        print()
    
    return scaling_results

def main():
    """Main test execution"""
    print("🎯 NVMe Doorbell Optimization Comprehensive Test Suite")
    print("=" * 80)
    print()
    
    # Run all test scenarios
    optimization_results = test_nvme_doorbell_optimization()
    print()
    
    strategy_results = test_sq_mapping_strategies()
    print()
    
    scaling_results = analyze_sq_scaling()
    print()
    
    # Generate comprehensive report
    print("📊 COMPREHENSIVE ANALYSIS REPORT")
    print("=" * 80)
    
    # Optimization effectiveness
    thread_0_savings = optimization_results['thread_0_optimal']['bandwidth_savings']
    print(f"🚀 Thread 0 Leadership Optimization:")
    print(f"   Bandwidth Savings: {thread_0_savings:.1f}%")
    print(f"   Doorbell Reduction: {32 - optimization_results['thread_0_optimal']['total_doorbell_writes']}/32 threads")
    
    # Fallback resilience
    fallback_success_rate = optimization_results['fallback_patterns']['successful_accesses'] / 32
    print(f"\n🔄 Fallback Pattern Resilience:")
    print(f"   Success Rate: {fallback_success_rate:.1%}")
    print(f"   Leadership Transfers: {optimization_results['fallback_patterns']['leadership_transfers']}")
    
    # SQ contention impact
    contention_stall_rate = optimization_results['sq_contention']['total_stalls'] / optimization_results['sq_contention']['total_commands']
    print(f"\n⚠️  SQ Contention Analysis:")
    print(f"   Stall Rate (128 SQs): {contention_stall_rate:.1%}")
    print(f"   Utilization: {optimization_results['sq_contention']['utilization']:.1%}")
    
    # Strategy comparison
    best_strategy = min(strategy_results.items(), key=lambda x: x[1]['contention_stalls'])
    print(f"\n🏆 Best Strategy: {best_strategy[0].upper()}")
    print(f"   Contention Stalls: {best_strategy[1]['contention_stalls']}")
    print(f"   Thread 0 Efficiency: {best_strategy[1]['thread_0_efficiency']:.1%}")
    
    # Scaling insights
    optimal_sq_count = min(scaling_results.items(), key=lambda x: x[1]['contention_stalls'])[0]
    print(f"\n📈 Optimal SQ Configuration:")
    print(f"   Recommended SQ Count: {optimal_sq_count}")
    print(f"   Threads per SQ: {scaling_results[optimal_sq_count]['threads_per_sq']:.1f}")
    print(f"   Efficiency: {scaling_results[optimal_sq_count]['efficiency']:.1%}")
    
    # Save results
    all_results = {
        'optimization_test': optimization_results,
        'strategy_comparison': strategy_results,
        'scaling_analysis': scaling_results,
        'timestamp': time.time()
    }
    
    with open('nvme_doorbell_optimization_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\n📄 Results saved to nvme_doorbell_optimization_results.json")
    
    return all_results

if __name__ == "__main__":
    main()