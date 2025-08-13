#!/usr/bin/env python3
import simpy
import random
from enum import Enum
from src.base.packet import Packet
from src.components.storage.nvme_doorbell_system import NVMeDoorbellController, DoorbellAccessPattern

class FallbackAccessPattern(Enum):
    """Fallback patterns when Thread 0 is unavailable"""
    NEXT_ACTIVE_LEADER = "next_active_leader"        # Next available thread becomes leader
    MULTI_LEADER = "multi_leader"                    # Multiple threads share leadership
    SQ_STRIPING = "sq_striping"                      # Stripe across multiple SQs
    ADAPTIVE_GROUPING = "adaptive_grouping"          # Dynamic grouping based on active threads

class ThreadState(Enum):
    """Thread execution states"""
    ACTIVE = "active"
    DIVERGED = "diverged"
    STALLED = "stalled"
    TERMINATED = "terminated"

class WarpExecutionContext:
    """
    Warp execution context tracking thread states and leadership
    """
    def __init__(self, warp_id, num_threads=32):
        self.warp_id = warp_id
        self.num_threads = num_threads
        self.thread_states = {i: ThreadState.ACTIVE for i in range(num_threads)}
        self.current_leader = 0
        self.leadership_history = [0]
        self.divergence_mask = [True] * num_threads
        
    def update_thread_state(self, thread_id, new_state):
        """Update thread state and recalculate leadership if needed"""
        if thread_id < self.num_threads:
            old_state = self.thread_states[thread_id]
            self.thread_states[thread_id] = new_state
            
            # If current leader becomes inactive, find new leader
            if thread_id == self.current_leader and new_state != ThreadState.ACTIVE:
                self._elect_new_leader()
                
    def _elect_new_leader(self):
        """Elect new leader from active threads"""
        # Find lowest numbered active thread
        for thread_id in range(self.num_threads):
            if self.thread_states[thread_id] == ThreadState.ACTIVE:
                old_leader = self.current_leader
                self.current_leader = thread_id
                self.leadership_history.append(thread_id)
                print(f"Warp {self.warp_id}: Leadership transferred from Thread {old_leader} to Thread {thread_id}")
                return thread_id
        
        # No active threads found
        self.current_leader = None
        return None
    
    def get_active_threads(self):
        """Get list of currently active thread IDs"""
        return [tid for tid, state in self.thread_states.items() if state == ThreadState.ACTIVE]
    
    def get_convergent_threads(self):
        """Get threads that are convergent (following same execution path)"""
        active_threads = self.get_active_threads()
        return [tid for tid in active_threads if self.divergence_mask[tid]]
    
    def simulate_branch_divergence(self, divergent_threads):
        """Simulate branch divergence affecting specified threads"""
        for thread_id in divergent_threads:
            if thread_id < self.num_threads:
                self.divergence_mask[thread_id] = False
                if self.thread_states[thread_id] == ThreadState.ACTIVE:
                    self.thread_states[thread_id] = ThreadState.DIVERGED
        
        # Check if leader is affected
        if self.current_leader in divergent_threads:
            self._elect_new_leader()

class NVMeFallbackController:
    """
    NVMe Doorbell Controller with fallback patterns for Thread 0 unavailability
    """
    def __init__(self, env, num_submission_queues=1024, sq_depth=64, 
                 fallback_pattern=FallbackAccessPattern.NEXT_ACTIVE_LEADER):
        self.env = env
        self.num_submission_queues = num_submission_queues
        self.sq_depth = sq_depth
        self.fallback_pattern = fallback_pattern
        
        # Base doorbell controller
        self.doorbell_controller = NVMeDoorbellController(
            env, num_submission_queues, sq_depth, 
            DoorbellAccessPattern.THREAD_0_LEADER
        )
        
        # Fallback management
        self.warp_contexts = {}  # warp_id -> WarpExecutionContext
        self.fallback_activations = 0
        self.leadership_transfers = 0
        self.multi_leader_operations = 0
        self.sq_striping_operations = 0
        
        # Performance tracking
        self.fallback_overhead = 0
        self.traditional_overhead = 0
        
    def get_or_create_warp_context(self, warp_id):
        """Get or create warp execution context"""
        if warp_id not in self.warp_contexts:
            self.warp_contexts[warp_id] = WarpExecutionContext(warp_id)
        return self.warp_contexts[warp_id]
    
    def submit_storage_command_with_fallback(self, command_packet, thread_id, warp_id, 
                                           thread_states=None):
        """
        Submit storage command with intelligent fallback when Thread 0 unavailable
        """
        warp_context = self.get_or_create_warp_context(warp_id)
        
        # Update thread states if provided
        if thread_states:
            for tid, state in thread_states.items():
                warp_context.update_thread_state(tid, state)
        
        # Check if Thread 0 is available
        thread_0_available = (warp_context.thread_states.get(0) == ThreadState.ACTIVE)
        current_leader = warp_context.current_leader
        
        if thread_0_available and thread_id == 0:
            # Standard Thread 0 leadership path
            return self._submit_with_thread0_leadership(command_packet, thread_id, warp_id, warp_context)
        
        elif not thread_0_available or current_leader != 0:
            # Fallback patterns activated
            self.fallback_activations += 1
            return self._submit_with_fallback_pattern(command_packet, thread_id, warp_id, warp_context)
        
        else:
            # Non-leader thread in normal operation
            return self._submit_as_follower(command_packet, thread_id, warp_id, warp_context)
    
    def _submit_with_thread0_leadership(self, command_packet, thread_id, warp_id, warp_context):
        """Standard Thread 0 leadership submission"""
        result = self.doorbell_controller.submit_command_with_thread0_optimization(
            command_packet, thread_id, warp_id, is_thread_leader=True
        )
        result['access_pattern'] = 'thread_0_leadership'
        return result
    
    def _submit_with_fallback_pattern(self, command_packet, thread_id, warp_id, warp_context):
        """Handle submission using fallback patterns"""
        
        if self.fallback_pattern == FallbackAccessPattern.NEXT_ACTIVE_LEADER:
            return self._next_active_leader_pattern(command_packet, thread_id, warp_id, warp_context)
            
        elif self.fallback_pattern == FallbackAccessPattern.MULTI_LEADER:
            return self._multi_leader_pattern(command_packet, thread_id, warp_id, warp_context)
            
        elif self.fallback_pattern == FallbackAccessPattern.SQ_STRIPING:
            return self._sq_striping_pattern(command_packet, thread_id, warp_id, warp_context)
            
        elif self.fallback_pattern == FallbackAccessPattern.ADAPTIVE_GROUPING:
            return self._adaptive_grouping_pattern(command_packet, thread_id, warp_id, warp_context)
        
        # Default fallback
        return self._next_active_leader_pattern(command_packet, thread_id, warp_id, warp_context)
    
    def _next_active_leader_pattern(self, command_packet, thread_id, warp_id, warp_context):
        """Next active thread becomes the leader"""
        current_leader = warp_context.current_leader
        
        if thread_id == current_leader:
            # This thread is the new leader
            self.leadership_transfers += 1
            
            result = self.doorbell_controller.submit_command_with_thread0_optimization(
                command_packet, thread_id, warp_id, is_thread_leader=True
            )
            
            # Additional overhead for leadership transfer
            transfer_overhead = 3  # cycles for leadership establishment
            result['doorbell_latency'] = result.get('doorbell_latency', 0) + transfer_overhead
            result['total_latency'] = result.get('total_latency', 0) + transfer_overhead
            result['access_pattern'] = 'next_active_leader'
            result['leadership_transfer'] = True
            
            self.fallback_overhead += transfer_overhead
            return result
        else:
            # Follower thread
            return self._submit_as_follower(command_packet, thread_id, warp_id, warp_context)
    
    def _multi_leader_pattern(self, command_packet, thread_id, warp_id, warp_context):
        """Multiple threads act as leaders for different SQ ranges"""
        active_threads = warp_context.get_active_threads()
        
        if not active_threads:
            return {'success': False, 'reason': 'NO_ACTIVE_THREADS'}
        
        # Divide SQs among active threads
        threads_per_sq_group = max(1, len(active_threads) // 4)  # Up to 4 leader groups
        leader_group = thread_id // threads_per_sq_group
        is_group_leader = (thread_id % threads_per_sq_group == 0) and (thread_id in active_threads)
        
        if is_group_leader:
            self.multi_leader_operations += 1
            
            # Select SQ based on leader group
            base_sq = (warp_id * 4 + leader_group) % self.num_submission_queues
            
            result = self.doorbell_controller.submit_command_with_thread0_optimization(
                command_packet, thread_id, warp_id, is_thread_leader=True
            )
            
            # Override SQ selection
            result['sq_id'] = base_sq
            result['access_pattern'] = 'multi_leader'
            result['leader_group'] = leader_group
            
            # Reduced broadcast overhead (smaller groups)
            result['broadcast_latency'] = max(1, threads_per_sq_group // 4)
            result['total_latency'] = result.get('doorbell_latency', 0) + result['broadcast_latency']
            
            return result
        else:
            # Follower in multi-leader pattern
            return {
                'success': True,
                'sq_id': -1,  # Determined by group leader
                'doorbell_latency': 0,
                'broadcast_latency': 1,
                'total_latency': 1,
                'access_pattern': 'multi_leader_follower',
                'received_broadcast': True
            }
    
    def _sq_striping_pattern(self, command_packet, thread_id, warp_id, warp_context):
        """Stripe storage access across multiple SQs to reduce contention"""
        active_threads = warp_context.get_active_threads()
        
        if thread_id not in active_threads:
            return {'success': False, 'reason': 'THREAD_INACTIVE'}
        
        self.sq_striping_operations += 1
        
        # Each active thread uses different SQ
        sq_id = (warp_id * 32 + thread_id) % self.num_submission_queues
        
        # Direct submission without warp coordination
        sq = self.doorbell_controller.submission_queues[sq_id]
        success = sq.submit_command(command_packet, thread_id, warp_id)
        
        if success:
            doorbell_latency = sq.ring_doorbell(sq.tail, thread_id)
            
            return {
                'success': True,
                'sq_id': sq_id,
                'doorbell_latency': doorbell_latency,
                'broadcast_latency': 0,  # No broadcast needed
                'total_latency': doorbell_latency,
                'access_pattern': 'sq_striping',
                'independent_access': True
            }
        else:
            return {
                'success': False,
                'reason': 'SQ_FULL',
                'sq_id': sq_id,
                'access_pattern': 'sq_striping'
            }
    
    def _adaptive_grouping_pattern(self, command_packet, thread_id, warp_id, warp_context):
        """Adaptive grouping based on current thread convergence"""
        convergent_threads = warp_context.get_convergent_threads()
        
        if not convergent_threads:
            # Fall back to SQ striping if no convergent threads
            return self._sq_striping_pattern(command_packet, thread_id, warp_id, warp_context)
        
        # Group convergent threads, use lowest ID as leader
        group_leader = min(convergent_threads)
        
        if thread_id == group_leader:
            # Adaptive group leader
            result = self.doorbell_controller.submit_command_with_thread0_optimization(
                command_packet, thread_id, warp_id, is_thread_leader=True
            )
            
            # Broadcast only to convergent threads
            convergent_count = len(convergent_threads)
            result['broadcast_latency'] = max(1, convergent_count // 4)
            result['total_latency'] = result.get('doorbell_latency', 0) + result['broadcast_latency']
            result['access_pattern'] = 'adaptive_grouping'
            result['group_size'] = convergent_count
            
            return result
        
        elif thread_id in convergent_threads:
            # Convergent follower
            return {
                'success': True,
                'sq_id': -1,
                'doorbell_latency': 0,
                'broadcast_latency': 1,
                'total_latency': 1,
                'access_pattern': 'adaptive_grouping_follower',
                'received_broadcast': True
            }
        else:
            # Divergent thread - independent access
            return self._sq_striping_pattern(command_packet, thread_id, warp_id, warp_context)
    
    def _submit_as_follower(self, command_packet, thread_id, warp_id, warp_context):
        """Submit as follower thread (receives broadcast)"""
        return {
            'success': True,
            'sq_id': -1,  # Determined by leader
            'doorbell_latency': 0,
            'broadcast_latency': 2,
            'total_latency': 2,
            'access_pattern': 'follower',
            'received_broadcast': True
        }
    
    def get_fallback_statistics(self):
        """Get comprehensive fallback pattern statistics"""
        base_stats = self.doorbell_controller.get_system_stats()
        
        return {
            **base_stats,
            'fallback_pattern': self.fallback_pattern.value,
            'fallback_activations': self.fallback_activations,
            'leadership_transfers': self.leadership_transfers,
            'multi_leader_operations': self.multi_leader_operations,
            'sq_striping_operations': self.sq_striping_operations,
            'fallback_overhead_cycles': self.fallback_overhead,
            'traditional_overhead_cycles': self.traditional_overhead,
            'warp_contexts': len(self.warp_contexts)
        }

class NVMeFallbackBenchmark:
    """
    Benchmark different fallback patterns under various thread availability scenarios
    """
    def __init__(self, env):
        self.env = env
        self.results = {}
    
    def benchmark_pattern(self, pattern, num_warps=10, thread_failure_rate=0.1):
        """Benchmark a specific fallback pattern"""
        controller = NVMeFallbackController(
            self.env, 
            num_submission_queues=1024,
            fallback_pattern=pattern
        )
        
        results = []
        
        for warp_id in range(num_warps):
            # Simulate random thread failures
            thread_states = {}
            for thread_id in range(32):
                if random.random() < thread_failure_rate:
                    thread_states[thread_id] = random.choice([
                        ThreadState.DIVERGED, ThreadState.STALLED
                    ])
                else:
                    thread_states[thread_id] = ThreadState.ACTIVE
            
            # Submit storage commands for this warp
            warp_results = []
            for thread_id in range(32):
                command = Packet(
                    id=f"warp_{warp_id}_thread_{thread_id}",
                    type="nvme_write",
                    address=thread_id * 4096,
                    size=4096
                )
                
                result = controller.submit_storage_command_with_fallback(
                    command, thread_id, warp_id, thread_states
                )
                warp_results.append(result)
            
            results.append(warp_results)
        
        # Analyze results
        stats = controller.get_fallback_statistics()
        
        return {
            'pattern': pattern.value,
            'stats': stats,
            'detailed_results': results
        }
    
    def run_comprehensive_benchmark(self):
        """Run benchmark across all fallback patterns"""
        patterns = [
            FallbackAccessPattern.NEXT_ACTIVE_LEADER,
            FallbackAccessPattern.MULTI_LEADER,
            FallbackAccessPattern.SQ_STRIPING,
            FallbackAccessPattern.ADAPTIVE_GROUPING
        ]
        
        print("🔬 NVMe Fallback Pattern Benchmark")
        print("=" * 50)
        
        for pattern in patterns:
            result = self.benchmark_pattern(pattern)
            self.results[pattern.value] = result
            
            stats = result['stats']
            print(f"\n📊 {pattern.value.upper()}:")
            print(f"   Fallback Activations: {stats['fallback_activations']}")
            print(f"   Leadership Transfers: {stats['leadership_transfers']}")
            print(f"   Total Doorbell Accesses: {stats['total_doorbell_accesses']}")
            print(f"   Contention Stalls: {stats['contention_stalls']}")
            print(f"   Fallback Overhead: {stats['fallback_overhead_cycles']} cycles")
        
        return self.results

# Example usage
if __name__ == "__main__":
    env = simpy.Environment()
    benchmark = NVMeFallbackBenchmark(env)
    results = benchmark.run_comprehensive_benchmark()
    
    print(f"\n📈 Benchmark Results Summary:")
    for pattern, result in results.items():
        stats = result['stats']
        efficiency = (stats['total_commands'] - stats['contention_stalls']) / max(1, stats['total_commands'])
        print(f"   {pattern}: {efficiency:.1%} efficiency")