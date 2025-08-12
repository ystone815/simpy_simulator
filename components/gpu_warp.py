import simpy
import random
from enum import Enum
from base.packet import Packet
from components.gpu_thread import ThreadContext, ThreadState

class WarpState(Enum):
    READY = "ready"              # Ready to execute
    ACTIVE = "active"            # Currently executing
    STALLED = "stalled"          # Waiting for memory/dependency
    DIVERGED = "diverged"        # Branch divergence active
    COMPLETED = "completed"      # Execution finished
    WAITING_BARRIER = "waiting_barrier"  # Waiting at sync barrier

class WarpInstruction:
    """Represents a CUDA instruction for warp execution"""
    def __init__(self, opcode, operands=None, latency=1, throughput=1, divergent=False):
        self.opcode = opcode
        self.operands = operands or []
        self.latency = latency  # Execution latency in cycles
        self.throughput = throughput  # Instructions per cycle
        self.memory_access = False
        self.divergent = divergent  # Can cause thread divergence

class Warp:
    """
    CUDA Warp model - group of 32 threads executing in SIMT fashion.
    Handles thread divergence, synchronization, and instruction execution.
    """
    def __init__(self, env, warp_id, sm_id, num_threads=32):
        self.env = env
        self.warp_id = warp_id
        self.sm_id = sm_id
        self.num_threads = num_threads
        self.state = WarpState.READY
        
        # Thread management
        self.thread_context = ThreadContext(env, warp_id, num_threads)
        self.active_mask = (1 << num_threads) - 1  # All threads active initially
        
        # Execution state
        self.program_counter = 0
        self.instruction_queue = []
        self.divergence_stack = []  # For handling nested divergence
        
        # Performance tracking
        self.cycles_executed = 0
        self.instructions_executed = 0
        self.stall_cycles = 0
        self.divergence_events = 0
        
        # Synchronization
        self.barrier_count = 0
        self.sync_mask = 0
        
        # Storage access optimization
        self.storage_access_stats = {
            'thread_0_accesses': 0,
            'broadcast_operations': 0,
            'bandwidth_saved': 0
        }
        
    def add_instruction(self, instruction):
        """Add instruction to warp's execution queue"""
        self.instruction_queue.append(instruction)
    
    def execute_instruction(self, instruction):
        """Execute a single instruction across all active threads"""
        active_threads = self.thread_context.get_active_threads()
        
        if not active_threads:
            return 0  # No active threads
        
        execution_cycles = instruction.latency
        
        if instruction.opcode == "BRANCH":
            # Handle branch divergence
            self._handle_branch_divergence(instruction)
        elif instruction.opcode == "SYNC":
            # Warp-level synchronization
            self._handle_warp_sync()
        elif instruction.opcode == "LOAD" or instruction.opcode == "STORE":
            # Memory access
            execution_cycles += self._handle_memory_access(instruction)
        else:
            # Regular arithmetic/logic operation
            for thread in active_threads:
                self._execute_thread_instruction(thread, instruction)
        
        self.instructions_executed += 1
        self.cycles_executed += execution_cycles
        
        return execution_cycles
    
    def _execute_thread_instruction(self, thread, instruction):
        """Execute instruction on a single thread"""
        if instruction.opcode == "ADD":
            # Example: ADD R1, R2, R3
            if len(instruction.operands) >= 3:
                dst, src1, src2 = instruction.operands[:3]
                val1 = thread.get_register(src1)
                val2 = thread.get_register(src2)
                thread.set_register(dst, val1 + val2)
        
        elif instruction.opcode == "MUL":
            # Example: MUL R1, R2, R3
            if len(instruction.operands) >= 3:
                dst, src1, src2 = instruction.operands[:3]
                val1 = thread.get_register(src1)
                val2 = thread.get_register(src2)
                thread.set_register(dst, val1 * val2)
        
        thread.program_counter += 1
    
    def _handle_branch_divergence(self, instruction):
        """Handle branch divergence when threads take different paths"""
        condition = instruction.operands[0] if instruction.operands else None
        
        # Evaluate condition for each thread
        true_threads, false_threads = self.thread_context.handle_branch_divergence(
            lambda t: t.get_register(condition) != 0 if condition else random.choice([True, False])
        )
        
        if true_threads and false_threads:
            # Divergence occurred
            self.divergence_events += 1
            self.state = WarpState.DIVERGED
            
            # Save current state for reconvergence
            self.divergence_stack.append({
                'pc': self.program_counter,
                'active_mask': self.active_mask,
                'true_threads': [t.thread_id for t in true_threads],
                'false_threads': [t.thread_id for t in false_threads]
            })
            
            # Execute true path first
            for thread in false_threads:
                thread.state = ThreadState.DIVERGED
            
            # Update active mask for true path
            self._update_active_mask(true_threads)
        
        self.program_counter += 1
    
    def _handle_warp_sync(self):
        """Handle warp-level synchronization (__syncwarp)"""
        self.thread_context.sync_threads()
        self.state = WarpState.WAITING_BARRIER
        
        # All active threads must reach this point
        active_threads = self.thread_context.get_active_threads()
        self.sync_mask = sum(1 << t.thread_id for t in active_threads)
    
    def _handle_memory_access(self, instruction):
        """Handle memory access instruction with coalescing"""
        active_threads = self.thread_context.get_active_threads()
        
        # Simulate memory coalescing
        addresses = []
        for thread in active_threads:
            if instruction.operands:
                base_addr = thread.get_register(instruction.operands[0])
                addresses.append(base_addr + thread.thread_id * 4)  # Assuming 4-byte elements
        
        # Check for coalesced access
        coalesced = self._check_memory_coalescing(addresses)
        
        if coalesced:
            # Single memory transaction
            return 100  # Coalesced access latency
        else:
            # Multiple memory transactions
            num_transactions = len(set(addr // 128 for addr in addresses))  # 128-byte cache lines
            return 100 + (num_transactions - 1) * 50  # Additional latency per transaction
    
    def _check_memory_coalescing(self, addresses):
        """Check if memory addresses can be coalesced into single transaction"""
        if not addresses:
            return True
        
        # Check if all addresses fall within same 128-byte cache line
        cache_line_size = 128
        first_line = addresses[0] // cache_line_size
        return all(addr // cache_line_size == first_line for addr in addresses)
    
    def _update_active_mask(self, active_threads):
        """Update active thread mask"""
        self.active_mask = 0
        for thread in active_threads:
            self.active_mask |= (1 << thread.thread_id)
    
    def reconverge_threads(self):
        """Reconverge diverged threads at convergence point"""
        if self.divergence_stack:
            # Pop most recent divergence
            divergence_info = self.divergence_stack.pop()
            
            # Reactivate all threads
            for thread in self.thread_context.threads:
                if thread.state == ThreadState.DIVERGED:
                    thread.state = ThreadState.ACTIVE
            
            self.active_mask = divergence_info['active_mask']
            
            if not self.divergence_stack:
                self.state = WarpState.ACTIVE
    
    def is_ready_to_execute(self):
        """Check if warp is ready for instruction execution"""
        return (self.state in [WarpState.READY, WarpState.ACTIVE] and 
                self.thread_context.get_active_threads() and
                self.instruction_queue)
    
    def get_occupancy_contribution(self):
        """Calculate this warp's contribution to SM occupancy"""
        if self.state == WarpState.COMPLETED:
            return 0
        return 1  # Each active warp contributes 1 to occupancy
    
    def get_utilization(self):
        """Get warp utilization (active threads / total threads)"""
        active_count = len(self.thread_context.get_active_threads())
        return active_count / self.num_threads
    
    def get_warp_leader(self):
        """Get the warp leader (thread 0) if active"""
        if self.thread_context.threads and self.thread_context.threads[0].is_active():
            return self.thread_context.threads[0]
        return None
    
    def execute_storage_access(self, storage_request, broadcast_to_all=True):
        """Execute storage access using thread 0 as leader with broadcast"""
        leader = self.get_warp_leader()
        if not leader:
            return None
        
        # Only thread 0 performs the actual storage access
        self.storage_access_stats['thread_0_accesses'] += 1
        
        # Simulate storage access latency for leader
        access_cycles = storage_request.get('latency', 10)
        
        if broadcast_to_all:
            # Broadcast result to all active threads using shuffle
            active_threads = self.thread_context.get_active_threads()
            self.storage_access_stats['broadcast_operations'] += 1
            self.storage_access_stats['bandwidth_saved'] += (len(active_threads) - 1) * storage_request.get('size', 1)
            
            # Simulate broadcast latency (1 cycle per shuffle operation)
            broadcast_cycles = 1
            total_cycles = access_cycles + broadcast_cycles
        else:
            total_cycles = access_cycles
        
        return {
            'success': True,
            'cycles': total_cycles,
            'leader_thread': leader.thread_id,
            'threads_served': len(self.thread_context.get_active_threads()) if broadcast_to_all else 1
        }
    
    def get_performance_stats(self):
        """Get performance statistics for this warp"""
        ipc = self.instructions_executed / max(self.cycles_executed, 1)
        stall_ratio = self.stall_cycles / max(self.cycles_executed, 1)
        
        return {
            'warp_id': self.warp_id,
            'instructions_executed': self.instructions_executed,
            'cycles_executed': self.cycles_executed,
            'ipc': ipc,
            'stall_cycles': self.stall_cycles,
            'stall_ratio': stall_ratio,
            'divergence_events': self.divergence_events,
            'utilization': self.get_utilization(),
            'storage_access_stats': self.storage_access_stats
        }
    
    def __repr__(self):
        active_threads = len(self.thread_context.get_active_threads())
        return (f"Warp(id={self.warp_id}, sm={self.sm_id}, state={self.state.value}, "
                f"active_threads={active_threads}/{self.num_threads})")


class WarpLevelPrimitives:
    """
    Implementation of CUDA warp-level primitive operations.
    Includes shuffle, vote, and match operations.
    """
    
    @staticmethod
    def warp_shuffle(warp, src_thread_id, value):
        """Implement __shfl_sync - shuffle value from source thread"""
        if 0 <= src_thread_id < warp.num_threads:
            src_thread = warp.thread_context.threads[src_thread_id]
            return src_thread.get_register(value) if src_thread.is_active() else 0
        return 0
    
    @staticmethod
    def warp_vote_all(warp, predicate_reg):
        """Implement __all_sync - return true if predicate is true for all active threads"""
        active_threads = warp.thread_context.get_active_threads()
        return all(thread.get_register(predicate_reg) != 0 for thread in active_threads)
    
    @staticmethod
    def warp_vote_any(warp, predicate_reg):
        """Implement __any_sync - return true if predicate is true for any active thread"""
        active_threads = warp.thread_context.get_active_threads()
        return any(thread.get_register(predicate_reg) != 0 for thread in active_threads)
    
    @staticmethod
    def warp_ballot(warp, predicate_reg):
        """Implement __ballot_sync - return bitmask of threads where predicate is true"""
        ballot = 0
        for thread in warp.thread_context.threads:
            if thread.is_active() and thread.get_register(predicate_reg) != 0:
                ballot |= (1 << thread.thread_id)
        return ballot
    
    @staticmethod
    def warp_match_any(warp, value_reg):
        """Implement __match_any_sync - return mask of threads with same value"""
        active_threads = warp.thread_context.get_active_threads()
        if not active_threads:
            return 0
        
        # Group threads by their register values
        value_groups = {}
        for thread in active_threads:
            value = thread.get_register(value_reg)
            if value not in value_groups:
                value_groups[value] = []
            value_groups[value].append(thread.thread_id)
        
        # Return mask for the largest group (or first group found)
        largest_group = max(value_groups.values(), key=len)
        mask = 0
        for thread_id in largest_group:
            mask |= (1 << thread_id)
        
        return mask