import simpy
from enum import Enum
from src.base.packet import Packet

class ThreadState(Enum):
    ACTIVE = "active"
    STALLED = "stalled"
    COMPLETED = "completed"
    DIVERGED = "diverged"

class GPUThread:
    """
    Individual GPU thread model within a warp.
    Tracks thread state, program counter, and register context.
    """
    def __init__(self, thread_id, warp_id):
        self.thread_id = thread_id
        self.warp_id = warp_id
        self.state = ThreadState.ACTIVE
        self.program_counter = 0
        self.registers = {}  # Thread-local register storage
        self.predicate_mask = True  # For conditional execution
        self.divergence_stack = []  # For handling branch divergence
        
    def set_register(self, reg_id, value):
        """Set register value for this thread"""
        self.registers[reg_id] = value
    
    def get_register(self, reg_id):
        """Get register value for this thread"""
        return self.registers.get(reg_id, 0)
    
    def set_predicate(self, mask):
        """Set predicate mask for conditional execution"""
        self.predicate_mask = mask
    
    def is_active(self):
        """Check if thread is actively executing"""
        return self.state == ThreadState.ACTIVE and self.predicate_mask
    
    def push_divergence(self, pc, mask):
        """Push divergence information onto stack"""
        self.divergence_stack.append((pc, mask))
    
    def pop_divergence(self):
        """Pop divergence information from stack"""
        if self.divergence_stack:
            return self.divergence_stack.pop()
        return None
    
    def __repr__(self):
        return f"Thread(id={self.thread_id}, warp={self.warp_id}, state={self.state.value}, pc={self.program_counter})"


class ThreadContext:
    """
    Manages execution context for multiple threads within a warp.
    Handles register allocation and thread synchronization.
    """
    def __init__(self, env, warp_id, num_threads=32):
        self.env = env
        self.warp_id = warp_id
        self.num_threads = num_threads
        self.threads = []
        
        # Create individual threads
        for i in range(num_threads):
            thread = GPUThread(thread_id=i, warp_id=warp_id)
            self.threads.append(thread)
        
        # Shared register file allocation tracking
        self.register_allocator = RegisterAllocator(env)
        
    def get_active_threads(self):
        """Get list of currently active threads"""
        return [t for t in self.threads if t.is_active()]
    
    def get_thread_mask(self):
        """Get bitmask of active threads"""
        mask = 0
        for i, thread in enumerate(self.threads):
            if thread.is_active():
                mask |= (1 << i)
        return mask
    
    def set_all_predicate(self, condition):
        """Set predicate mask for all threads based on condition"""
        for thread in self.threads:
            thread.set_predicate(condition(thread))
    
    def sync_threads(self):
        """Synchronize all threads in the warp (warp-level sync)"""
        # Wait for all active threads to reach sync point
        for thread in self.threads:
            if thread.state == ThreadState.ACTIVE:
                thread.state = ThreadState.STALLED
        
        # All threads resume together
        for thread in self.threads:
            if thread.state == ThreadState.STALLED:
                thread.state = ThreadState.ACTIVE
    
    def handle_branch_divergence(self, branch_condition):
        """Handle branch divergence within the warp"""
        true_threads = []
        false_threads = []
        
        for thread in self.threads:
            if thread.is_active():
                if branch_condition(thread):
                    true_threads.append(thread)
                else:
                    false_threads.append(thread)
        
        # Return separate execution paths
        return true_threads, false_threads


class RegisterAllocator:
    """
    Simulates GPU register file allocation and access.
    Tracks register usage per thread and handles register pressure.
    """
    def __init__(self, env, registers_per_sm=65536, threads_per_sm=2048):
        self.env = env
        self.total_registers = registers_per_sm  # 64K 32-bit registers for H100/B200
        self.threads_per_sm = threads_per_sm
        self.max_registers_per_thread = self.total_registers // self.threads_per_sm
        
        self.allocated_registers = {}  # thread_id -> num_registers
        self.register_pressure = 0
        
    def allocate_registers(self, thread_id, num_registers):
        """Allocate registers for a thread"""
        if num_registers > self.max_registers_per_thread:
            raise ValueError(f"Thread {thread_id} requesting too many registers: {num_registers}")
        
        available = self.total_registers - sum(self.allocated_registers.values())
        if num_registers > available:
            raise ValueError(f"Not enough registers available: requested {num_registers}, available {available}")
        
        self.allocated_registers[thread_id] = num_registers
        self.register_pressure = sum(self.allocated_registers.values()) / self.total_registers
        
    def deallocate_registers(self, thread_id):
        """Deallocate registers for a thread"""
        if thread_id in self.allocated_registers:
            del self.allocated_registers[thread_id]
            self.register_pressure = sum(self.allocated_registers.values()) / self.total_registers
    
    def get_register_pressure(self):
        """Get current register pressure (0.0 to 1.0)"""
        return self.register_pressure
    
    def can_allocate(self, num_registers):
        """Check if registers can be allocated"""
        available = self.total_registers - sum(self.allocated_registers.values())
        return num_registers <= available
    
    def account_storage_optimization(self, threads_saved, registers_per_thread=32):
        """Account for register savings from thread 0 storage access pattern"""
        # When only thread 0 accesses storage, other threads don't need storage-related registers
        self.storage_register_savings += (threads_saved - 1) * registers_per_thread
        return self.storage_register_savings
    
    def get_optimized_register_pressure(self):
        """Get register pressure accounting for storage access optimizations"""
        effective_usage = sum(self.allocated_registers.values()) - self.storage_register_savings
        return effective_usage / self.total_registers