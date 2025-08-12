import simpy
import random
from enum import Enum
from collections import deque
from base.packet import Packet
from components.gpu_warp import Warp, WarpState, WarpInstruction

class SMState(Enum):
    IDLE = "idle"
    ACTIVE = "active"
    STALLED = "stalled"
    FULL = "full"  # No more resources available

class ExecutionUnit(Enum):
    FP32_CORE = "fp32"
    INT32_CORE = "int32"
    TENSOR_CORE = "tensor"
    SPECIAL_FUNC = "sfu"
    LOAD_STORE = "ls"

class WarpScheduler:
    """
    Warp Scheduler - manages warp execution order and instruction issue.
    Each SM has 4 warp schedulers in H100/B200.
    """
    def __init__(self, env, scheduler_id, sm_id):
        self.env = env
        self.scheduler_id = scheduler_id
        self.sm_id = sm_id
        
        # Warp queues
        self.ready_warps = deque()  # Warps ready to execute
        self.stalled_warps = deque()  # Warps waiting for resources
        self.active_warp = None  # Currently executing warp
        
        # Scheduling policy
        self.scheduling_policy = "round_robin"  # round_robin, priority, oldest_first
        self.max_warps_per_scheduler = 16  # Max warps this scheduler can handle
        
        # Performance tracking
        self.instructions_issued = 0
        self.cycles_idle = 0
        self.cycles_active = 0
        
        self.action = env.process(self.run())
    
    def add_warp(self, warp):
        """Add warp to scheduler"""
        if len(self.ready_warps) + len(self.stalled_warps) >= self.max_warps_per_scheduler:
            return False  # Scheduler full
        
        if warp.is_ready_to_execute():
            self.ready_warps.append(warp)
        else:
            self.stalled_warps.append(warp)
        return True
    
    def run(self):
        """Main scheduling loop"""
        while True:
            yield self.env.timeout(1)  # One clock cycle
            
            if self.ready_warps:
                self.cycles_active += 1
                warp = self._select_next_warp()
                if warp and warp.instruction_queue:
                    instruction = warp.instruction_queue.pop(0)
                    execution_cycles = warp.execute_instruction(instruction)
                    self.instructions_issued += 1
                    
                    # Simulate instruction latency
                    if execution_cycles > 1:
                        yield self.env.timeout(execution_cycles - 1)
                    
                    # Check if warp should be moved to stalled queue
                    if not warp.is_ready_to_execute():
                        if warp in self.ready_warps:
                            self.ready_warps.remove(warp)
                        if warp.state != WarpState.COMPLETED:
                            self.stalled_warps.append(warp)
            else:
                self.cycles_idle += 1
            
            # Move stalled warps back to ready if they become ready
            ready_again = []
            for warp in list(self.stalled_warps):
                if warp.is_ready_to_execute():
                    ready_again.append(warp)
            
            for warp in ready_again:
                self.stalled_warps.remove(warp)
                self.ready_warps.append(warp)
    
    def _select_next_warp(self):
        """Select next warp to execute based on scheduling policy"""
        if not self.ready_warps:
            return None
        
        if self.scheduling_policy == "round_robin":
            warp = self.ready_warps.popleft()
            if warp.state != WarpState.COMPLETED and warp.instruction_queue:
                self.ready_warps.append(warp)  # Put back at end
            return warp
        
        elif self.scheduling_policy == "oldest_first":
            return min(self.ready_warps, key=lambda w: w.warp_id)
        
        else:  # Default to first available
            return self.ready_warps[0] if self.ready_warps else None
    
    def get_utilization(self):
        """Get scheduler utilization"""
        total_cycles = self.cycles_active + self.cycles_idle
        if total_cycles == 0:
            return 0
        return self.cycles_active / total_cycles


class StreamingMultiprocessor:
    """
    Streaming Multiprocessor (SM) - core execution unit of GPU.
    Manages warps, execution units, and memory hierarchy.
    """
    def __init__(self, env, sm_id, gpu_type="H100"):
        self.env = env
        self.sm_id = sm_id
        self.gpu_type = gpu_type
        self.state = SMState.IDLE
        
        # Architecture-specific configuration
        if gpu_type == "H100":
            self.max_warps = 64
            self.max_threads = 2048  # 64 warps * 32 threads
            self.num_schedulers = 4
            self.fp32_cores = 128
            self.tensor_cores = 4
            self.shared_memory_size = 228 * 1024  # 228 KB
        elif gpu_type == "B200":
            self.max_warps = 64  # Same as H100 for compute capability 10.0
            self.max_threads = 2048
            self.num_schedulers = 4
            self.fp32_cores = 128  # Estimated
            self.tensor_cores = 4  # Enhanced tensor cores
            self.shared_memory_size = 228 * 1024
        
        # Execution units
        self.execution_units = {
            ExecutionUnit.FP32_CORE: self.fp32_cores,
            ExecutionUnit.INT32_CORE: self.fp32_cores,  # Same as FP32
            ExecutionUnit.TENSOR_CORE: self.tensor_cores,
            ExecutionUnit.SPECIAL_FUNC: 4,  # Special function units
            ExecutionUnit.LOAD_STORE: 4,   # Load/store units
        }
        
        # Resource tracking
        self.available_units = self.execution_units.copy()
        
        # Warp schedulers
        self.warp_schedulers = []
        for i in range(self.num_schedulers):
            scheduler = WarpScheduler(env, scheduler_id=i, sm_id=sm_id)
            self.warp_schedulers.append(scheduler)
        
        # Warp management
        self.active_warps = []
        self.completed_warps = []
        self.warp_counter = 0
        
        # Memory subsystem (will be connected externally)
        self.l1_cache = None
        self.shared_memory = None
        self.register_file = None
        
        # Performance tracking
        self.total_instructions = 0
        self.total_cycles = 0
        self.stall_cycles = 0
        self.occupancy_samples = []
        
        self.action = env.process(self.run())
    
    def connect_memory_subsystem(self, l1_cache, shared_memory, register_file):
        """Connect memory subsystem components"""
        self.l1_cache = l1_cache
        self.shared_memory = shared_memory
        self.register_file = register_file
    
    def launch_warp(self, instructions):
        """Launch new warp with given instruction sequence"""
        if len(self.active_warps) >= self.max_warps:
            return False  # SM full
        
        warp = Warp(self.env, warp_id=self.warp_counter, sm_id=self.sm_id)
        self.warp_counter += 1
        
        # Add instructions to warp
        for instr in instructions:
            warp.add_instruction(instr)
        
        # Assign warp to least loaded scheduler
        scheduler = min(self.warp_schedulers, 
                       key=lambda s: len(s.ready_warps) + len(s.stalled_warps))
        
        if scheduler.add_warp(warp):
            self.active_warps.append(warp)
            self.state = SMState.ACTIVE
            return True
        
        return False
    
    def run(self):
        """Main SM execution loop"""
        while True:
            yield self.env.timeout(1)  # One clock cycle
            self.total_cycles += 1
            
            # Update occupancy
            active_warp_count = len([w for w in self.active_warps if w.state != WarpState.COMPLETED])
            occupancy = active_warp_count / self.max_warps
            self.occupancy_samples.append(occupancy)
            
            # Check for completed warps
            newly_completed = []
            for warp in self.active_warps:
                if warp.state == WarpState.COMPLETED:
                    newly_completed.append(warp)
            
            for warp in newly_completed:
                self.active_warps.remove(warp)
                self.completed_warps.append(warp)
            
            # Update SM state
            if not self.active_warps:
                self.state = SMState.IDLE
            elif len(self.active_warps) >= self.max_warps:
                self.state = SMState.FULL
            else:
                self.state = SMState.ACTIVE
    
    def execute_instruction(self, instruction, warp):
        """Execute instruction using appropriate execution unit"""
        required_unit = self._get_required_execution_unit(instruction)
        
        if self.available_units[required_unit] > 0:
            self.available_units[required_unit] -= 1
            
            # Execute instruction
            execution_cycles = instruction.latency
            self.total_instructions += 1
            
            # Restore execution unit after latency
            def restore_unit():
                yield self.env.timeout(execution_cycles)
                self.available_units[required_unit] += 1
            
            self.env.process(restore_unit())
            return execution_cycles
        else:
            # Execution unit busy, stall warp
            warp.state = WarpState.STALLED
            warp.stall_cycles += 1
            self.stall_cycles += 1
            return 0
    
    def _get_required_execution_unit(self, instruction):
        """Determine which execution unit is needed for instruction"""
        if instruction.opcode in ["ADD", "MUL", "MAD", "FMA"]:
            return ExecutionUnit.FP32_CORE
        elif instruction.opcode in ["IADD", "IMUL", "IMADD"]:
            return ExecutionUnit.INT32_CORE
        elif instruction.opcode in ["HMMA", "WMMA", "MMA"]:
            return ExecutionUnit.TENSOR_CORE
        elif instruction.opcode in ["SIN", "COS", "EXP", "LOG"]:
            return ExecutionUnit.SPECIAL_FUNC
        elif instruction.opcode in ["LOAD", "STORE"]:
            return ExecutionUnit.LOAD_STORE
        else:
            return ExecutionUnit.FP32_CORE  # Default
    
    def get_occupancy(self):
        """Get current occupancy (active warps / max warps)"""
        active_count = len([w for w in self.active_warps if w.state != WarpState.COMPLETED])
        return active_count / self.max_warps
    
    def get_average_occupancy(self):
        """Get average occupancy over time"""
        if not self.occupancy_samples:
            return 0
        return sum(self.occupancy_samples) / len(self.occupancy_samples)
    
    def get_ipc(self):
        """Get instructions per cycle"""
        if self.total_cycles == 0:
            return 0
        return self.total_instructions / self.total_cycles
    
    def get_stall_ratio(self):
        """Get stall cycle ratio"""
        if self.total_cycles == 0:
            return 0
        return self.stall_cycles / self.total_cycles
    
    def get_scheduler_utilizations(self):
        """Get utilization of each warp scheduler"""
        return [scheduler.get_utilization() for scheduler in self.warp_schedulers]
    
    def get_performance_stats(self):
        """Get comprehensive performance statistics"""
        return {
            'sm_id': self.sm_id,
            'gpu_type': self.gpu_type,
            'total_instructions': self.total_instructions,
            'total_cycles': self.total_cycles,
            'ipc': self.get_ipc(),
            'occupancy': self.get_occupancy(),
            'average_occupancy': self.get_average_occupancy(),
            'stall_ratio': self.get_stall_ratio(),
            'active_warps': len(self.active_warps),
            'completed_warps': len(self.completed_warps),
            'scheduler_utilizations': self.get_scheduler_utilizations(),
            'execution_unit_availability': self.available_units.copy()
        }
    
    def __repr__(self):
        return (f"SM(id={self.sm_id}, type={self.gpu_type}, state={self.state.value}, "
                f"active_warps={len(self.active_warps)}/{self.max_warps}, "
                f"occupancy={self.get_occupancy():.2f})")


class SMCluster:
    """
    Manages multiple SMs as a cluster for load balancing and coordination.
    """
    def __init__(self, env, cluster_id, num_sms, gpu_type="H100"):
        self.env = env
        self.cluster_id = cluster_id
        self.num_sms = num_sms
        self.gpu_type = gpu_type
        
        # Create SMs
        self.sms = []
        for i in range(num_sms):
            sm = StreamingMultiprocessor(env, sm_id=i, gpu_type=gpu_type)
            self.sms.append(sm)
    
    def launch_kernel(self, grid_size, block_size, instructions):
        """Launch CUDA kernel across SMs"""
        total_blocks = grid_size[0] * grid_size[1] * grid_size[2]
        threads_per_block = block_size[0] * block_size[1] * block_size[2]
        warps_per_block = (threads_per_block + 31) // 32  # Round up
        
        blocks_launched = 0
        sm_index = 0
        
        for block_id in range(total_blocks):
            # Find SM with available capacity
            attempts = 0
            while attempts < len(self.sms):
                sm = self.sms[sm_index]
                
                # Try to launch warps for this block
                warps_launched = 0
                for warp_id in range(warps_per_block):
                    if sm.launch_warp(instructions):
                        warps_launched += 1
                    else:
                        break
                
                if warps_launched == warps_per_block:
                    blocks_launched += 1
                    break
                
                sm_index = (sm_index + 1) % len(self.sms)
                attempts += 1
            
            if blocks_launched < block_id + 1:
                print(f"Warning: Could not launch all blocks. Launched {blocks_launched}/{total_blocks}")
                break
        
        return blocks_launched
    
    def get_cluster_stats(self):
        """Get performance statistics for entire cluster"""
        total_instructions = sum(sm.total_instructions for sm in self.sms)
        total_cycles = max(sm.total_cycles for sm in self.sms) if self.sms else 0
        avg_occupancy = sum(sm.get_occupancy() for sm in self.sms) / len(self.sms)
        
        return {
            'cluster_id': self.cluster_id,
            'num_sms': self.num_sms,
            'total_instructions': total_instructions,
            'total_cycles': total_cycles,
            'cluster_ipc': total_instructions / max(total_cycles, 1),
            'average_occupancy': avg_occupancy,
            'sm_stats': [sm.get_performance_stats() for sm in self.sms]
        }