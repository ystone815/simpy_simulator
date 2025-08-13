#!/usr/bin/env python3
import simpy
import random
from enum import Enum
from src.base.packet import Packet

class DoorbellAccessPattern(Enum):
    """NVMe doorbell access patterns"""
    THREAD_0_LEADER = "thread_0_leader"     # Thread 0 handles all doorbell writes
    ROUND_ROBIN = "round_robin"             # Threads rotate through available SQs
    HASH_BASED = "hash_based"               # Hash thread ID to SQ assignment
    DYNAMIC_LOAD_BALANCE = "dynamic_lb"     # Dynamic load balancing across SQs

class NVMeSubmissionQueue:
    """
    NVMe Submission Queue with doorbell register
    Models realistic SQ constraints and doorbell access patterns
    """
    def __init__(self, env, sq_id, queue_depth=64, doorbell_register_addr=None):
        self.env = env
        self.sq_id = sq_id
        self.queue_depth = queue_depth
        self.doorbell_register_addr = doorbell_register_addr or (0x1000 + sq_id * 8)
        
        # Queue state
        self.queue = simpy.Store(env, capacity=queue_depth)
        self.head = 0
        self.tail = 0
        self.outstanding_commands = 0
        
        # Access statistics
        self.doorbell_writes = 0
        self.queue_full_stalls = 0
        self.total_commands_submitted = 0
        
        # Thread access tracking
        self.thread_access_count = {}
        self.warp_access_count = {}
        
    def is_full(self):
        """Check if submission queue is full"""
        return len(self.queue.items) >= self.queue_depth
    
    def get_available_slots(self):
        """Get number of available slots in queue"""
        return self.queue_depth - len(self.queue.items)
    
    def submit_command(self, command_packet, thread_id=None, warp_id=None):
        """Submit command to SQ and update doorbell"""
        if self.is_full():
            self.queue_full_stalls += 1
            return False
        
        # Track thread/warp access patterns
        if thread_id is not None:
            self.thread_access_count[thread_id] = self.thread_access_count.get(thread_id, 0) + 1
        if warp_id is not None:
            self.warp_access_count[warp_id] = self.warp_access_count.get(warp_id, 0) + 1
        
        # Submit to queue (non-blocking check)
        if len(self.queue.items) < self.queue.capacity:
            self.queue.items.append(command_packet)
        self.tail = (self.tail + 1) % self.queue_depth
        self.total_commands_submitted += 1
        
        return True
    
    def ring_doorbell(self, new_tail_value, thread_id=None):
        """Ring doorbell to notify NVMe controller of new submissions"""
        self.doorbell_writes += 1
        self.tail = new_tail_value
        
        # Simulate doorbell register write latency
        doorbell_latency = 1  # 1 cycle for register write
        return doorbell_latency
    
    def get_queue_stats(self):
        """Get submission queue statistics"""
        return {
            'sq_id': self.sq_id,
            'queue_depth': self.queue_depth,
            'current_occupancy': len(self.queue.items),
            'utilization': len(self.queue.items) / self.queue_depth,
            'doorbell_writes': self.doorbell_writes,
            'queue_full_stalls': self.queue_full_stalls,
            'total_commands': self.total_commands_submitted,
            'thread_access_distribution': dict(self.thread_access_count),
            'warp_access_distribution': dict(self.warp_access_count)
        }

class NVMeDoorbellController:
    """
    NVMe Doorbell Controller managing multiple submission queues
    Handles thread-to-SQ mapping and access patterns
    """
    def __init__(self, env, num_submission_queues=1024, sq_depth=64, access_pattern=DoorbellAccessPattern.THREAD_0_LEADER):
        self.env = env
        self.num_submission_queues = num_submission_queues
        self.sq_depth = sq_depth
        self.access_pattern = access_pattern
        
        # Create submission queues
        self.submission_queues = []
        for sq_id in range(num_submission_queues):
            sq = NVMeSubmissionQueue(env, sq_id, sq_depth)
            self.submission_queues.append(sq)
        
        # Thread to SQ mapping strategies
        self.thread_sq_mapping = {}
        self.warp_sq_mapping = {}
        self.round_robin_counter = 0
        
        # Performance tracking
        self.total_doorbell_accesses = 0
        self.thread_0_doorbell_accesses = 0
        self.broadcast_operations = 0
        self.contention_stalls = 0
        
    def get_sq_for_thread(self, thread_id, warp_id=None):
        """Get appropriate SQ for thread based on access pattern"""
        if self.access_pattern == DoorbellAccessPattern.THREAD_0_LEADER:
            # All threads in warp use same SQ, assigned by thread 0
            if warp_id is not None:
                if warp_id not in self.warp_sq_mapping:
                    # Thread 0 of warp assigns SQ for entire warp
                    sq_id = hash(warp_id) % self.num_submission_queues
                    self.warp_sq_mapping[warp_id] = sq_id
                return self.warp_sq_mapping[warp_id]
            else:
                # Fallback to thread-based assignment
                sq_id = hash(thread_id) % self.num_submission_queues
                return sq_id
                
        elif self.access_pattern == DoorbellAccessPattern.ROUND_ROBIN:
            # Round robin assignment across SQs
            sq_id = self.round_robin_counter % self.num_submission_queues
            self.round_robin_counter += 1
            return sq_id
            
        elif self.access_pattern == DoorbellAccessPattern.HASH_BASED:
            # Hash-based assignment
            sq_id = hash(thread_id) % self.num_submission_queues
            return sq_id
            
        elif self.access_pattern == DoorbellAccessPattern.DYNAMIC_LOAD_BALANCE:
            # Find least loaded SQ
            min_occupancy = float('inf')
            best_sq_id = 0
            for sq_id, sq in enumerate(self.submission_queues):
                if len(sq.queue.items) < min_occupancy:
                    min_occupancy = len(sq.queue.items)
                    best_sq_id = sq_id
            return best_sq_id
        
        return 0  # Default fallback

    def submit_command_with_thread0_optimization(self, command_packet, thread_id, warp_id, is_thread_leader=False):
        """
        Submit command using Thread 0 leadership pattern for doorbell optimization
        """
        sq_id = self.get_sq_for_thread(thread_id, warp_id)
        sq = self.submission_queues[sq_id]
        
        if self.access_pattern == DoorbellAccessPattern.THREAD_0_LEADER:
            if is_thread_leader:
                # Thread 0 performs actual submission and doorbell write
                success = sq.submit_command(command_packet, thread_id, warp_id)
                if success:
                    doorbell_latency = sq.ring_doorbell(sq.tail, thread_id)
                    self.thread_0_doorbell_accesses += 1
                    self.total_doorbell_accesses += 1
                    
                    # Simulate broadcast to other threads in warp (31 threads)
                    broadcast_latency = 2  # Warp shuffle broadcast
                    self.broadcast_operations += 1
                    
                    return {
                        'success': True,
                        'sq_id': sq_id,
                        'doorbell_latency': doorbell_latency,
                        'broadcast_latency': broadcast_latency,
                        'total_latency': doorbell_latency + broadcast_latency,
                        'thread_0_optimization': True
                    }
                else:
                    self.contention_stalls += 1
                    return {
                        'success': False,
                        'reason': 'SQ_FULL',
                        'sq_id': sq_id,
                        'thread_0_optimization': True
                    }
            else:
                # Non-leader threads wait for broadcast from thread 0
                # They don't directly access doorbell
                return {
                    'success': True,
                    'sq_id': sq_id,
                    'doorbell_latency': 0,  # No direct doorbell access
                    'broadcast_latency': 2,  # Receive broadcast
                    'total_latency': 2,
                    'thread_0_optimization': True,
                    'received_broadcast': True
                }
        else:
            # Traditional approach - each thread accesses doorbell independently
            success = sq.submit_command(command_packet, thread_id, warp_id)
            if success:
                doorbell_latency = sq.ring_doorbell(sq.tail, thread_id)
                self.total_doorbell_accesses += 1
                
                return {
                    'success': True,
                    'sq_id': sq_id,
                    'doorbell_latency': doorbell_latency,
                    'broadcast_latency': 0,
                    'total_latency': doorbell_latency,
                    'thread_0_optimization': False
                }
            else:
                self.contention_stalls += 1
                return {
                    'success': False,
                    'reason': 'SQ_FULL',
                    'sq_id': sq_id,
                    'thread_0_optimization': False
                }

    def get_system_stats(self):
        """Get comprehensive system statistics"""
        sq_stats = [sq.get_queue_stats() for sq in self.submission_queues]
        
        total_utilization = sum(len(sq.queue.items) for sq in self.submission_queues) / (self.num_submission_queues * self.sq_depth)
        total_commands = sum(sq.total_commands_submitted for sq in self.submission_queues)
        total_stalls = sum(sq.queue_full_stalls for sq in self.submission_queues)
        
        thread_0_efficiency = 0
        if self.total_doorbell_accesses > 0:
            thread_0_efficiency = self.thread_0_doorbell_accesses / self.total_doorbell_accesses
        
        return {
            'access_pattern': self.access_pattern.value,
            'num_submission_queues': self.num_submission_queues,
            'total_doorbell_accesses': self.total_doorbell_accesses,
            'thread_0_doorbell_accesses': self.thread_0_doorbell_accesses,
            'thread_0_efficiency': thread_0_efficiency,
            'broadcast_operations': self.broadcast_operations,
            'contention_stalls': self.contention_stalls,
            'total_utilization': total_utilization,
            'total_commands': total_commands,
            'total_stalls': total_stalls,
            'sq_stats': sq_stats
        }

class NVMeDoorbellOptimizer:
    """
    NVMe Doorbell access optimizer that manages thread-to-SQ assignment
    and coordinates with GPU warp execution
    """
    def __init__(self, env, doorbell_controller, gpu_context=None):
        self.env = env
        self.doorbell_controller = doorbell_controller
        self.gpu_context = gpu_context
        
        # Optimization strategies
        self.warp_sq_cache = {}  # Cache SQ assignments per warp
        self.load_balancing_enabled = True
        self.thread_0_broadcast_enabled = True
        
    def optimize_storage_access(self, warp_id, storage_commands):
        """
        Optimize storage access for a warp using Thread 0 leadership
        """
        if not storage_commands:
            return []
        
        results = []
        
        if self.thread_0_broadcast_enabled:
            # Use Thread 0 leadership pattern
            for i, command in enumerate(storage_commands):
                is_thread_leader = (i == 0)  # First command represents thread 0
                
                result = self.doorbell_controller.submit_command_with_thread0_optimization(
                    command_packet=command,
                    thread_id=i,
                    warp_id=warp_id,
                    is_thread_leader=is_thread_leader
                )
                results.append(result)
        else:
            # Traditional approach - each thread independent
            for i, command in enumerate(storage_commands):
                result = self.doorbell_controller.submit_command_with_thread0_optimization(
                    command_packet=command,
                    thread_id=i,
                    warp_id=warp_id,
                    is_thread_leader=False
                )
                results.append(result)
        
        return results
    
    def get_optimization_efficiency(self):
        """Calculate optimization efficiency metrics"""
        stats = self.doorbell_controller.get_system_stats()
        
        # Calculate bandwidth savings from Thread 0 optimization
        if stats['total_doorbell_accesses'] > 0:
            bandwidth_savings = (stats['thread_0_doorbell_accesses'] * 31) / stats['total_doorbell_accesses']
            # 31 threads saved per Thread 0 access
        else:
            bandwidth_savings = 0
        
        # Calculate contention reduction
        contention_reduction = 1.0 - (stats['contention_stalls'] / max(1, stats['total_commands']))
        
        return {
            'bandwidth_savings_percent': bandwidth_savings * 100,
            'contention_reduction_percent': contention_reduction * 100,
            'thread_0_efficiency': stats['thread_0_efficiency'] * 100,
            'broadcast_operations': stats['broadcast_operations'],
            'total_doorbell_accesses': stats['total_doorbell_accesses']
        }

# Example usage and integration
def create_nvme_doorbell_system(env, num_sqs=1024, access_pattern=DoorbellAccessPattern.THREAD_0_LEADER):
    """Create a complete NVMe doorbell system with optimization"""
    
    # Create doorbell controller
    doorbell_controller = NVMeDoorbellController(
        env=env,
        num_submission_queues=num_sqs,
        sq_depth=64,
        access_pattern=access_pattern
    )
    
    # Create optimizer
    optimizer = NVMeDoorbellOptimizer(env, doorbell_controller)
    
    return doorbell_controller, optimizer

if __name__ == "__main__":
    # Example simulation
    import simpy
    
    env = simpy.Environment()
    controller, optimizer = create_nvme_doorbell_system(env, num_sqs=1024)
    
    # Simulate some warp storage access
    def simulate_warp_storage_access(env, warp_id, optimizer):
        commands = []
        for i in range(32):  # 32 threads in warp
            cmd = Packet(
                id=f"warp_{warp_id}_thread_{i}",
                type="nvme_write",
                address=i * 4096,
                size=4096,
                data=b"x" * 4096
            )
            commands.append(cmd)
        
        results = optimizer.optimize_storage_access(warp_id, commands)
        
        print(f"Warp {warp_id} storage access results:")
        thread_0_result = results[0] if results else None
        if thread_0_result:
            print(f"  Thread 0: SQ {thread_0_result['sq_id']}, "
                  f"Latency: {thread_0_result['total_latency']}, "
                  f"Success: {thread_0_result['success']}")
        
        return results
    
    # Run simulation
    env.process(simulate_warp_storage_access(env, 0, optimizer))
    env.run(until=1000)
    
    # Print optimization results
    efficiency = optimizer.get_optimization_efficiency()
    print(f"\nOptimization Results:")
    print(f"  Bandwidth Savings: {efficiency['bandwidth_savings_percent']:.1f}%")
    print(f"  Thread 0 Efficiency: {efficiency['thread_0_efficiency']:.1f}%")
    print(f"  Broadcast Operations: {efficiency['broadcast_operations']}")