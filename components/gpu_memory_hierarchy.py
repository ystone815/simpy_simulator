import simpy
import random
from enum import Enum
from collections import defaultdict, OrderedDict
from base.packet import Packet

class MemoryAccessType(Enum):
    READ = "read"
    WRITE = "write"
    ATOMIC = "atomic"

class CachePolicy(Enum):
    LRU = "lru"
    LFU = "lfu"
    RANDOM = "random"

class MemoryLevel(Enum):
    L1 = "l1"
    L2 = "l2" 
    L3 = "l3"
    SHARED_MEM = "shared"
    GLOBAL_MEM = "global"
    CONSTANT_MEM = "constant"
    TEXTURE_MEM = "texture"

class CacheLine:
    """Represents a cache line with metadata"""
    def __init__(self, address, data=None, size=128):
        self.address = address
        self.data = data
        self.size = size
        self.valid = True
        self.dirty = False
        self.last_access_time = 0
        self.access_count = 0
        self.tag = address // size

class GPUCache:
    """
    Generic GPU cache implementation with configurable policies.
    Supports different cache levels (L1, L2, L3) and access patterns.
    """
    def __init__(self, env, cache_id, level, size_kb, line_size=128, 
                 associativity=4, policy=CachePolicy.LRU, latency_cycles=1):
        self.env = env
        self.cache_id = cache_id
        self.level = level
        self.size_bytes = size_kb * 1024
        self.line_size = line_size
        self.associativity = associativity
        self.policy = policy
        self.latency_cycles = latency_cycles
        
        # Calculate cache geometry
        self.num_lines = self.size_bytes // line_size
        self.num_sets = self.num_lines // associativity
        
        # Cache storage - dictionary of sets, each set has associativity entries
        self.cache_sets = defaultdict(lambda: OrderedDict())
        
        # Performance counters
        self.hits = 0
        self.misses = 0
        self.total_accesses = 0
        self.evictions = 0
        
        # Port connections
        self.upper_level_port = simpy.Store(env)  # From CPU/SM
        self.lower_level_port = simpy.Store(env)  # To lower level cache/memory
        
        self.action = env.process(self.run())
    
    def run(self):
        """Main cache operation loop"""
        while True:
            # Handle incoming memory requests
            request_packet = yield self.upper_level_port.get()
            
            # Process the memory request
            yield self.env.process(self.handle_memory_request(request_packet))
    
    def handle_memory_request(self, packet):
        """Handle memory access request"""
        address = packet.address
        access_type = MemoryAccessType(packet.type)
        size = packet.size
        
        self.total_accesses += 1
        
        # Calculate cache line address
        line_addr = (address // self.line_size) * self.line_size
        set_index = (line_addr // self.line_size) % self.num_sets
        tag = line_addr // (self.line_size * self.num_sets)
        
        # Access latency
        yield self.env.timeout(self.latency_cycles)
        
        # Check for cache hit
        cache_set = self.cache_sets[set_index]
        hit = False
        
        if tag in cache_set:
            # Cache hit
            self.hits += 1
            hit = True
            cache_line = cache_set[tag]
            cache_line.last_access_time = self.env.now
            cache_line.access_count += 1
            
            # Move to end for LRU
            if self.policy == CachePolicy.LRU:
                cache_set.move_to_end(tag)
            
            # Handle write operations
            if access_type == MemoryAccessType.WRITE:
                cache_line.dirty = True
            
            # Create response packet
            response_packet = Packet(
                id=packet.id,
                type='cache_hit',
                source_id=f"{self.level.value}_cache_{self.cache_id}",
                destination_id=packet.source_id,
                address=address,
                size=size,
                data=cache_line.data if access_type == MemoryAccessType.READ else None,
                cache_hit=True,
                latency=self.latency_cycles
            )
        else:
            # Cache miss
            self.misses += 1
            
            # Need to fetch from lower level
            if hasattr(self, 'lower_level_cache') and self.lower_level_cache:
                # Forward request to lower level cache
                lower_request = Packet(
                    id=packet.id,
                    type=packet.type,
                    source_id=f"{self.level.value}_cache_{self.cache_id}",
                    destination_id=f"lower_level",
                    address=line_addr,  # Request full cache line
                    size=self.line_size,
                    data=packet.data
                )
                
                yield self.lower_level_port.put(lower_request)
                lower_response = yield self.lower_level_port.get()
                
                # Install cache line
                self._install_cache_line(set_index, tag, line_addr, lower_response.data)
                
                response_packet = Packet(
                    id=packet.id,
                    type='cache_miss',
                    source_id=f"{self.level.value}_cache_{self.cache_id}",
                    destination_id=packet.source_id,
                    address=address,
                    size=size,
                    data=lower_response.data if access_type == MemoryAccessType.READ else None,
                    cache_hit=False,
                    latency=self.latency_cycles + lower_response.latency
                )
            else:
                # No lower level - simulate memory access
                memory_latency = random.randint(100, 300)  # DRAM access latency
                yield self.env.timeout(memory_latency)
                
                response_packet = Packet(
                    id=packet.id,
                    type='memory_access',
                    source_id=f"{self.level.value}_cache_{self.cache_id}",
                    destination_id=packet.source_id,
                    address=address,
                    size=size,
                    data=b'\x00' * size if access_type == MemoryAccessType.READ else None,
                    cache_hit=False,
                    latency=self.latency_cycles + memory_latency
                )
        
        # Send response back
        yield self.upper_level_port.put(response_packet)
    
    def _install_cache_line(self, set_index, tag, address, data):
        """Install new cache line, possibly evicting old one"""
        cache_set = self.cache_sets[set_index]
        
        if len(cache_set) >= self.associativity:
            # Need to evict
            if self.policy == CachePolicy.LRU:
                evict_tag, evict_line = cache_set.popitem(last=False)
            elif self.policy == CachePolicy.LFU:
                evict_tag = min(cache_set.keys(), key=lambda k: cache_set[k].access_count)
                evict_line = cache_set.pop(evict_tag)
            else:  # RANDOM
                evict_tag = random.choice(list(cache_set.keys()))
                evict_line = cache_set.pop(evict_tag)
            
            self.evictions += 1
            
            # Write back if dirty
            if evict_line.dirty and hasattr(self, 'lower_level_cache'):
                # Simulate writeback
                pass
        
        # Install new line
        new_line = CacheLine(address, data, self.line_size)
        new_line.last_access_time = self.env.now
        cache_set[tag] = new_line
    
    def get_hit_rate(self):
        """Calculate cache hit rate"""
        if self.total_accesses == 0:
            return 0
        return self.hits / self.total_accesses
    
    def get_miss_rate(self):
        """Calculate cache miss rate"""
        return 1.0 - self.get_hit_rate()
    
    def get_stats(self):
        """Get cache performance statistics"""
        return {
            'cache_id': self.cache_id,
            'level': self.level.value,
            'size_kb': self.size_bytes // 1024,
            'total_accesses': self.total_accesses,
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': self.get_hit_rate(),
            'miss_rate': self.get_miss_rate(),
            'evictions': self.evictions
        }


class SharedMemory:
    """
    GPU Shared Memory model - fast, explicitly managed memory within SM.
    Includes bank conflict detection and modeling.
    """
    def __init__(self, env, sm_id, size_kb=228, num_banks=32, bank_width=4):
        self.env = env
        self.sm_id = sm_id
        self.size_bytes = size_kb * 1024
        self.num_banks = num_banks
        self.bank_width = bank_width  # bytes
        self.bank_size = self.size_bytes // num_banks
        
        # Bank conflict tracking
        self.bank_conflicts = 0
        self.bank_accesses = [0] * num_banks
        self.total_accesses = 0
        
        # Memory banks (simulated as availability counters)
        self.bank_busy_until = [0] * num_banks
        
        # Port for memory requests
        self.request_port = simpy.Store(env)
        self.response_port = simpy.Store(env)
        
        self.action = env.process(self.run())
    
    def run(self):
        """Main shared memory operation loop"""
        while True:
            request_packet = yield self.request_port.get()
            yield self.env.process(self.handle_shared_memory_request(request_packet))
    
    def handle_shared_memory_request(self, packet):
        """Handle shared memory access from warp"""
        addresses = packet.address if isinstance(packet.address, list) else [packet.address]
        access_type = packet.type
        warp_id = packet.source_id
        
        self.total_accesses += 1
        
        # Determine which banks are accessed
        banks_accessed = set()
        for addr in addresses:
            bank_id = (addr // self.bank_width) % self.num_banks
            banks_accessed.add(bank_id)
            self.bank_accesses[bank_id] += 1
        
        # Calculate bank conflicts
        num_accesses = len(addresses)
        num_banks_used = len(banks_accessed)
        
        if num_accesses > num_banks_used:
            # Bank conflict detected
            conflicts = num_accesses - num_banks_used
            self.bank_conflicts += conflicts
            
            # Additional latency due to serialization
            conflict_cycles = conflicts
            yield self.env.timeout(conflict_cycles)
        
        # Base shared memory access latency (very fast)
        base_latency = 1
        yield self.env.timeout(base_latency)
        
        # Create response
        response_packet = Packet(
            id=packet.id,
            type='shmem_response',
            source_id=f"SharedMem_SM_{self.sm_id}",
            destination_id=packet.source_id,
            address=packet.address,
            size=packet.size,
            data=b'\x00' * packet.size if access_type == 'read' else None,
            bank_conflicts=len(banks_accessed) < num_accesses
        )
        
        yield self.response_port.put(response_packet)
    
    def get_bank_conflict_rate(self):
        """Get bank conflict rate"""
        if self.total_accesses == 0:
            return 0
        return self.bank_conflicts / self.total_accesses
    
    def get_stats(self):
        """Get shared memory statistics"""
        return {
            'sm_id': self.sm_id,
            'size_kb': self.size_bytes // 1024,
            'total_accesses': self.total_accesses,
            'bank_conflicts': self.bank_conflicts,
            'conflict_rate': self.get_bank_conflict_rate(),
            'bank_utilization': [count/max(sum(self.bank_accesses), 1) 
                               for count in self.bank_accesses]
        }


class RegisterFile:
    """
    GPU Register File model - manages register allocation and access.
    """
    def __init__(self, env, sm_id, num_registers=65536, register_width=32):
        self.env = env
        self.sm_id = sm_id
        self.num_registers = num_registers  # 64K registers for H100/B200
        self.register_width = register_width  # bits
        
        # Register allocation tracking
        self.allocated_registers = {}  # warp_id -> [start_reg, count]
        self.free_registers = num_registers
        
        # Performance tracking
        self.register_accesses = 0
        self.allocation_failures = 0
        
        # Register bank conflicts (registers are banked too)
        self.num_banks = 8  # Typical number of register banks
        self.bank_conflicts = 0
    
    def allocate_registers(self, warp_id, num_registers_needed):
        """Allocate registers for a warp"""
        if num_registers_needed > self.free_registers:
            self.allocation_failures += 1
            return False
        
        # Simple allocation - find contiguous block
        start_register = self.num_registers - self.free_registers
        self.allocated_registers[warp_id] = (start_register, num_registers_needed)
        self.free_registers -= num_registers_needed
        
        return True
    
    def deallocate_registers(self, warp_id):
        """Deallocate registers for a warp"""
        if warp_id in self.allocated_registers:
            start_reg, count = self.allocated_registers[warp_id]
            del self.allocated_registers[warp_id]
            self.free_registers += count
            return True
        return False
    
    def access_register(self, warp_id, register_id):
        """Access a register for read/write"""
        self.register_accesses += 1
        
        # Check for bank conflicts (simplified)
        bank_id = register_id % self.num_banks
        # In real implementation, would check for simultaneous access to same bank
        
        return True  # Register access is very fast (1 cycle)
    
    def get_register_pressure(self):
        """Get register pressure (utilization)"""
        return (self.num_registers - self.free_registers) / self.num_registers
    
    def get_stats(self):
        """Get register file statistics"""
        return {
            'sm_id': self.sm_id,
            'total_registers': self.num_registers,
            'allocated_registers': self.num_registers - self.free_registers,
            'register_pressure': self.get_register_pressure(),
            'register_accesses': self.register_accesses,
            'allocation_failures': self.allocation_failures,
            'bank_conflicts': self.bank_conflicts
        }


class MemoryHierarchy:
    """
    Complete GPU memory hierarchy manager.
    Connects all memory components and handles data movement.
    """
    def __init__(self, env, gpu_type="H100"):
        self.env = env
        self.gpu_type = gpu_type
        
        # Memory components
        self.l1_caches = {}  # sm_id -> L1Cache
        self.l2_cache = None
        self.shared_memories = {}  # sm_id -> SharedMemory
        self.register_files = {}  # sm_id -> RegisterFile
        
        # Global memory simulation
        self.global_memory_latency = 300  # cycles
        self.global_memory_bandwidth = 1000  # GB/s (simplified)
        
    def create_sm_memory_subsystem(self, sm_id):
        """Create memory subsystem for one SM"""
        # L1 Cache (combined L1/Texture cache)
        if self.gpu_type in ["H100", "B200"]:
            l1_cache = GPUCache(
                env=self.env,
                cache_id=sm_id,
                level=MemoryLevel.L1,
                size_kb=256,  # 256KB total (L1 + Texture + Shared)
                line_size=128,
                associativity=4,
                latency_cycles=1
            )
        
        # Shared Memory
        shared_mem = SharedMemory(
            env=self.env,
            sm_id=sm_id,
            size_kb=228,  # 228KB for H100/B200
            num_banks=32
        )
        
        # Register File
        register_file = RegisterFile(
            env=self.env,
            sm_id=sm_id,
            num_registers=65536  # 64K registers
        )
        
        self.l1_caches[sm_id] = l1_cache
        self.shared_memories[sm_id] = shared_mem
        self.register_files[sm_id] = register_file
        
        return l1_cache, shared_mem, register_file
    
    def create_l2_cache(self, size_mb=50):
        """Create shared L2 cache"""
        self.l2_cache = GPUCache(
            env=self.env,
            cache_id=0,
            level=MemoryLevel.L2,
            size_kb=size_mb * 1024,
            line_size=128,
            associativity=16,
            latency_cycles=10
        )
        
        # Connect L1 caches to L2
        for l1_cache in self.l1_caches.values():
            l1_cache.lower_level_cache = self.l2_cache
    
    def get_memory_stats(self):
        """Get statistics for entire memory hierarchy"""
        stats = {
            'gpu_type': self.gpu_type,
            'l1_caches': [cache.get_stats() for cache in self.l1_caches.values()],
            'l2_cache': self.l2_cache.get_stats() if self.l2_cache else None,
            'shared_memories': [shmem.get_stats() for shmem in self.shared_memories.values()],
            'register_files': [rf.get_stats() for rf in self.register_files.values()]
        }
        return stats