#!/usr/bin/env python3
import simpy
import random
from enum import Enum
from src.base.packet import Packet
from src.components.storage.nvme_doorbell_system import NVMeDoorbellController
from src.components.storage.nvme_fallback_patterns import NVMeFallbackController

class GraphFormat(Enum):
    """Graph storage formats following cuGraph patterns"""
    COO = "coordinate_list"      # Edge-centric: perfect load balance
    CSR = "compressed_sparse_row"  # Node-centric: memory efficient
    HYBRID = "hybrid_adaptive"   # Adaptive based on node degree

class AccessPattern(Enum):
    """Storage access patterns for GNN operations"""
    THREAD_0_LEADER = "thread_0_leader"          # Thread 0 handles all storage
    MULTI_THREAD = "multi_thread_parallel"      # Each thread independent access
    EDGE_CENTRIC = "edge_centric_cugraph"       # cuGraph style edge parallel
    HYBRID_ADAPTIVE = "hybrid_degree_based"     # Adaptive based on graph structure

class GraphStorageConfig:
    """Configuration for graph storage and access patterns"""
    def __init__(self, format=GraphFormat.HYBRID, access_pattern=AccessPattern.HYBRID_ADAPTIVE,
                 degree_threshold=32, enable_sq_lock_optimization=True):
        self.format = format
        self.access_pattern = access_pattern
        self.degree_threshold = degree_threshold  # Switch threshold for hybrid mode
        self.enable_sq_lock_optimization = enable_sq_lock_optimization
        
        # Performance tuning parameters
        self.coalescing_factor = 4   # Threads per coalesced memory access
        self.warp_specialization = True  # Different warps for different node types
        self.memory_prefetch_enabled = True

class CuGraphInspiredGNN:
    """
    GNN implementation inspired by cuGraph's thread assignment strategies
    with configurable access patterns and SQ doorbell optimization
    """
    def __init__(self, env, config=None):
        self.env = env
        self.config = config or GraphStorageConfig()
        
        # Graph data structures (simulated)
        self.coo_edges = []      # [(src, dst, edge_data), ...]
        self.csr_offsets = []    # Node pointer array for CSR
        self.csr_indices = []    # Neighbor indices for CSR
        self.node_degrees = {}   # Node ID -> degree count
        
        # Performance tracking
        self.access_stats = {
            'thread_0_accesses': 0,
            'multi_thread_accesses': 0,
            'edge_centric_accesses': 0,
            'hybrid_switches': 0,
            'sq_lock_contentions': 0,
            'memory_coalescing_hits': 0
        }
        
        # NVMe doorbell management
        self.nvme_controller = None
        self.sq_lock_manager = SQLockManager()
        
    def initialize_graph(self, num_nodes=1000, num_edges=5000, degree_distribution='power_law'):
        """Initialize graph with specified characteristics"""
        print(f"Initializing graph: {num_nodes} nodes, {num_edges} edges")
        
        # Generate graph based on distribution
        if degree_distribution == 'power_law':
            self._generate_power_law_graph(num_nodes, num_edges)
        elif degree_distribution == 'uniform':
            self._generate_uniform_graph(num_nodes, num_edges)
        else:
            self._generate_random_graph(num_nodes, num_edges)
        
        # Build CSR representation
        self._build_csr_representation()
        
        # Analyze graph characteristics for optimization
        self._analyze_graph_characteristics()
        
    def _generate_power_law_graph(self, num_nodes, num_edges):
        """Generate power-law graph (few hubs, many low-degree nodes)"""
        # Simulate power-law degree distribution
        for edge_id in range(num_edges):
            # 80% of edges connect to 20% of nodes (hubs)
            if random.random() < 0.8:
                src = random.randint(0, int(num_nodes * 0.2))  # Hub nodes
            else:
                src = random.randint(0, num_nodes - 1)  # Any node
            
            dst = random.randint(0, num_nodes - 1)
            if src != dst:  # Avoid self-loops
                edge_data = {'weight': random.uniform(0.1, 1.0)}
                self.coo_edges.append((src, dst, edge_data))
                
                # Track degrees
                self.node_degrees[src] = self.node_degrees.get(src, 0) + 1
                self.node_degrees[dst] = self.node_degrees.get(dst, 0) + 1
    
    def _generate_uniform_graph(self, num_nodes, num_edges):
        """Generate uniform degree distribution graph"""
        edges_per_node = num_edges // num_nodes
        for node in range(num_nodes):
            for _ in range(edges_per_node):
                neighbor = random.randint(0, num_nodes - 1)
                if neighbor != node:
                    edge_data = {'weight': random.uniform(0.1, 1.0)}
                    self.coo_edges.append((node, neighbor, edge_data))
                    self.node_degrees[node] = self.node_degrees.get(node, 0) + 1
    
    def _generate_random_graph(self, num_nodes, num_edges):
        """Generate random graph"""
        for _ in range(num_edges):
            src = random.randint(0, num_nodes - 1)
            dst = random.randint(0, num_nodes - 1)
            if src != dst:
                edge_data = {'weight': random.uniform(0.1, 1.0)}
                self.coo_edges.append((src, dst, edge_data))
                self.node_degrees[src] = self.node_degrees.get(src, 0) + 1
    
    def _build_csr_representation(self):
        """Build CSR representation from COO edges"""
        if not self.coo_edges:
            return
            
        max_node = max(max(src, dst) for src, dst, _ in self.coo_edges)
        num_nodes = max_node + 1
        
        # Sort edges by source node
        sorted_edges = sorted(self.coo_edges, key=lambda x: x[0])
        
        # Build CSR offsets and indices
        self.csr_offsets = [0] * (num_nodes + 1)
        self.csr_indices = []
        
        current_node = 0
        for src, dst, _ in sorted_edges:
            # Fill gaps for nodes with no outgoing edges
            while current_node < src:
                current_node += 1
                self.csr_offsets[current_node] = len(self.csr_indices)
            
            if current_node == src:
                self.csr_indices.append(dst)
            
            if current_node < src:
                current_node = src
                self.csr_offsets[current_node] = len(self.csr_indices) - 1
        
        # Fill remaining offsets
        while current_node < num_nodes:
            current_node += 1
            self.csr_offsets[current_node] = len(self.csr_indices)
    
    def _analyze_graph_characteristics(self):
        """Analyze graph to optimize access patterns"""
        if not self.node_degrees:
            return
            
        degrees = list(self.node_degrees.values())
        avg_degree = sum(degrees) / len(degrees)
        max_degree = max(degrees)
        
        print(f"Graph analysis:")
        print(f"  Average degree: {avg_degree:.2f}")
        print(f"  Maximum degree: {max_degree}")
        print(f"  Hub threshold: {self.config.degree_threshold}")
        
        # Count hub nodes
        hub_nodes = sum(1 for d in degrees if d > self.config.degree_threshold)
        print(f"  Hub nodes (>{self.config.degree_threshold}): {hub_nodes}/{len(degrees)}")
    
    def execute_message_passing(self, warp_id, num_threads=32, message_data=None):
        """
        Execute message passing with configurable access patterns
        """
        if self.config.access_pattern == AccessPattern.THREAD_0_LEADER:
            return self._execute_thread_0_pattern(warp_id, num_threads, message_data)
        elif self.config.access_pattern == AccessPattern.MULTI_THREAD:
            return self._execute_multi_thread_pattern(warp_id, num_threads, message_data)
        elif self.config.access_pattern == AccessPattern.EDGE_CENTRIC:
            return self._execute_edge_centric_pattern(warp_id, num_threads, message_data)
        elif self.config.access_pattern == AccessPattern.HYBRID_ADAPTIVE:
            return self._execute_hybrid_adaptive_pattern(warp_id, num_threads, message_data)
        else:
            return self._execute_thread_0_pattern(warp_id, num_threads, message_data)
    
    def _execute_thread_0_pattern(self, warp_id, num_threads, message_data):
        """Original Thread 0 leadership pattern"""
        results = []
        
        # Thread 0 performs all storage access
        if self.nvme_controller:
            batch_request = self._create_batch_storage_request(warp_id, num_threads, message_data)
            
            # Thread 0 leadership access
            result = self.nvme_controller.submit_command_with_thread0_optimization(
                batch_request, 0, warp_id, is_thread_leader=True
            )
            
            if result['success']:
                # Broadcast to all threads via shuffle
                broadcast_latency = 2  # Shuffle operation latency
                for thread_id in range(num_threads):
                    thread_result = {
                        'thread_id': thread_id,
                        'access_pattern': 'thread_0_leader',
                        'storage_latency': result['doorbell_latency'] if thread_id == 0 else 0,
                        'broadcast_latency': broadcast_latency if thread_id > 0 else 0,
                        'success': True
                    }
                    results.append(thread_result)
                
                self.access_stats['thread_0_accesses'] += 1
        
        return results
    
    def _execute_multi_thread_pattern(self, warp_id, num_threads, message_data):
        """Multi-thread parallel access pattern (cuGraph style for small graphs)"""
        results = []
        
        # Each thread performs independent storage access
        for thread_id in range(num_threads):
            if self.config.enable_sq_lock_optimization:
                # Check SQ lock contention
                sq_access_result = self.sq_lock_manager.request_sq_access(thread_id, warp_id)
                if not sq_access_result['granted']:
                    # SQ lock contention - fallback or retry
                    self.access_stats['sq_lock_contentions'] += 1
            
            # Independent storage request per thread
            storage_request = self._create_individual_storage_request(thread_id, warp_id, message_data)
            
            if self.nvme_controller:
                result = self.nvme_controller.submit_command_with_thread0_optimization(
                    storage_request, thread_id, warp_id, is_thread_leader=False
                )
                
                thread_result = {
                    'thread_id': thread_id,
                    'access_pattern': 'multi_thread',
                    'storage_latency': result.get('doorbell_latency', 5),
                    'broadcast_latency': 0,
                    'success': result.get('success', True),
                    'sq_contention': sq_access_result.get('contention_cycles', 0) if self.config.enable_sq_lock_optimization else 0
                }
                results.append(thread_result)
        
        self.access_stats['multi_thread_accesses'] += 1
        return results
    
    def _execute_edge_centric_pattern(self, warp_id, num_threads, message_data):
        """cuGraph-inspired edge-centric pattern for perfect load balancing"""
        results = []
        
        # Assign edges to threads instead of nodes
        warp_edges = self._get_edges_for_warp(warp_id, num_threads)
        
        for thread_id in range(num_threads):
            if thread_id < len(warp_edges):
                edge = warp_edges[thread_id]
                src, dst, edge_data = edge
                
                # Each thread processes one edge
                storage_request = self._create_edge_storage_request(src, dst, edge_data)
                
                if self.nvme_controller:
                    result = self.nvme_controller.submit_command_with_thread0_optimization(
                        storage_request, thread_id, warp_id, is_thread_leader=False
                    )
                    
                    # Memory coalescing optimization
                    coalescing_benefit = self._calculate_memory_coalescing(thread_id, warp_edges)
                    if coalescing_benefit > 0:
                        self.access_stats['memory_coalescing_hits'] += 1
                    
                    thread_result = {
                        'thread_id': thread_id,
                        'access_pattern': 'edge_centric',
                        'edge_processed': (src, dst),
                        'storage_latency': result.get('doorbell_latency', 3),
                        'coalescing_benefit': coalescing_benefit,
                        'success': result.get('success', True)
                    }
                    results.append(thread_result)
            else:
                # Idle thread (fewer edges than threads)
                thread_result = {
                    'thread_id': thread_id,
                    'access_pattern': 'edge_centric',
                    'edge_processed': None,
                    'storage_latency': 0,
                    'success': True,
                    'idle': True
                }
                results.append(thread_result)
        
        self.access_stats['edge_centric_accesses'] += 1
        return results
    
    def _execute_hybrid_adaptive_pattern(self, warp_id, num_threads, message_data):
        """Hybrid pattern: choose strategy based on node degrees"""
        # Analyze nodes assigned to this warp
        warp_nodes = self._get_nodes_for_warp(warp_id, num_threads)
        node_degrees = [self.node_degrees.get(node, 0) for node in warp_nodes]
        avg_degree = sum(node_degrees) / len(node_degrees) if node_degrees else 0
        max_degree = max(node_degrees) if node_degrees else 0
        
        # Decision logic for hybrid approach
        if max_degree > self.config.degree_threshold:
            # Hub nodes present - use Thread 0 pattern for storage efficiency
            pattern = 'thread_0_leader'
            results = self._execute_thread_0_pattern(warp_id, num_threads, message_data)
        elif avg_degree < self.config.degree_threshold / 4:
            # Low-degree nodes - use edge-centric for load balance
            pattern = 'edge_centric'
            results = self._execute_edge_centric_pattern(warp_id, num_threads, message_data)
        else:
            # Medium-degree nodes - use multi-thread for parallelism
            pattern = 'multi_thread'
            results = self._execute_multi_thread_pattern(warp_id, num_threads, message_data)
        
        # Update results with hybrid information
        for result in results:
            result['hybrid_decision'] = pattern
            result['avg_degree'] = avg_degree
            result['max_degree'] = max_degree
        
        self.access_stats['hybrid_switches'] += 1
        return results
    
    def _get_edges_for_warp(self, warp_id, num_threads):
        """Get edges assigned to this warp for edge-centric processing"""
        start_edge = warp_id * num_threads
        end_edge = min(start_edge + num_threads, len(self.coo_edges))
        return self.coo_edges[start_edge:end_edge]
    
    def _get_nodes_for_warp(self, warp_id, num_threads):
        """Get nodes assigned to this warp"""
        start_node = warp_id * num_threads
        end_node = start_node + num_threads
        return list(range(start_node, end_node))
    
    def _calculate_memory_coalescing(self, thread_id, warp_edges):
        """Calculate memory coalescing benefit for edge-centric access"""
        if thread_id == 0 or thread_id >= len(warp_edges):
            return 0
        
        # Check if consecutive threads access consecutive memory locations
        current_edge = warp_edges[thread_id]
        prev_edge = warp_edges[thread_id - 1] if thread_id > 0 else None
        
        if prev_edge and abs(current_edge[0] - prev_edge[0]) <= 4:
            return 1  # Coalesced access
        return 0
    
    def _create_batch_storage_request(self, warp_id, num_threads, message_data):
        """Create batch storage request for Thread 0 pattern"""
        return Packet(
            id=f"batch_warp_{warp_id}",
            type="gnn_batch_access",
            address=warp_id * num_threads,
            size=num_threads * 8,
            data=message_data or b'x' * (num_threads * 8)
        )
    
    def _create_individual_storage_request(self, thread_id, warp_id, message_data):
        """Create individual storage request for multi-thread pattern"""
        return Packet(
            id=f"individual_warp_{warp_id}_thread_{thread_id}",
            type="gnn_individual_access",
            address=warp_id * 32 + thread_id,
            size=8,
            data=message_data or b'x' * 8
        )
    
    def _create_edge_storage_request(self, src, dst, edge_data):
        """Create edge-specific storage request for edge-centric pattern"""
        return Packet(
            id=f"edge_{src}_{dst}",
            type="gnn_edge_access",
            address=src * 1000 + dst,  # Simple addressing scheme
            size=16,  # src + dst feature data
            data=str(edge_data).encode()
        )
    
    def get_performance_stats(self):
        """Get comprehensive performance statistics"""
        total_accesses = sum([
            self.access_stats['thread_0_accesses'],
            self.access_stats['multi_thread_accesses'],
            self.access_stats['edge_centric_accesses']
        ])
        
        return {
            'graph_config': {
                'format': self.config.format.value,
                'access_pattern': self.config.access_pattern.value,
                'degree_threshold': self.config.degree_threshold
            },
            'graph_stats': {
                'num_nodes': len(self.node_degrees),
                'num_edges': len(self.coo_edges),
                'avg_degree': sum(self.node_degrees.values()) / len(self.node_degrees) if self.node_degrees else 0
            },
            'access_stats': self.access_stats,
            'performance_metrics': {
                'total_accesses': total_accesses,
                'coalescing_efficiency': self.access_stats['memory_coalescing_hits'] / max(1, total_accesses),
                'contention_rate': self.access_stats['sq_lock_contentions'] / max(1, total_accesses)
            }
        }

class SQLockManager:
    """Manage SQ doorbell lock contention for multi-thread access"""
    def __init__(self, num_sqs=1024):
        self.num_sqs = num_sqs
        self.sq_locks = {}  # sq_id -> lock_owner_thread
        self.contention_stats = {}
        
    def request_sq_access(self, thread_id, warp_id):
        """Request SQ access with contention detection"""
        sq_id = (warp_id * 32 + thread_id) % self.num_sqs
        
        if sq_id in self.sq_locks:
            # Contention detected
            contention_cycles = random.randint(1, 5)  # Contention delay
            self.contention_stats[sq_id] = self.contention_stats.get(sq_id, 0) + 1
            return {
                'granted': False,
                'contention_cycles': contention_cycles,
                'sq_id': sq_id
            }
        else:
            # No contention
            self.sq_locks[sq_id] = thread_id
            return {
                'granted': True,
                'contention_cycles': 0,
                'sq_id': sq_id
            }
    
    def release_sq_access(self, sq_id):
        """Release SQ lock"""
        if sq_id in self.sq_locks:
            del self.sq_locks[sq_id]
    
    def get_contention_stats(self):
        """Get SQ contention statistics"""
        return {
            'total_contentions': sum(self.contention_stats.values()),
            'contended_sqs': len(self.contention_stats),
            'avg_contentions_per_sq': sum(self.contention_stats.values()) / max(1, len(self.contention_stats))
        }