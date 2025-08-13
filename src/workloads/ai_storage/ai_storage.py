import simpy
import random
import math
from enum import Enum
from collections import defaultdict, OrderedDict
from src.base.packet import Packet

class KVCacheAccessPattern(Enum):
    SEQUENTIAL = "sequential"
    RANDOM = "random"
    ATTENTION_PATTERN = "attention"
    SLIDING_WINDOW = "sliding_window"

class VectorDBIndexType(Enum):
    FLAT = "flat"
    IVF = "ivf"
    HNSW = "hnsw"
    FAISS = "faiss"

class GraphAccessPattern(Enum):
    BFS = "breadth_first"
    DFS = "depth_first"
    RANDOM_WALK = "random_walk"
    NEIGHBORHOOD = "neighborhood"
    UNIFORM = "uniform"

class KVCacheStorage:
    """
    Key-Value Cache Storage for LLM inference with compression and adaptive retention.
    Simulates modern KV cache optimizations like DynamicKV and compression techniques.
    """
    def __init__(self, env, cache_id, max_tokens=4096, compression_ratio=0.5, retention_policy="dynamic"):
        self.env = env
        self.cache_id = cache_id
        self.max_tokens = max_tokens
        self.compression_ratio = compression_ratio
        self.retention_policy = retention_policy
        
        # Cache storage (token_id -> {key, value, metadata})
        self.key_cache = {}
        self.value_cache = {}
        self.token_metadata = {}  # importance scores, last_access, etc.
        
        # Compression state
        self.compressed_tokens = set()
        self.compression_enabled = True
        
        # Performance tracking
        self.cache_hits = 0
        self.cache_misses = 0
        self.evictions = 0
        self.compressions = 0
        self.decompressions = 0
        self.total_accesses = 0
        
        # Attention-based importance tracking
        self.attention_scores = defaultdict(float)
        self.access_frequency = defaultdict(int)
        self.recency_scores = defaultdict(float)
        
        # Dynamic retention parameters
        self.importance_threshold = 0.1
        self.layer_retention_ratios = {}  # layer_id -> retention_ratio
        
        self.request_port = simpy.Store(env)
        self.response_port = simpy.Store(env)
        self.action = env.process(self.run())
    
    def run(self):
        """Main KV cache operation loop"""
        while True:
            request = yield self.request_port.get()
            yield self.env.process(self.handle_kv_request(request))
    
    def handle_kv_request(self, packet):
        """Handle KV cache access request with thread 0 leadership optimization"""
        request_type = packet.type  # 'store', 'retrieve', 'evict'
        layer_id = packet.get('layer_id', 0)
        token_positions = packet.get('token_positions', [])
        attention_scores = packet.get('attention_scores', [])
        warp_context = packet.get('warp_context', None)
        
        self.total_accesses += 1
        
        # Check if this is a thread 0 leader request
        is_thread_leader = packet.get('thread_leader', False)
        thread_broadcast = packet.get('broadcast_to_warp', True)
        
        if request_type == 'store':
            result = yield self.env.process(self._store_kv_tokens(
                layer_id, token_positions, packet.data, attention_scores, 
                warp_context, is_thread_leader
            ))
        
        elif request_type == 'retrieve':
            result = yield self.env.process(self._retrieve_kv_tokens(
                layer_id, token_positions, packet.get('access_pattern', KVCacheAccessPattern.SEQUENTIAL),
                warp_context, is_thread_leader, thread_broadcast
            ))
            
            # Send response
            response = Packet(
                id=packet.id,
                type='kv_response',
                source_id=f"KVCache_{self.cache_id}",
                destination_id=packet.source_id,
                data=result['data'],
                cache_hit=result['hit'],
                compression_used=result['compressed'],
                latency=result['latency']
            )
            yield self.response_port.put(response)
        
        elif request_type == 'adaptive_evict':
            yield self.env.process(self._adaptive_eviction(layer_id))
    
    def _store_kv_tokens(self, layer_id, token_positions, kv_data, attention_scores, warp_context=None, is_thread_leader=False):
        """Store KV pairs for tokens with importance-based compression and thread 0 optimization"""
        base_latency = 1  # Base storage latency
        compression_latency = 0
        thread_optimization_savings = 0
        
        # Thread 0 leadership optimization for storage
        if warp_context and is_thread_leader:
            # Only thread 0 performs actual storage operations
            # Reduce redundant memory transactions from other threads
            thread_optimization_savings = len(token_positions) * 0.1  # 10% per token savings
            base_latency = max(0.5, base_latency - thread_optimization_savings)
        
        for i, token_pos in enumerate(token_positions):
            token_key = (layer_id, token_pos)
            
            # Calculate token importance
            attention_score = attention_scores[i] if i < len(attention_scores) else 0.5
            importance = self._calculate_token_importance(token_pos, attention_score)
            
            # Update metadata
            self.token_metadata[token_key] = {
                'importance': importance,
                'last_access': self.env.now,
                'access_count': self.access_frequency[token_key],
                'layer_id': layer_id,
                'compressed': False
            }
            
            self.attention_scores[token_key] = attention_score
            self.access_frequency[token_key] += 1
            self.recency_scores[token_key] = self.env.now
            
            # Decide whether to compress based on importance
            if self.compression_enabled and importance < self.importance_threshold:
                # Compress token
                compressed_data = self._compress_kv_data(kv_data[i] if i < len(kv_data) else None)
                self.key_cache[token_key] = compressed_data['key']
                self.value_cache[token_key] = compressed_data['value']
                self.compressed_tokens.add(token_key)
                self.token_metadata[token_key]['compressed'] = True
                self.compressions += 1
                compression_latency += 2  # Compression overhead
            else:
                # Store uncompressed
                data_item = kv_data[i] if i < len(kv_data) else {'key': b'', 'value': b''}
                self.key_cache[token_key] = data_item.get('key', b'')
                self.value_cache[token_key] = data_item.get('value', b'')
            
            # Check if eviction is needed
            if len(self.key_cache) > self.max_tokens:
                yield self.env.process(self._evict_least_important())
        
        total_latency = base_latency + compression_latency
        yield self.env.timeout(total_latency)
    
    def _retrieve_kv_tokens(self, layer_id, token_positions, access_pattern, warp_context=None, is_thread_leader=False, thread_broadcast=True):
        """Retrieve KV pairs with pattern-aware optimization and thread 0 leadership"""
        base_latency = 1
        decompression_latency = 0
        cache_hits = 0
        retrieved_data = []
        broadcast_latency = 0
        
        # Thread 0 leadership optimization
        if warp_context and is_thread_leader:
            # Thread 0 performs storage access, then broadcasts to warp
            base_latency *= 0.7  # 30% reduction for consolidated access
            
            if thread_broadcast:
                # Add broadcast overhead (shuffle operations)
                warp_size = getattr(warp_context, 'num_threads', 32)
                broadcast_latency = max(1, int(warp_size / 4))  # 4 threads per shuffle cycle
        
        # Optimize access based on pattern
        if access_pattern == KVCacheAccessPattern.ATTENTION_PATTERN:
            # Sort by attention scores for better cache locality
            token_positions = sorted(token_positions, 
                                   key=lambda pos: self.attention_scores.get((layer_id, pos), 0), 
                                   reverse=True)
        
        for token_pos in token_positions:
            token_key = (layer_id, token_pos)
            
            if token_key in self.key_cache:
                self.cache_hits += 1
                cache_hits += 1
                
                # Update access metadata
                self.access_frequency[token_key] += 1
                self.recency_scores[token_key] = self.env.now
                self.token_metadata[token_key]['last_access'] = self.env.now
                
                # Handle decompression if needed
                if token_key in self.compressed_tokens:
                    decompressed_data = self._decompress_kv_data(
                        self.key_cache[token_key], 
                        self.value_cache[token_key]
                    )
                    retrieved_data.append(decompressed_data)
                    self.decompressions += 1
                    decompression_latency += 1
                else:
                    retrieved_data.append({
                        'key': self.key_cache[token_key],
                        'value': self.value_cache[token_key]
                    })
            else:
                self.cache_misses += 1
                # Would need to fetch from slower storage
                retrieved_data.append(None)
        
        total_latency = base_latency + decompression_latency + broadcast_latency
        yield self.env.timeout(total_latency)
        
        result = {
            'data': retrieved_data,
            'hit': cache_hits == len(token_positions),
            'hit_ratio': cache_hits / len(token_positions),
            'compressed': decompression_latency > 0,
            'latency': total_latency,
            'thread_leader_access': is_thread_leader,
            'broadcast_latency': broadcast_latency
        }
        
        return result
    
    def _calculate_token_importance(self, token_pos, attention_score):
        """Calculate token importance for retention decisions"""
        # Combine multiple factors for importance
        recency_factor = 1.0 / (1.0 + self.env.now - self.recency_scores.get((0, token_pos), 0))
        frequency_factor = math.log(1 + self.access_frequency.get((0, token_pos), 0))
        attention_factor = attention_score
        
        # Position-based importance (earlier tokens often more important)
        position_factor = 1.0 / (1.0 + token_pos / 100)
        
        importance = 0.4 * attention_factor + 0.3 * frequency_factor + 0.2 * recency_factor + 0.1 * position_factor
        return min(importance, 1.0)
    
    def _compress_kv_data(self, kv_item):
        """Simulate KV data compression (quantization, etc.)"""
        if not kv_item:
            return {'key': b'', 'value': b''}
        
        # Simulate compression (e.g., 4-bit quantization)
        key_data = kv_item.get('key', b'')
        value_data = kv_item.get('value', b'')
        
        compressed_size = int(len(key_data + value_data) * self.compression_ratio)
        
        return {
            'key': key_data[:compressed_size//2],  # Simplified compression
            'value': value_data[:compressed_size//2]
        }
    
    def _decompress_kv_data(self, compressed_key, compressed_value):
        """Simulate KV data decompression"""
        # Simulate decompression (restore original size)
        original_key_size = int(len(compressed_key) / self.compression_ratio)
        original_value_size = int(len(compressed_value) / self.compression_ratio)
        
        return {
            'key': compressed_key + b'\x00' * (original_key_size - len(compressed_key)),
            'value': compressed_value + b'\x00' * (original_value_size - len(compressed_value))
        }
    
    def _evict_least_important(self):
        """Evict least important tokens using DynamicKV-style policy"""
        if not self.token_metadata:
            return
        
        # Find least important token
        least_important_token = min(self.token_metadata.keys(), 
                                  key=lambda k: self.token_metadata[k]['importance'])
        
        # Remove from cache
        if least_important_token in self.key_cache:
            del self.key_cache[least_important_token]
        if least_important_token in self.value_cache:
            del self.value_cache[least_important_token]
        if least_important_token in self.compressed_tokens:
            self.compressed_tokens.remove(least_important_token)
        
        del self.token_metadata[least_important_token]
        self.evictions += 1
        
        yield self.env.timeout(1)  # Eviction latency
    
    def _adaptive_eviction(self, layer_id):
        """Adaptive eviction based on layer-specific retention ratios"""
        retention_ratio = self.layer_retention_ratios.get(layer_id, 0.8)
        layer_tokens = [(k, v) for k, v in self.token_metadata.items() if v['layer_id'] == layer_id]
        
        if not layer_tokens:
            return
        
        # Sort by importance
        layer_tokens.sort(key=lambda x: x[1]['importance'])
        
        # Evict lowest importance tokens to meet retention ratio
        num_to_keep = int(len(layer_tokens) * retention_ratio)
        tokens_to_evict = layer_tokens[:-num_to_keep] if num_to_keep > 0 else layer_tokens
        
        for token_key, _ in tokens_to_evict:
            if token_key in self.key_cache:
                del self.key_cache[token_key]
            if token_key in self.value_cache:
                del self.value_cache[token_key]
            if token_key in self.compressed_tokens:
                self.compressed_tokens.remove(token_key)
            del self.token_metadata[token_key]
            self.evictions += 1
        
        yield self.env.timeout(len(tokens_to_evict))  # Eviction latency
    
    def get_cache_stats(self):
        """Get KV cache performance statistics"""
        hit_rate = self.cache_hits / max(self.total_accesses, 1)
        compression_rate = len(self.compressed_tokens) / max(len(self.key_cache), 1)
        
        return {
            'cache_id': self.cache_id,
            'max_tokens': self.max_tokens,
            'stored_tokens': len(self.key_cache),
            'compressed_tokens': len(self.compressed_tokens),
            'hit_rate': hit_rate,
            'miss_rate': 1.0 - hit_rate,
            'compression_rate': compression_rate,
            'total_accesses': self.total_accesses,
            'evictions': self.evictions,
            'compressions': self.compressions,
            'decompressions': self.decompressions
        }


class VectorDatabase:
    """
    Vector Database storage for embedding similarity search (RAG workloads).
    Includes indexing, approximate nearest neighbor search, and tiered storage.
    """
    def __init__(self, env, db_id, vector_dim=1024, index_type=VectorDBIndexType.HNSW, 
                 max_vectors=1000000, index_build_threads=8):
        self.env = env
        self.db_id = db_id
        self.vector_dim = vector_dim
        self.index_type = index_type
        self.max_vectors = max_vectors
        self.index_build_threads = index_build_threads
        
        # Vector storage
        self.vectors = {}  # vector_id -> embedding
        self.metadata = {}  # vector_id -> metadata
        self.vector_counter = 0
        
        # Index structures
        self.index_built = False
        self.index_structure = None
        self.index_build_time = 0
        
        # Tiered storage (hot/cold data)
        self.hot_vectors = set()  # Frequently accessed vectors
        self.cold_vectors = set()  # Rarely accessed vectors
        self.access_frequency = defaultdict(int)
        
        # Performance tracking
        self.total_queries = 0
        self.index_hits = 0
        self.exact_searches = 0
        self.build_operations = 0
        
        # Search performance parameters by index type
        self.index_params = {
            VectorDBIndexType.FLAT: {"build_time_per_vector": 0.001, "search_time_base": 1.0},
            VectorDBIndexType.IVF: {"build_time_per_vector": 0.01, "search_time_base": 0.1},
            VectorDBIndexType.HNSW: {"build_time_per_vector": 0.05, "search_time_base": 0.05},
            VectorDBIndexType.FAISS: {"build_time_per_vector": 0.02, "search_time_base": 0.08}
        }
        
        self.request_port = simpy.Store(env)
        self.response_port = simpy.Store(env)
        self.action = env.process(self.run())
    
    def run(self):
        """Main vector database operation loop"""
        while True:
            request = yield self.request_port.get()
            yield self.env.process(self.handle_vector_request(request))
    
    def handle_vector_request(self, packet):
        """Handle vector database request with thread 0 leadership optimization"""
        request_type = packet.type  # 'insert', 'search', 'build_index', 'update_tier'
        warp_context = packet.get('warp_context', None)
        is_thread_leader = packet.get('thread_leader', False)
        
        if request_type == 'insert':
            result = yield self.env.process(self._insert_vectors(
                packet.data, packet.get('metadata_list', []), warp_context, is_thread_leader
            ))
        
        elif request_type == 'search':
            result = yield self.env.process(self._search_similar_vectors(
                packet.data,  # Query vector
                packet.get('k', 10),  # Number of neighbors
                packet.get('search_params', {}),
                warp_context, is_thread_leader
            ))
        
        elif request_type == 'build_index':
            result = yield self.env.process(self._build_index())
        
        elif request_type == 'update_tier':
            result = yield self.env.process(self._update_storage_tiers())
        
        else:
            result = {'error': f'Unknown request type: {request_type}'}
        
        # Send response
        response = Packet(
            id=packet.id,
            type='vector_db_response',
            source_id=f"VectorDB_{self.db_id}",
            destination_id=packet.source_id,
            data=result
        )
        yield self.response_port.put(response)
    
    def _insert_vectors(self, vectors, metadata_list, warp_context=None, is_thread_leader=False):
        """Insert new vectors into the database with thread 0 leadership"""
        insertion_latency = 0
        inserted_count = 0
        
        # Thread 0 leadership optimization for batch insertion
        if warp_context and is_thread_leader:
            # Thread 0 handles batch insertion, reducing memory contention
            insertion_latency_multiplier = 0.6  # 40% reduction
        else:
            insertion_latency_multiplier = 1.0
        
        for i, vector in enumerate(vectors):
            if len(self.vectors) >= self.max_vectors:
                break
            
            vector_id = self.vector_counter
            self.vector_counter += 1
            
            # Store vector and metadata
            self.vectors[vector_id] = vector
            self.metadata[vector_id] = metadata_list[i] if i < len(metadata_list) else {}
            
            # Initially mark as hot data
            self.hot_vectors.add(vector_id)
            
            inserted_count += 1
            insertion_latency += 0.1 * insertion_latency_multiplier  # Per-vector insertion cost
        
        # Invalidate index if significant insertion
        if inserted_count > len(self.vectors) * 0.1:
            self.index_built = False
        
        yield self.env.timeout(insertion_latency)
        
        return {
            'inserted_count': inserted_count,
            'total_vectors': len(self.vectors),
            'index_invalidated': not self.index_built,
            'latency': insertion_latency
        }
    
    def _search_similar_vectors(self, query_vector, k, search_params, warp_context=None, is_thread_leader=False):
        """Search for k most similar vectors with thread 0 leadership"""
        self.total_queries += 1
        search_latency = 0
        broadcast_latency = 0
        
        # Thread 0 leadership optimization
        if warp_context and is_thread_leader:
            # Thread 0 performs search, broadcasts results
            search_efficiency = 0.8  # 20% improvement
            warp_size = getattr(warp_context, 'num_threads', 32)
            broadcast_latency = max(1, int(warp_size / 8))  # Broadcast search results
        else:
            search_efficiency = 1.0
        
        if not self.vectors:
            return {'results': [], 'latency': 0, 'method': 'empty'}
        
        if self.index_built and self.index_type != VectorDBIndexType.FLAT:
            # Use index-based search
            search_latency = self.index_params[self.index_type]["search_time_base"]
            search_latency *= math.log(len(self.vectors)) / math.log(2)  # Log complexity
            search_latency *= search_efficiency  # Apply thread 0 optimization
            self.index_hits += 1
            search_method = f"index_{self.index_type.value}"
        else:
            # Fallback to exact search
            search_latency = len(self.vectors) * 0.001 * search_efficiency  # Linear scan
            self.exact_searches += 1
            search_method = "exact_search"
        
        # Simulate tiered access - hot data is faster
        hot_vectors_accessed = sum(1 for vid in self.vectors.keys() if vid in self.hot_vectors)
        cold_vectors_accessed = len(self.vectors) - hot_vectors_accessed
        
        # Cold storage penalty
        if cold_vectors_accessed > 0:
            search_latency += cold_vectors_accessed * 0.01
        
        total_search_latency = search_latency + broadcast_latency
        yield self.env.timeout(total_search_latency)
        
        # Simulate finding k nearest neighbors
        # In practice, this would involve actual distance calculations
        candidate_vectors = list(self.vectors.keys())
        random.shuffle(candidate_vectors)  # Simplified similarity ranking
        results = candidate_vectors[:min(k, len(candidate_vectors))]
        
        # Update access frequency for tiering
        for vector_id in results:
            self.access_frequency[vector_id] += 1
        
        return {
            'results': results,
            'distances': [random.uniform(0, 1) for _ in results],  # Simulated distances
            'latency': total_search_latency,
            'method': search_method,
            'hot_vectors_accessed': hot_vectors_accessed,
            'cold_vectors_accessed': cold_vectors_accessed,
            'thread_leader_access': is_thread_leader,
            'broadcast_latency': broadcast_latency
        }
    
    def _build_index(self):
        """Build vector index for faster searches"""
        if len(self.vectors) < 100:
            return {'status': 'insufficient_data', 'vectors_count': len(self.vectors)}
        
        self.build_operations += 1
        
        # Calculate build time based on index type and vector count
        build_time_per_vector = self.index_params[self.index_type]["build_time_per_vector"]
        total_build_time = len(self.vectors) * build_time_per_vector
        
        # Parallel build with multiple threads
        parallel_build_time = total_build_time / self.index_build_threads
        
        yield self.env.timeout(parallel_build_time)
        
        self.index_built = True
        self.index_build_time = parallel_build_time
        
        return {
            'status': 'success',
            'index_type': self.index_type.value,
            'build_time': parallel_build_time,
            'vectors_indexed': len(self.vectors),
            'parallel_threads': self.index_build_threads
        }
    
    def _update_storage_tiers(self):
        """Update hot/cold storage tiers based on access patterns"""
        if not self.vectors:
            return {'status': 'no_vectors'}
        
        # Calculate access frequency thresholds
        access_counts = list(self.access_frequency.values())
        if access_counts:
            hot_threshold = sorted(access_counts, reverse=True)[len(access_counts) // 4]  # Top 25%
        else:
            hot_threshold = 0
        
        # Update tiers
        new_hot_vectors = set()
        new_cold_vectors = set()
        
        for vector_id in self.vectors.keys():
            if self.access_frequency[vector_id] >= hot_threshold:
                new_hot_vectors.add(vector_id)
            else:
                new_cold_vectors.add(vector_id)
        
        # Track tier changes
        promoted_to_hot = new_hot_vectors - self.hot_vectors
        demoted_to_cold = self.hot_vectors - new_hot_vectors
        
        self.hot_vectors = new_hot_vectors
        self.cold_vectors = new_cold_vectors
        
        # Simulate tier update latency
        tier_update_latency = (len(promoted_to_hot) + len(demoted_to_cold)) * 0.01
        yield self.env.timeout(tier_update_latency)
        
        return {
            'status': 'success',
            'hot_vectors': len(self.hot_vectors),
            'cold_vectors': len(self.cold_vectors),
            'promoted': len(promoted_to_hot),
            'demoted': len(demoted_to_cold),
            'latency': tier_update_latency
        }
    
    def get_db_stats(self):
        """Get vector database performance statistics"""
        index_hit_rate = self.index_hits / max(self.total_queries, 1)
        hot_data_ratio = len(self.hot_vectors) / max(len(self.vectors), 1)
        
        return {
            'db_id': self.db_id,
            'vector_dim': self.vector_dim,
            'index_type': self.index_type.value,
            'total_vectors': len(self.vectors),
            'index_built': self.index_built,
            'index_build_time': self.index_build_time,
            'total_queries': self.total_queries,
            'index_hit_rate': index_hit_rate,
            'exact_search_rate': self.exact_searches / max(self.total_queries, 1),
            'hot_vectors': len(self.hot_vectors),
            'cold_vectors': len(self.cold_vectors),
            'hot_data_ratio': hot_data_ratio,
            'build_operations': self.build_operations
        }


class GNNStorage:
    """
    Graph Neural Network storage for graph structure and feature data.
    Handles graph sampling, neighborhood queries, and batch loading.
    """
    def __init__(self, env, storage_id, max_nodes=100000, max_edges=1000000, 
                 feature_dim=128, sampling_strategy="uniform"):
        self.env = env
        self.storage_id = storage_id
        self.max_nodes = max_nodes
        self.max_edges = max_edges
        self.feature_dim = feature_dim
        self.sampling_strategy = sampling_strategy
        
        # Graph structure
        self.adjacency_list = defaultdict(list)  # node_id -> [neighbor_ids]
        self.edge_features = {}  # (src, dst) -> feature_vector
        self.node_features = {}  # node_id -> feature_vector
        self.node_count = 0
        self.edge_count = 0
        
        # Sampling cache
        self.neighborhood_cache = {}  # node_id -> cached_neighbors
        self.subgraph_cache = {}  # subgraph_id -> cached_subgraph
        
        # Access pattern tracking
        self.node_access_frequency = defaultdict(int)
        self.neighborhood_queries = 0
        self.subgraph_samples = 0
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Batch processing
        self.batch_queue = []
        self.current_batch_size = 0
        self.max_batch_size = 1024
        
        self.request_port = simpy.Store(env)
        self.response_port = simpy.Store(env)
        self.action = env.process(self.run())
    
    def run(self):
        """Main GNN storage operation loop"""
        while True:
            request = yield self.request_port.get()
            yield self.env.process(self.handle_gnn_request(request))
    
    def handle_gnn_request(self, packet):
        """Handle GNN storage request with thread 0 leadership optimization"""
        request_type = packet.type
        warp_context = packet.get('warp_context', None)
        is_thread_leader = packet.get('thread_leader', False)
        
        if request_type == 'add_nodes':
            result = yield self.env.process(self._add_nodes(
                packet.data.get('node_ids', []),
                packet.data.get('features', []),
                warp_context, is_thread_leader
            ))
        
        elif request_type == 'add_edges':
            result = yield self.env.process(self._add_edges(
                packet.data.get('edges', []),  # [(src, dst), ...]
                packet.data.get('edge_features', []),
                warp_context, is_thread_leader
            ))
        
        elif request_type == 'sample_neighborhood':
            result = yield self.env.process(self._sample_neighborhood(
                packet.data.get('node_id'),
                packet.data.get('num_hops', 2),
                packet.data.get('num_neighbors', 10),
                packet.data.get('access_pattern', GraphAccessPattern.UNIFORM),
                warp_context, is_thread_leader
            ))
        
        elif request_type == 'batch_sample':
            result = yield self.env.process(self._batch_sample_subgraphs(
                packet.data.get('seed_nodes', []),
                packet.data.get('batch_size', 32)
            ))
        
        else:
            result = {'error': f'Unknown request type: {request_type}'}
        
        # Send response
        response = Packet(
            id=packet.id,
            type='gnn_storage_response',
            source_id=f"GNNStorage_{self.storage_id}",
            destination_id=packet.source_id,
            data=result
        )
        yield self.response_port.put(response)
    
    def _add_nodes(self, node_ids, features, warp_context=None, is_thread_leader=False):
        """Add nodes with features to the graph using thread 0 leadership"""
        nodes_added = 0
        addition_latency = 0
        
        # Thread 0 leadership for graph modifications
        if warp_context and is_thread_leader:
            # Thread 0 handles batch node additions
            latency_multiplier = 0.5  # 50% improvement for batch operations
        else:
            latency_multiplier = 1.0
        
        for i, node_id in enumerate(node_ids):
            if self.node_count >= self.max_nodes:
                break
            
            if node_id not in self.node_features:
                # Add node feature
                feature = features[i] if i < len(features) else [0.0] * self.feature_dim
                self.node_features[node_id] = feature
                nodes_added += 1
                self.node_count += 1
                addition_latency += 0.01 * latency_multiplier  # Per-node addition cost
        
        yield self.env.timeout(addition_latency)
        
        return {
            'nodes_added': nodes_added,
            'total_nodes': self.node_count,
            'latency': addition_latency
        }
    
    def _add_edges(self, edges, edge_features, warp_context=None, is_thread_leader=False):
        """Add edges to the graph structure using thread 0 leadership"""
        edges_added = 0
        addition_latency = 0
        
        # Thread 0 leadership for edge operations
        if warp_context and is_thread_leader:
            latency_multiplier = 0.6  # 40% improvement for batch edge operations
        else:
            latency_multiplier = 1.0
        
        for i, (src, dst) in enumerate(edges):
            if self.edge_count >= self.max_edges:
                break
            
            # Add to adjacency list
            if dst not in self.adjacency_list[src]:
                self.adjacency_list[src].append(dst)
                edges_added += 1
                self.edge_count += 1
                
                # Add edge features if provided
                if i < len(edge_features):
                    self.edge_features[(src, dst)] = edge_features[i]
                
                # Invalidate neighborhood cache for affected nodes
                if src in self.neighborhood_cache:
                    del self.neighborhood_cache[src]
                
                addition_latency += 0.005 * latency_multiplier  # Per-edge addition cost
        
        yield self.env.timeout(addition_latency)
        
        return {
            'edges_added': edges_added,
            'total_edges': self.edge_count,
            'latency': addition_latency
        }
    
    def _sample_neighborhood(self, center_node, num_hops, num_neighbors, access_pattern, warp_context=None, is_thread_leader=False):
        """Sample neighborhood around a center node with thread 0 leadership"""
        if center_node not in self.node_features:
            return {'error': 'Node not found', 'node_id': center_node}
        
        self.neighborhood_queries += 1
        self.node_access_frequency[center_node] += 1
        
        # Thread 0 leadership for neighborhood sampling
        broadcast_latency = 0
        if warp_context and is_thread_leader:
            # Thread 0 performs sampling, broadcasts to other threads
            sampling_efficiency = 0.7  # 30% improvement
            warp_size = getattr(warp_context, 'num_threads', 32)
            broadcast_latency = max(1, int(warp_size / 8))  # Broadcast sampled data
        else:
            sampling_efficiency = 1.0
        
        # Check cache first
        cache_key = (center_node, num_hops, num_neighbors)
        if cache_key in self.neighborhood_cache:
            self.cache_hits += 1
            cached_result = self.neighborhood_cache[cache_key]
            yield self.env.timeout(1)  # Cache access latency
            return cached_result
        
        self.cache_misses += 1
        
        # Sample neighborhood based on access pattern
        sampling_latency = 0
        visited_nodes = {center_node}
        current_frontier = [center_node]
        
        for hop in range(num_hops):
            next_frontier = []
            
            for node in current_frontier:
                neighbors = self.adjacency_list.get(node, [])
                
                # Apply sampling strategy
                if access_pattern == GraphAccessPattern.BFS:
                    # Breadth-first sampling
                    sampled_neighbors = neighbors[:num_neighbors]
                elif access_pattern == GraphAccessPattern.RANDOM_WALK:
                    # Random walk sampling
                    sampled_neighbors = random.sample(neighbors, min(num_neighbors, len(neighbors)))
                else:
                    # Uniform sampling
                    sampled_neighbors = random.sample(neighbors, min(num_neighbors, len(neighbors)))
                
                for neighbor in sampled_neighbors:
                    if neighbor not in visited_nodes:
                        visited_nodes.add(neighbor)
                        next_frontier.append(neighbor)
                        self.node_access_frequency[neighbor] += 1
                
                sampling_latency += len(neighbors) * 0.001 * sampling_efficiency  # Neighbor access cost
            
            current_frontier = next_frontier
            if not current_frontier:
                break
        
        # Collect features for sampled nodes
        sampled_nodes = list(visited_nodes)
        node_features = {node_id: self.node_features.get(node_id, []) for node_id in sampled_nodes}
        
        # Collect edge information
        sampled_edges = []
        for src in sampled_nodes:
            for dst in self.adjacency_list.get(src, []):
                if dst in visited_nodes:
                    sampled_edges.append((src, dst))
        
        total_sampling_latency = sampling_latency + broadcast_latency
        
        result = {
            'center_node': center_node,
            'sampled_nodes': sampled_nodes,
            'sampled_edges': sampled_edges,
            'node_features': node_features,
            'num_nodes': len(sampled_nodes),
            'num_edges': len(sampled_edges),
            'latency': total_sampling_latency,
            'access_pattern': access_pattern.value,
            'thread_leader_access': is_thread_leader,
            'broadcast_latency': broadcast_latency
        }
        
        # Cache the result
        self.neighborhood_cache[cache_key] = result
        
        yield self.env.timeout(total_sampling_latency)
        return result
    
    def _batch_sample_subgraphs(self, seed_nodes, batch_size):
        """Sample multiple subgraphs for batch training"""
        self.subgraph_samples += 1
        
        batch_results = []
        total_latency = 0
        
        # Process in batches to optimize memory access
        for i in range(0, len(seed_nodes), batch_size):
            batch_seeds = seed_nodes[i:i + batch_size]
            batch_start_time = self.env.now
            
            # Sample subgraphs for this batch
            for seed_node in batch_seeds:
                subgraph_result = yield self.env.process(
                    self._sample_neighborhood(
                        seed_node, 
                        num_hops=2, 
                        num_neighbors=15,
                        access_pattern=GraphAccessPattern.NEIGHBORHOOD
                    )
                )
                batch_results.append(subgraph_result)
            
            batch_latency = self.env.now - batch_start_time
            total_latency += batch_latency
            
            # Simulate batch loading overhead
            yield self.env.timeout(0.1)
        
        return {
            'batch_size': len(seed_nodes),
            'subgraphs_sampled': len(batch_results),
            'total_latency': total_latency,
            'avg_latency_per_subgraph': total_latency / max(len(batch_results), 1),
            'subgraph_results': batch_results
        }
    
    def get_storage_stats(self):
        """Get GNN storage performance statistics"""
        cache_hit_rate = self.cache_hits / max(self.cache_hits + self.cache_misses, 1)
        
        # Calculate graph properties
        avg_degree = sum(len(neighbors) for neighbors in self.adjacency_list.values()) / max(self.node_count, 1)
        
        return {
            'storage_id': self.storage_id,
            'nodes': self.node_count,
            'edges': self.edge_count,
            'feature_dim': self.feature_dim,
            'avg_degree': avg_degree,
            'neighborhood_queries': self.neighborhood_queries,
            'subgraph_samples': self.subgraph_samples,
            'cache_hit_rate': cache_hit_rate,
            'cache_entries': len(self.neighborhood_cache),
            'sampling_strategy': self.sampling_strategy,
            'most_accessed_nodes': sorted(self.node_access_frequency.items(), 
                                        key=lambda x: x[1], reverse=True)[:10]
        }