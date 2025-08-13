"""Factory classes for creating configured simulation objects"""
import simpy
from typing import Dict, Any, Optional, List
from pathlib import Path

from config import (
    SimulationConfig, SystemConfig, GPUConfig, GNNConfig,
    BenchmarkConfig, load_gnn_config, load_gpu_config, load_benchmark_config
)
from config.gnn_config import AccessPattern, GraphFormat, ExecutionMode
from src.workloads.gnn import UnifiedGNNEngine
from .config_loader import ConfigLoader

class SimulationFactory:
    """Factory for creating simulation environments and components"""
    
    @staticmethod
    def create_environment(config_path: str = None, **overrides) -> simpy.Environment:
        """Create SimPy environment with optional configuration"""
        if config_path:
            config = ConfigLoader.load_simulation_config(config_path)
            # Apply overrides
            for key, value in overrides.items():
                if hasattr(config, key):
                    setattr(config, key, value)
        
        env = simpy.Environment()
        
        # Set random seed if specified
        seed = overrides.get('random_seed') or (config.random_seed if config_path else None)
        if seed is not None:
            import random
            random.seed(seed)
        
        return env
    
    @staticmethod
    def create_system_config(config_path: str = None, **overrides) -> SystemConfig:
        """Create system configuration with overrides"""
        if config_path:
            config = ConfigLoader.load_system_config(config_path)
        else:
            config = SystemConfig()
        
        # Apply overrides
        for key, value in overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        return config

class GPUSystemFactory:
    """Factory for creating GPU systems and components"""
    
    @staticmethod
    def create_gpu_config(gpu_type: str = "H100", config_path: str = None, **overrides) -> GPUConfig:
        """Create GPU configuration for specified architecture"""
        if config_path:
            config = load_gpu_config(config_path, gpu_type)
        else:
            if gpu_type.upper() == "H100":
                from config.gpu_config import H100Config
                config = H100Config()
            elif gpu_type.upper() == "B200":
                from config.gpu_config import B200Config
                config = B200Config()
            else:
                config = GPUConfig(name=gpu_type)
        
        # Apply overrides
        for key, value in overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        return config
    
    @staticmethod
    def create_gpu_system(env: simpy.Environment, gpu_config: GPUConfig, system_config: SystemConfig):
        """Create complete GPU system with all components"""
        # This would create actual GPU, memory, SM components
        # For now, return configuration for use by other factories
        return {
            'env': env,
            'gpu_config': gpu_config,
            'system_config': system_config,
            'total_sms': gpu_config.num_sms,
            'total_threads': gpu_config.total_threads,
            'memory_bandwidth': gpu_config.memory_bandwidth_gbps
        }

class GNNEngineFactory:
    """Factory for creating GNN engines with different configurations"""
    
    @staticmethod
    def create_gnn_config(config_path: str = None, execution_mode: str = "adaptive_hybrid", **overrides) -> GNNConfig:
        """Create GNN configuration with specified mode"""
        if config_path:
            config = load_gnn_config(config_path)
        else:
            config = GNNConfig()
        
        # Set execution mode
        if isinstance(execution_mode, str):
            config.execution_mode = ExecutionMode(execution_mode)
        
        # Apply overrides
        for key, value in overrides.items():
            if hasattr(config, key):
                if key == 'execution_mode' and isinstance(value, str):
                    setattr(config, key, ExecutionMode(value))
                elif key == 'graph_format' and isinstance(value, str):
                    setattr(config, key, GraphFormat(value))
                elif key == 'access_pattern' and isinstance(value, dict):
                    # Handle nested access pattern config
                    for pattern_key, pattern_value in value.items():
                        if hasattr(config.access_pattern, pattern_key):
                            if pattern_key == 'pattern' and isinstance(pattern_value, str):
                                setattr(config.access_pattern, pattern_key, AccessPattern(pattern_value))
                            else:
                                setattr(config.access_pattern, pattern_key, pattern_value)
                else:
                    setattr(config, key, value)
        
        return config
    
    @staticmethod
    def create_gnn_engine(env: simpy.Environment, config_path: str = None, **overrides) -> UnifiedGNNEngine:
        """Create unified GNN engine with configuration"""
        config = GNNEngineFactory.create_gnn_config(config_path, **overrides)
        return UnifiedGNNEngine(env, config)
    
    @staticmethod
    def create_storage_optimized_engine(env: simpy.Environment, **overrides) -> UnifiedGNNEngine:
        """Create GNN engine optimized for storage efficiency"""
        defaults = {
            'execution_mode': 'storage_optimized',
            'access_pattern': {
                'pattern': 'thread_0_leader',
                'enable_thread0_optimization': True,
                'sq_lock_optimization': False
            }
        }
        defaults.update(overrides)
        return GNNEngineFactory.create_gnn_engine(env, None, **defaults)
    
    @staticmethod
    def create_compute_optimized_engine(env: simpy.Environment, **overrides) -> UnifiedGNNEngine:
        """Create GNN engine optimized for compute performance"""
        defaults = {
            'execution_mode': 'compute_optimized',
            'access_pattern': {
                'pattern': 'multi_thread_parallel',
                'enable_thread0_optimization': False,
                'sq_lock_optimization': True
            }
        }
        defaults.update(overrides)
        return GNNEngineFactory.create_gnn_engine(env, None, **defaults)
    
    @staticmethod
    def create_adaptive_engine(env: simpy.Environment, **overrides) -> UnifiedGNNEngine:
        """Create adaptive GNN engine that switches patterns dynamically"""
        defaults = {
            'execution_mode': 'adaptive_hybrid',
            'access_pattern': {
                'pattern': 'hybrid_degree_based',
                'enable_thread0_optimization': True,
                'sq_lock_optimization': True,
                'degree_threshold': 32
            }
        }
        defaults.update(overrides)
        return GNNEngineFactory.create_gnn_engine(env, None, **defaults)

class BenchmarkFactory:
    """Factory for creating benchmark configurations and test suites"""
    
    @staticmethod
    def create_benchmark_config(config_path: str = None, benchmark_type: str = "default", **overrides) -> BenchmarkConfig:
        """Create benchmark configuration"""
        if config_path:
            config = load_benchmark_config(config_path, benchmark_type)
        else:
            if benchmark_type == "gnn":
                from config.benchmark_config import GNNBenchmarkConfig
                config = GNNBenchmarkConfig()
            elif benchmark_type == "comparison":
                from config.benchmark_config import ComparisonBenchmarkConfig  
                config = ComparisonBenchmarkConfig()
            else:
                config = BenchmarkConfig()
        
        # Apply overrides
        for key, value in overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        return config
    
    @staticmethod
    def create_gnn_benchmark_suite(env: simpy.Environment, config_path: str = None) -> Dict[str, Any]:
        """Create complete GNN benchmark test suite"""
        config = BenchmarkFactory.create_benchmark_config(config_path, "gnn")
        
        test_suite = {
            'config': config,
            'env': env,
            'test_engines': {},
            'test_configurations': []
        }
        
        # Create engines for different patterns
        for pattern in config.access_patterns:
            engine_config = {
                'execution_mode': 'adaptive_hybrid',
                'access_pattern': {'pattern': pattern},
                'num_warps': 8
            }
            
            engine = GNNEngineFactory.create_gnn_engine(env, None, **engine_config)
            test_suite['test_engines'][pattern] = engine
        
        # Generate test configurations
        for workload_size in config.workload_sizes:
            for pattern in config.access_patterns:
                for graph_type in config.graph_types:
                    test_config = {
                        'name': f"{graph_type}_{pattern}_w{workload_size}",
                        'workload_size': workload_size,
                        'pattern': pattern,
                        'graph_type': graph_type,
                        'num_warps': workload_size,
                        'nodes': workload_size * 32,
                        'edges': workload_size * 32 * 5
                    }
                    test_suite['test_configurations'].append(test_config)
        
        return test_suite
    
    @staticmethod
    def create_comparison_benchmark(env: simpy.Environment, config_path: str = None) -> Dict[str, Any]:
        """Create architecture comparison benchmark"""
        config = BenchmarkFactory.create_benchmark_config(config_path, "comparison")
        
        comparison_suite = {
            'config': config,
            'env': env,
            'gpu_configs': {},
            'test_matrix': []
        }
        
        # Create GPU configurations for comparison
        for gpu_arch in config.gpu_architectures:
            gpu_config = GPUSystemFactory.create_gpu_config(gpu_arch)
            comparison_suite['gpu_configs'][gpu_arch] = gpu_config
        
        # Generate test matrix
        for gpu_arch in config.gpu_architectures:
            for optimization in config.optimization_levels:
                for workload_size in config.workload_sizes:
                    test_config = {
                        'name': f"{gpu_arch}_{optimization}_w{workload_size}",
                        'gpu_architecture': gpu_arch,
                        'optimization_level': optimization,
                        'workload_size': workload_size,
                        'num_runs': config.num_runs_per_config
                    }
                    comparison_suite['test_matrix'].append(test_config)
        
        return comparison_suite

class ConfigurableFactory:
    """Meta-factory that uses configuration files to determine what to create"""
    
    @staticmethod
    def create_from_config(config_path: str, component_type: str = "auto", **overrides) -> Any:
        """Create any component from configuration file"""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        # Auto-detect component type from filename if not specified
        if component_type == "auto":
            filename = config_path.stem.lower()
            if "gnn" in filename:
                component_type = "gnn"
            elif "gpu" in filename:
                component_type = "gpu"
            elif "benchmark" in filename:
                component_type = "benchmark"
            else:
                component_type = "simulation"
        
        # Create environment first
        env = SimulationFactory.create_environment(str(config_path), **overrides)
        
        # Create specific component
        if component_type == "gnn":
            return GNNEngineFactory.create_gnn_engine(env, str(config_path), **overrides)
        elif component_type == "gpu":
            gpu_type = overrides.get('gpu_type', 'H100')
            return GPUSystemFactory.create_gpu_config(gpu_type, str(config_path), **overrides)
        elif component_type == "benchmark":
            benchmark_type = overrides.get('benchmark_type', 'default')
            return BenchmarkFactory.create_benchmark_config(str(config_path), benchmark_type, **overrides)
        else:
            return env
    
    @staticmethod
    def create_complete_system(config_path: str, **overrides) -> Dict[str, Any]:
        """Create complete simulation system from configuration"""
        env = SimulationFactory.create_environment(config_path, **overrides)
        system_config = SimulationFactory.create_system_config(config_path, **overrides)
        
        # Determine GPU type from config or overrides
        gpu_type = overrides.get('gpu_type', 'H100')
        gpu_config = GPUSystemFactory.create_gpu_config(gpu_type, config_path, **overrides)
        
        # Create GNN engine
        gnn_engine = GNNEngineFactory.create_gnn_engine(env, config_path, **overrides)
        
        # Create GPU system
        gpu_system = GPUSystemFactory.create_gpu_system(env, gpu_config, system_config)
        
        return {
            'env': env,
            'system_config': system_config,
            'gpu_config': gpu_config,
            'gpu_system': gpu_system,
            'gnn_engine': gnn_engine,
            'total_threads': gpu_config.total_threads,
            'total_sms': gpu_config.num_sms
        }