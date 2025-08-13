"""Configuration loading utilities"""
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

from config import (
    SimulationConfig, SystemConfig, GPUConfig, GNNConfig, BenchmarkConfig,
    load_gnn_config, load_gpu_config, load_benchmark_config
)

class ConfigLoader:
    """Utility class for loading and merging configuration files"""
    
    @staticmethod
    def load_yaml(config_path: str) -> Dict[str, Any]:
        """Load YAML configuration file"""
        config_file = Path(config_path)
        
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)
    
    @staticmethod
    def load_simulation_config(config_path: str) -> SimulationConfig:
        """Load simulation configuration from YAML"""
        return SimulationConfig.from_yaml(config_path)
    
    @staticmethod
    def load_system_config(config_path: str) -> SystemConfig:
        """Load system configuration from YAML"""
        return SystemConfig.from_yaml(config_path)
    
    @staticmethod
    def merge_configs(*config_paths: str) -> Dict[str, Any]:
        """Merge multiple configuration files"""
        merged_config = {}
        
        for path in config_paths:
            config = ConfigLoader.load_yaml(path)
            merged_config = ConfigLoader._deep_merge(merged_config, config)
        
        return merged_config
    
    @staticmethod
    def _deep_merge(base_dict: Dict[str, Any], override_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries"""
        result = base_dict.copy()
        
        for key, value in override_dict.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = ConfigLoader._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    @staticmethod
    def validate_config(config: Dict[str, Any], required_keys: list) -> bool:
        """Validate that configuration contains required keys"""
        for key in required_keys:
            if '.' in key:
                # Handle nested keys like 'gnn.execution_mode'
                keys = key.split('.')
                current = config
                for k in keys:
                    if k not in current:
                        raise ValueError(f"Missing required configuration key: {key}")
                    current = current[k]
            else:
                if key not in config:
                    raise ValueError(f"Missing required configuration key: {key}")
        
        return True

def load_config_with_overrides(config_path: str, overrides: Dict[str, Any] = None) -> Dict[str, Any]:
    """Load configuration file and apply overrides"""
    config = ConfigLoader.load_yaml(config_path)
    
    if overrides:
        config = ConfigLoader._deep_merge(config, overrides)
    
    return config

def create_config_template(template_type: str = "default", output_path: str = None) -> str:
    """Create a configuration template file"""
    if output_path is None:
        output_path = f"{template_type}_config_template.yaml"
    
    templates = {
        "default": {
            "simulation": {
                "simulation_time": 50000,
                "random_seed": 42,
                "log_level": "INFO",
                "enable_tracing": False,
                "enable_metrics": True
            },
            "gpu": {
                "name": "H100",
                "num_sms": 144,
                "memory_bandwidth_gbps": 2000.0,
                "memory_size_gb": 80
            },
            "gnn": {
                "execution_mode": "adaptive_hybrid",
                "graph_format": "hybrid_adaptive",
                "access_pattern": {
                    "pattern": "hybrid_degree_based",
                    "degree_threshold": 32,
                    "enable_thread0_optimization": True
                },
                "graph": {
                    "num_nodes": 1000,
                    "num_edges": 5000,
                    "graph_type": "random"
                }
            }
        },
        
        "gnn_benchmark": {
            "simulation": {
                "simulation_time": 50000,
                "random_seed": 42
            },
            "gnn": {
                "execution_mode": "adaptive_hybrid",
                "num_layers": 3,
                "num_warps": 8
            },
            "benchmark": {
                "name": "gnn_benchmark",
                "run_performance_tests": True,
                "access_patterns": [
                    "thread_0_leader",
                    "multi_thread_parallel",
                    "edge_centric_cugraph",
                    "hybrid_degree_based"
                ],
                "workload_sizes": [5, 10, 20, 40],
                "save_results": True,
                "results_directory": "results/gnn_benchmarks"
            }
        },
        
        "gpu_comparison": {
            "simulation": {
                "simulation_time": 50000,
                "random_seed": 42
            },
            "benchmark": {
                "name": "gpu_architecture_comparison",
                "gpu_architectures": ["H100", "B200"],
                "optimization_levels": ["none", "thread_0", "hybrid"],
                "workload_sizes": [10, 20, 40, 80],
                "num_runs_per_config": 5,
                "save_results": True,
                "results_directory": "results/gpu_comparison"
            }
        }
    }
    
    if template_type not in templates:
        raise ValueError(f"Unknown template type: {template_type}. Available: {list(templates.keys())}")
    
    template = templates[template_type]
    
    output_file = Path(output_path)
    with open(output_file, 'w') as f:
        yaml.dump(template, f, default_flow_style=False, indent=2)
    
    return str(output_file)

class ConfigValidator:
    """Validates configuration files against expected schemas"""
    
    GNN_REQUIRED_KEYS = [
        'gnn.execution_mode',
        'gnn.access_pattern.pattern',
        'gnn.graph.num_nodes',
        'gnn.graph.num_edges'
    ]
    
    BENCHMARK_REQUIRED_KEYS = [
        'benchmark.name',
        'benchmark.save_results'
    ]
    
    @staticmethod
    def validate_gnn_config(config_path: str) -> bool:
        """Validate GNN configuration file"""
        config = ConfigLoader.load_yaml(config_path)
        return ConfigLoader.validate_config(config, ConfigValidator.GNN_REQUIRED_KEYS)
    
    @staticmethod
    def validate_benchmark_config(config_path: str) -> bool:
        """Validate benchmark configuration file"""
        config = ConfigLoader.load_yaml(config_path)
        return ConfigLoader.validate_config(config, ConfigValidator.BENCHMARK_REQUIRED_KEYS)
    
    @staticmethod
    def validate_all_configs(config_dir: str = "config") -> Dict[str, bool]:
        """Validate all configuration files in directory"""
        config_path = Path(config_dir)
        results = {}
        
        if not config_path.exists():
            return {"error": f"Config directory {config_dir} not found"}
        
        for config_file in config_path.glob("*.yaml"):
            try:
                filename = config_file.stem.lower()
                
                if "gnn" in filename:
                    results[str(config_file)] = ConfigValidator.validate_gnn_config(str(config_file))
                elif "benchmark" in filename:
                    results[str(config_file)] = ConfigValidator.validate_benchmark_config(str(config_file))
                else:
                    # Generic validation - just check if it loads
                    ConfigLoader.load_yaml(str(config_file))
                    results[str(config_file)] = True
                    
            except Exception as e:
                results[str(config_file)] = f"Validation failed: {e}"
        
        return results