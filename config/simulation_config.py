"""Simulation configuration classes"""
from dataclasses import dataclass
from typing import Optional, Dict, Any
import yaml

@dataclass
class SimulationConfig:
    """Core simulation parameters"""
    simulation_time: int = 50000
    random_seed: Optional[int] = 42
    log_level: str = "INFO"
    enable_tracing: bool = False
    enable_metrics: bool = True
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'SimulationConfig':
        """Load configuration from YAML file"""
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(**data.get('simulation', {}))

@dataclass
class SystemConfig:
    """System-level configuration"""
    memory_size_gb: int = 32
    cpu_cores: int = 8
    nvme_sqs: int = 1024
    enable_doorbell_optimization: bool = True
    cache_sizes: Dict[str, int] = None
    
    def __post_init__(self):
        if self.cache_sizes is None:
            self.cache_sizes = {
                'l1_cache_kb': 128,
                'l2_cache_mb': 40,
                'shared_memory_kb': 164
            }
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'SystemConfig':
        """Load configuration from YAML file"""
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(**data.get('system', {}))

@dataclass
class MetricsConfig:
    """Performance metrics configuration"""
    collect_latency: bool = True
    collect_throughput: bool = True
    collect_storage_efficiency: bool = True
    collect_sq_contention: bool = True
    output_format: str = "json"
    save_detailed_logs: bool = False