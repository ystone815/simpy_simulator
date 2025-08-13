"""GPU architecture configuration classes"""
from dataclasses import dataclass
from typing import Dict, List, Optional
import yaml

@dataclass
class GPUConfig:
    """Base GPU configuration"""
    name: str = "Generic"
    num_sms: int = 108
    threads_per_warp: int = 32
    warps_per_sm: int = 64
    max_threads_per_sm: int = 2048
    memory_bandwidth_gbps: float = 2000.0
    memory_size_gb: int = 80
    tensor_cores: bool = True
    
    @property
    def total_threads(self) -> int:
        return self.num_sms * self.max_threads_per_sm
    
    @property 
    def total_warps(self) -> int:
        return self.num_sms * self.warps_per_sm

@dataclass
class H100Config(GPUConfig):
    """NVIDIA H100 GPU configuration"""
    name: str = "H100"
    num_sms: int = 144
    memory_bandwidth_gbps: float = 2000.0
    memory_size_gb: int = 80
    tensor_precision: List[str] = None
    transformer_engine: bool = True
    
    def __post_init__(self):
        if self.tensor_precision is None:
            self.tensor_precision = ["FP32", "FP16", "BF16", "FP8"]

@dataclass
class B200Config(GPUConfig):
    """NVIDIA B200 GPU configuration"""
    name: str = "B200"
    num_sms: int = 144  # Total across dual chiplets
    chiplets: int = 2
    sms_per_chiplet: int = 72
    memory_bandwidth_gbps: float = 8000.0
    memory_size_gb: int = 192
    tensor_precision: List[str] = None
    shader_execution_reordering: bool = True
    sparsity_support: bool = True
    
    def __post_init__(self):
        if self.tensor_precision is None:
            self.tensor_precision = ["FP32", "FP16", "BF16", "FP8", "FP4"]

@dataclass
class MemoryConfig:
    """GPU memory hierarchy configuration"""
    hbm_bandwidth_gbps: float = 2000.0
    hbm_size_gb: int = 80
    l1_cache_kb: int = 128
    l2_cache_mb: int = 40
    shared_memory_kb: int = 164
    register_file_kb: int = 256
    
    cache_hit_latency_cycles: Dict[str, int] = None
    cache_miss_penalty_cycles: Dict[str, int] = None
    
    def __post_init__(self):
        if self.cache_hit_latency_cycles is None:
            self.cache_hit_latency_cycles = {
                'l1': 4,
                'l2': 200,
                'shared_mem': 1,
                'register': 1
            }
        
        if self.cache_miss_penalty_cycles is None:
            self.cache_miss_penalty_cycles = {
                'l1_to_l2': 200,
                'l2_to_hbm': 350,
                'shared_mem_bank_conflict': 32
            }

def load_gpu_config(config_path: str, gpu_type: str = "H100") -> GPUConfig:
    """Load GPU configuration from YAML file"""
    with open(config_path, 'r') as f:
        data = yaml.safe_load(f)
    
    gpu_data = data.get('gpu', {})
    
    if gpu_type.upper() == "H100":
        return H100Config(**gpu_data)
    elif gpu_type.upper() == "B200":
        return B200Config(**gpu_data)
    else:
        return GPUConfig(**gpu_data)