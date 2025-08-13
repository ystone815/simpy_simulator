#!/usr/bin/env python3
"""
Example usage of the modular SimPy GPU Simulator
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils import GNNEngineFactory, SimulationFactory
from config.gnn_config import ExecutionMode, AccessPattern

def main():
    print("🚀 SimPy GPU Simulator - Example Usage")
    print("=" * 50)
    
    # Create simulation environment
    env = SimulationFactory.create_environment()
    
    # Create different types of GNN engines
    print("\n1. Creating storage-optimized GNN engine...")
    storage_engine = GNNEngineFactory.create_storage_optimized_engine(env)
    print(f"   Current pattern: {storage_engine.current_pattern.value}")
    
    print("\n2. Creating compute-optimized GNN engine...")
    compute_engine = GNNEngineFactory.create_compute_optimized_engine(env)
    print(f"   Current pattern: {compute_engine.current_pattern.value}")
    
    print("\n3. Creating adaptive GNN engine...")
    adaptive_engine = GNNEngineFactory.create_adaptive_engine(env)
    print(f"   Current pattern: {adaptive_engine.current_pattern.value}")
    
    # Test configuration loading
    print("\n4. Testing configuration loading...")
    try:
        config_engine = GNNEngineFactory.create_gnn_engine(env, "config/gnn_benchmark.yaml")
        print(f"   Config-based engine created successfully")
        print(f"   Execution mode: {config_engine.config.execution_mode.value}")
    except Exception as e:
        print(f"   Config loading failed (expected): {e}")
    
    print("\n✅ Example completed successfully!")

if __name__ == "__main__":
    main()
