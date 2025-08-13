#!/usr/bin/env python3
"""
Development environment setup script
"""
import os
import sys
import subprocess
from pathlib import Path

def setup_python_path():
    """Add project directories to Python path"""
    project_root = Path(__file__).parent.parent
    
    # Create __init__.py files if missing
    init_files = [
        project_root / "src" / "__init__.py",
        project_root / "src" / "components" / "__init__.py", 
        project_root / "src" / "workloads" / "__init__.py",
        project_root / "src" / "workloads" / "ai_storage" / "__init__.py",
        project_root / "src" / "workloads" / "cuda_kernels" / "__init__.py",
        project_root / "src" / "components" / "gpu" / "__init__.py",
        project_root / "src" / "components" / "storage" / "__init__.py",
        project_root / "src" / "components" / "networking" / "__init__.py"
    ]
    
    for init_file in init_files:
        if not init_file.exists():
            init_file.touch()
            print(f"Created {init_file}")

def install_dependencies():
    """Install required Python packages"""
    try:
        import simpy
        print("✅ simpy already installed")
    except ImportError:
        print("📦 Installing simpy...")
        subprocess.run([sys.executable, "-m", "pip", "install", "simpy", "--break-system-packages"], check=True)

def create_example_runner():
    """Create example script to test the new structure"""
    example_script = Path(__file__).parent / "example_usage.py"
    
    content = '''#!/usr/bin/env python3
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
    print("\\n1. Creating storage-optimized GNN engine...")
    storage_engine = GNNEngineFactory.create_storage_optimized_engine(env)
    print(f"   Current pattern: {storage_engine.current_pattern.value}")
    
    print("\\n2. Creating compute-optimized GNN engine...")
    compute_engine = GNNEngineFactory.create_compute_optimized_engine(env)
    print(f"   Current pattern: {compute_engine.current_pattern.value}")
    
    print("\\n3. Creating adaptive GNN engine...")
    adaptive_engine = GNNEngineFactory.create_adaptive_engine(env)
    print(f"   Current pattern: {adaptive_engine.current_pattern.value}")
    
    # Test configuration loading
    print("\\n4. Testing configuration loading...")
    try:
        config_engine = GNNEngineFactory.create_gnn_engine(env, "config/gnn_benchmark.yaml")
        print(f"   Config-based engine created successfully")
        print(f"   Execution mode: {config_engine.config.execution_mode.value}")
    except Exception as e:
        print(f"   Config loading failed (expected): {e}")
    
    print("\\n✅ Example completed successfully!")

if __name__ == "__main__":
    main()
'''
    
    with open(example_script, 'w') as f:
        f.write(content)
    
    os.chmod(example_script, 0o755)
    print(f"✅ Created example script: {example_script}")

def main():
    """Main setup function"""
    print("🔧 Setting up SimPy GPU Simulator development environment...")
    
    setup_python_path()
    install_dependencies() 
    create_example_runner()
    
    print("\\n✅ Development environment setup completed!")
    print("\\nNext steps:")
    print("  1. Run example: python scripts/example_usage.py")
    print("  2. Run benchmarks: python scripts/run_benchmark.py --help")
    print("  3. Create config templates: python scripts/run_benchmark.py --create-template gnn_benchmark")

if __name__ == "__main__":
    main()