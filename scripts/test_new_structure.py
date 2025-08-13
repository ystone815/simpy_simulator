#!/usr/bin/env python3
"""
Test script for the new modular structure
"""
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_config_system():
    """Test the configuration system"""
    print("🔧 Testing Configuration System")
    print("=" * 40)
    
    try:
        from config import SimulationConfig, GPUConfig, GNNConfig
        print("✅ Configuration classes imported successfully")
        
        # Test basic config creation
        sim_config = SimulationConfig()
        gpu_config = GPUConfig()
        gnn_config = GNNConfig()
        
        print(f"✅ Default configs created:")
        print(f"  - Simulation time: {sim_config.simulation_time}")
        print(f"  - GPU name: {gpu_config.name}")
        print(f"  - GNN execution mode: {gnn_config.execution_mode.value}")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_factory_system():
    """Test the factory system"""
    print("\n🏭 Testing Factory System")
    print("=" * 40)
    
    try:
        from src.utils.factory import SimulationFactory
        from src.utils.config_loader import create_config_template
        
        print("✅ Factory classes imported successfully")
        
        # Test environment creation
        env = SimulationFactory.create_environment()
        print(f"✅ SimPy environment created: {type(env)}")
        
        # Test config template creation
        template_path = create_config_template("default", "test_template.yaml")
        print(f"✅ Config template created: {template_path}")
        
        # Clean up
        if os.path.exists(template_path):
            os.remove(template_path)
        
        return True
        
    except Exception as e:
        print(f"❌ Factory test failed: {e}")
        return False

def test_directory_structure():
    """Test that the new directory structure is correct"""
    print("\n📁 Testing Directory Structure") 
    print("=" * 40)
    
    required_dirs = [
        "config",
        "src",
        "src/base", 
        "src/components",
        "src/components/gpu",
        "src/components/storage",
        "src/workloads",
        "src/workloads/gnn",
        "src/workloads/ai_storage",
        "src/utils",
        "tests",
        "tests/unit",
        "tests/integration", 
        "tests/benchmarks",
        "results",
        "scripts",
        "examples",
        "docs"
    ]
    
    missing_dirs = []
    for dir_path in required_dirs:
        full_path = project_root / dir_path
        if not full_path.exists():
            missing_dirs.append(dir_path)
        else:
            print(f"✅ {dir_path}")
    
    if missing_dirs:
        print(f"❌ Missing directories: {missing_dirs}")
        return False
    else:
        print("✅ All required directories present")
        return True

def test_config_files():
    """Test configuration files"""
    print("\n📄 Testing Configuration Files")
    print("=" * 40)
    
    config_files = [
        "config/default.yaml",
        "config/gnn_benchmark.yaml", 
        "config/b200_comparison.yaml"
    ]
    
    missing_files = []
    for config_file in config_files:
        full_path = project_root / config_file
        if not full_path.exists():
            missing_files.append(config_file)
        else:
            print(f"✅ {config_file}")
    
    if missing_files:
        print(f"❌ Missing config files: {missing_files}")
        return False
    else:
        print("✅ All configuration files present")
        return True

def test_yaml_loading():
    """Test YAML configuration loading"""
    print("\n📋 Testing YAML Configuration Loading")
    print("=" * 40)
    
    try:
        from src.utils.config_loader import ConfigLoader
        
        # Test loading default config
        config_data = ConfigLoader.load_yaml("config/default.yaml")
        print("✅ YAML loading successful")
        
        # Check basic structure
        required_sections = ["simulation", "gpu", "gnn", "benchmark"]
        missing_sections = []
        
        for section in required_sections:
            if section not in config_data:
                missing_sections.append(section)
            else:
                print(f"✅ Config section '{section}' found")
        
        if missing_sections:
            print(f"❌ Missing config sections: {missing_sections}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ YAML loading test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🎯 SimPy GPU Simulator - New Structure Test Suite")
    print("=" * 60)
    
    tests = [
        test_directory_structure,
        test_config_files, 
        test_yaml_loading,
        test_config_system,
        test_factory_system
    ]
    
    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test_func.__name__} crashed: {e}")
            results.append(False)
    
    print(f"\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    print(f"Success rate: {(passed/total)*100:.1f}%")
    
    if passed == total:
        print("✅ All tests passed! New structure is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())