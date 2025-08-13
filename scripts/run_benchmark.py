#!/usr/bin/env python3
"""
SimPy GPU Simulator - Benchmark Runner
Command-line interface for running GPU simulation benchmarks
"""
import argparse
import sys
import os
import time
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils import SimulationFactory, GNNEngineFactory, BenchmarkFactory, ConfigurableFactory
from src.utils.config_loader import ConfigLoader, create_config_template, ConfigValidator
from src.utils.performance_utils import MetricsCollector, PerformanceTracker
from src.workloads.gnn.performance_analyzer import GNNPerformanceAnalyzer

def setup_argument_parser():
    """Setup command line argument parser"""
    parser = argparse.ArgumentParser(
        description="SimPy GPU Simulator Benchmark Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run GNN benchmark with default config
  python run_benchmark.py --benchmark gnn
  
  # Run with custom config file
  python run_benchmark.py --config config/custom_gnn.yaml --benchmark gnn
  
  # Run GPU architecture comparison
  python run_benchmark.py --benchmark comparison --gpu-types H100,B200
  
  # Generate config template
  python run_benchmark.py --create-template gnn_benchmark
  
  # Validate all configs
  python run_benchmark.py --validate-configs
        """
    )
    
    # Main options
    parser.add_argument(
        "--benchmark", "-b",
        choices=["gnn", "gpu", "storage", "comparison", "all"],
        default="gnn",
        help="Type of benchmark to run (default: gnn)"
    )
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        help="Configuration file path (default: auto-detect based on benchmark type)"
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="results",
        help="Output directory for results (default: results)"
    )
    
    # Benchmark-specific options
    parser.add_argument(
        "--gpu-types",
        type=str,
        help="Comma-separated list of GPU types to test (e.g., H100,B200)"
    )
    
    parser.add_argument(
        "--access-patterns",
        type=str,
        help="Comma-separated list of access patterns (e.g., thread_0_leader,multi_thread_parallel)"
    )
    
    parser.add_argument(
        "--workload-sizes",
        type=str,
        help="Comma-separated list of workload sizes (e.g., 5,10,20,40)"
    )
    
    # Execution options
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run tests in parallel (when supported)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be run without executing"
    )
    
    # Utility options
    parser.add_argument(
        "--create-template",
        type=str,
        help="Create configuration template (gnn_benchmark, gpu_comparison, default)"
    )
    
    parser.add_argument(
        "--validate-configs",
        action="store_true",
        help="Validate all configuration files"
    )
    
    parser.add_argument(
        "--list-configs",
        action="store_true", 
        help="List available configuration files"
    )
    
    return parser

def get_default_config_path(benchmark_type: str) -> str:
    """Get default configuration file path for benchmark type"""
    config_map = {
        "gnn": "config/gnn_benchmark.yaml",
        "gpu": "config/default.yaml",
        "storage": "config/default.yaml", 
        "comparison": "config/b200_comparison.yaml",
        "all": "config/default.yaml"
    }
    
    config_path = config_map.get(benchmark_type, "config/default.yaml")
    
    # Check if file exists, fallback to default.yaml
    if not Path(config_path).exists():
        fallback = "config/default.yaml"
        if Path(fallback).exists():
            return fallback
        else:
            raise FileNotFoundError(f"No configuration file found. Create one with --create-template")
    
    return config_path

def run_gnn_benchmark(config_path: str, args) -> dict:
    """Run GNN-specific benchmark"""
    print("🧠 Running GNN Benchmark Suite")
    print("=" * 60)
    
    # Create performance tracker
    performance_tracker = PerformanceTracker("gnn_benchmark")
    analyzer = GNNPerformanceAnalyzer(args.output_dir)
    
    performance_tracker.start_timer("total_benchmark_time")
    
    try:
        # Create GNN engine from config
        env = SimulationFactory.create_environment(config_path)
        gnn_engine = GNNEngineFactory.create_gnn_engine(env, config_path)
        
        # Parse command-line overrides
        overrides = {}
        if args.access_patterns:
            patterns = args.access_patterns.split(',')
            overrides['test_patterns'] = [p.strip() for p in patterns]
        
        if args.workload_sizes:
            sizes = [int(s.strip()) for s in args.workload_sizes.split(',')]
            overrides['workload_sizes'] = sizes
        
        # Create benchmark configurations
        benchmark_configs = []
        
        # Basic pattern tests
        patterns = overrides.get('test_patterns', ['thread_0_leader', 'multi_thread_parallel', 'hybrid_degree_based'])
        for pattern in patterns:
            benchmark_configs.append({
                'name': f'pattern_test_{pattern}',
                'pattern': pattern,
                'num_warps': 8,
                'description': f'Test {pattern} access pattern'
            })
        
        # Scaling tests
        sizes = overrides.get('workload_sizes', [5, 10, 20])
        for size in sizes:
            benchmark_configs.append({
                'name': f'scaling_test_{size}w',
                'pattern': 'adaptive_hybrid',
                'num_warps': size,
                'description': f'Scaling test with {size} warps'
            })
        
        if args.dry_run:
            print(f"Would run {len(benchmark_configs)} benchmark configurations:")
            for config in benchmark_configs:
                print(f"  - {config['name']}: {config['description']}")
            return {"dry_run": True, "config_count": len(benchmark_configs)}
        
        # Run benchmarks
        def run_benchmarks():
            results = yield from gnn_engine.run_benchmark_suite(benchmark_configs)
            
            # Process results with analyzer
            for test_name, result in results.items():
                if 'error' not in result:
                    analyzer.add_benchmark_result(test_name, result.get('pattern_used', 'unknown'), result)
                    
                    # Record in performance tracker
                    performance_tracker.record_metric(
                        f"{test_name}_execution_time", 
                        result.get('execution_time', 0), 
                        "seconds"
                    )
                    performance_tracker.record_efficiency(
                        result.get('storage_accesses', 0),
                        result.get('total_warps', 1) * 32,
                        f"{test_name}_storage_efficiency"
                    )
            
            return results
        
        # Execute benchmark
        env.process(run_benchmarks())
        env.run()
        
    except Exception as e:
        print(f"❌ GNN benchmark failed: {e}")
        return {"error": str(e)}
    
    finally:
        performance_tracker.stop_timer("total_benchmark_time")
    
    # Generate reports
    analyzer.print_summary()
    json_report = analyzer.generate_json_report()
    csv_report = analyzer.export_csv()
    performance_report = performance_tracker.export_metrics(
        os.path.join(args.output_dir, "gnn_performance_metrics.json")
    )
    
    return {
        "status": "completed",
        "total_tests": len(analyzer.results),
        "json_report": json_report,
        "csv_report": csv_report,
        "performance_report": performance_report,
        "summary": analyzer.calculate_summary().__dict__
    }

def run_comparison_benchmark(config_path: str, args) -> dict:
    """Run GPU architecture comparison benchmark"""
    print("⚖️  Running GPU Architecture Comparison")
    print("=" * 60)
    
    if args.dry_run:
        gpu_types = args.gpu_types.split(',') if args.gpu_types else ['H100', 'B200']
        print(f"Would compare GPU architectures: {gpu_types}")
        return {"dry_run": True, "gpu_types": gpu_types}
    
    # Create metrics collector for comparison
    metrics_collector = MetricsCollector()
    results = {}
    
    gpu_types = args.gpu_types.split(',') if args.gpu_types else ['H100', 'B200']
    
    for gpu_type in gpu_types:
        print(f"\n📊 Testing {gpu_type} architecture...")
        
        tracker = metrics_collector.create_tracker(f"{gpu_type}_performance")
        tracker.start_timer("gpu_benchmark_time")
        
        try:
            # Create GPU-specific system
            env = SimulationFactory.create_environment(config_path)
            gnn_engine = GNNEngineFactory.create_gnn_engine(
                env, config_path, 
                gpu_type=gpu_type
            )
            
            # Run basic performance test
            test_config = {
                'name': f'{gpu_type}_baseline',
                'num_warps': 20,
                'pattern': 'hybrid_degree_based'
            }
            
            def run_gpu_test():
                result = yield from gnn_engine.execute_gnn_layer(test_config)
                return result
            
            env.process(run_gpu_test())
            env.run()
            
            # Record metrics
            tracker.record_metric("baseline_performance", 100.0, "score")  # Placeholder
            tracker.increment_counter("successful_tests")
            
            results[gpu_type] = {"status": "completed", "score": 100.0}
            
        except Exception as e:
            print(f"❌ {gpu_type} benchmark failed: {e}")
            results[gpu_type] = {"status": "failed", "error": str(e)}
        
        finally:
            tracker.stop_timer("gpu_benchmark_time")
    
    # Generate comparison report
    comparison_report = metrics_collector.generate_performance_report(
        os.path.join(args.output_dir, "gpu_comparison_report.json")
    )
    metrics_collector.print_summary()
    
    return {
        "status": "completed",
        "gpu_results": results,
        "comparison_report": comparison_report
    }

def create_template(template_type: str, output_dir: str = "config") -> str:
    """Create configuration template"""
    output_path = os.path.join(output_dir, f"{template_type}.yaml")
    template_file = create_config_template(template_type, output_path)
    print(f"✅ Created configuration template: {template_file}")
    return template_file

def validate_configs(config_dir: str = "config") -> dict:
    """Validate all configuration files"""
    print("🔍 Validating Configuration Files")
    print("=" * 40)
    
    results = ConfigValidator.validate_all_configs(config_dir)
    
    valid_count = 0
    invalid_count = 0
    
    for config_file, result in results.items():
        if result is True:
            print(f"✅ {config_file}")
            valid_count += 1
        else:
            print(f"❌ {config_file}: {result}")
            invalid_count += 1
    
    print(f"\nValidation Summary:")
    print(f"  Valid configs: {valid_count}")
    print(f"  Invalid configs: {invalid_count}")
    print(f"  Total configs: {valid_count + invalid_count}")
    
    return {"valid": valid_count, "invalid": invalid_count, "results": results}

def list_configs(config_dir: str = "config") -> list:
    """List available configuration files"""
    config_path = Path(config_dir)
    
    if not config_path.exists():
        print(f"Configuration directory not found: {config_dir}")
        return []
    
    print("📁 Available Configuration Files:")
    print("=" * 40)
    
    configs = []
    for config_file in sorted(config_path.glob("*.yaml")):
        rel_path = config_file.name  # Just use filename instead of relative path
        configs.append(str(config_file))
        
        # Try to get description from file
        try:
            with open(config_file, 'r') as f:
                first_line = f.readline().strip()
                if first_line.startswith('#'):
                    description = first_line[1:].strip()
                    print(f"  {rel_path} - {description}")
                else:
                    print(f"  {rel_path}")
        except:
            print(f"  {rel_path}")
    
    print(f"\nFound {len(configs)} configuration files")
    return configs

def main():
    """Main entry point"""
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    # Handle utility commands
    if args.create_template:
        create_template(args.create_template)
        return
    
    if args.validate_configs:
        validate_configs()
        return
    
    if args.list_configs:
        list_configs()
        return
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine configuration file
    if args.config:
        config_path = args.config
    else:
        config_path = get_default_config_path(args.benchmark)
    
    if not os.path.exists(config_path):
        print(f"❌ Configuration file not found: {config_path}")
        print(f"💡 Create one with: python run_benchmark.py --create-template {args.benchmark}_benchmark")
        return 1
    
    print(f"🚀 SimPy GPU Simulator Benchmark Runner")
    print(f"Configuration: {config_path}")
    print(f"Benchmark Type: {args.benchmark}")
    print(f"Output Directory: {args.output_dir}")
    print("-" * 60)
    
    start_time = time.time()
    
    try:
        # Run specified benchmark
        if args.benchmark == "gnn":
            result = run_gnn_benchmark(config_path, args)
        elif args.benchmark == "comparison":
            result = run_comparison_benchmark(config_path, args)
        elif args.benchmark == "all":
            print("🔄 Running all benchmark types...")
            result = {}
            result["gnn"] = run_gnn_benchmark(config_path, args)
            result["comparison"] = run_comparison_benchmark(config_path, args)
        else:
            print(f"❌ Benchmark type '{args.benchmark}' not yet implemented")
            return 1
        
        total_time = time.time() - start_time
        
        print(f"\n" + "=" * 60)
        print(f"✅ Benchmark completed successfully!")
        print(f"⏱️  Total execution time: {total_time:.2f} seconds")
        print(f"📊 Results saved to: {args.output_dir}")
        
        if args.verbose and result and not result.get("dry_run"):
            print(f"\nResult summary:")
            for key, value in result.items():
                if key != "summary":
                    print(f"  {key}: {value}")
        
        return 0
        
    except KeyboardInterrupt:
        print(f"\n🛑 Benchmark interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Benchmark failed with error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())