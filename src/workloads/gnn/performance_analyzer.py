"""GNN performance analysis and reporting tools"""
import json
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from pathlib import Path

@dataclass
class PerformanceResult:
    """Individual performance test result"""
    test_name: str
    pattern_used: str
    execution_time: float
    storage_efficiency: float
    throughput: float
    latency: float
    sq_contention_rate: float = 0.0
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class BenchmarkSummary:
    """Summary of benchmark results"""
    total_tests: int
    successful_tests: int
    failed_tests: int
    best_pattern: str
    worst_pattern: str
    avg_storage_efficiency: float
    avg_throughput: float
    total_execution_time: float

class GNNPerformanceAnalyzer:
    """Analyzes and reports GNN performance metrics"""
    
    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        self.results: List[PerformanceResult] = []
        self.start_time = time.time()
    
    def add_result(self, result: PerformanceResult):
        """Add a performance test result"""
        self.results.append(result)
    
    def add_benchmark_result(self, test_name: str, pattern: str, metrics: Dict[str, Any]):
        """Add benchmark result from metrics dictionary"""
        result = PerformanceResult(
            test_name=test_name,
            pattern_used=pattern,
            execution_time=metrics.get('execution_time', 0.0),
            storage_efficiency=metrics.get('storage_efficiency', 0.0),
            throughput=metrics.get('throughput', 0.0),
            latency=metrics.get('avg_latency_per_warp', 0.0),
            sq_contention_rate=metrics.get('sq_contention_rate', 0.0),
            error_message=metrics.get('error')
        )
        self.add_result(result)
    
    def calculate_summary(self) -> BenchmarkSummary:
        """Calculate benchmark summary statistics"""
        if not self.results:
            return BenchmarkSummary(0, 0, 0, "", "", 0.0, 0.0, 0.0)
        
        successful = [r for r in self.results if r.error_message is None]
        failed = [r for r in self.results if r.error_message is not None]
        
        if successful:
            # Find best and worst patterns by storage efficiency
            best_result = max(successful, key=lambda x: x.storage_efficiency)
            worst_result = min(successful, key=lambda x: x.storage_efficiency)
            
            avg_efficiency = sum(r.storage_efficiency for r in successful) / len(successful)
            avg_throughput = sum(r.throughput for r in successful) / len(successful)
        else:
            best_result = worst_result = None
            avg_efficiency = avg_throughput = 0.0
        
        return BenchmarkSummary(
            total_tests=len(self.results),
            successful_tests=len(successful),
            failed_tests=len(failed),
            best_pattern=best_result.pattern_used if best_result else "",
            worst_pattern=worst_result.pattern_used if worst_result else "",
            avg_storage_efficiency=avg_efficiency,
            avg_throughput=avg_throughput,
            total_execution_time=time.time() - self.start_time
        )
    
    def generate_json_report(self, filename: str = None) -> str:
        """Generate JSON performance report"""
        if filename is None:
            timestamp = int(time.time())
            filename = f"gnn_performance_report_{timestamp}.json"
        
        summary = self.calculate_summary()
        
        report = {
            "metadata": {
                "timestamp": time.time(),
                "total_runtime_seconds": summary.total_execution_time,
                "report_version": "1.0"
            },
            "summary": asdict(summary),
            "results": [result.to_dict() for result in self.results],
            "pattern_comparison": self._generate_pattern_comparison(),
            "recommendations": self._generate_recommendations()
        }
        
        output_path = self.results_dir / filename
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        return str(output_path)
    
    def _generate_pattern_comparison(self) -> Dict[str, Any]:
        """Generate comparison between access patterns"""
        pattern_stats = {}
        
        for result in self.results:
            if result.error_message:
                continue
                
            pattern = result.pattern_used
            if pattern not in pattern_stats:
                pattern_stats[pattern] = {
                    'count': 0,
                    'total_efficiency': 0.0,
                    'total_throughput': 0.0,
                    'total_latency': 0.0,
                    'total_execution_time': 0.0
                }
            
            stats = pattern_stats[pattern]
            stats['count'] += 1
            stats['total_efficiency'] += result.storage_efficiency
            stats['total_throughput'] += result.throughput
            stats['total_latency'] += result.latency
            stats['total_execution_time'] += result.execution_time
        
        # Calculate averages
        comparison = {}
        for pattern, stats in pattern_stats.items():
            if stats['count'] > 0:
                comparison[pattern] = {
                    'test_count': stats['count'],
                    'avg_storage_efficiency': stats['total_efficiency'] / stats['count'],
                    'avg_throughput': stats['total_throughput'] / stats['count'],
                    'avg_latency': stats['total_latency'] / stats['count'],
                    'avg_execution_time': stats['total_execution_time'] / stats['count']
                }
        
        return comparison
    
    def _generate_recommendations(self) -> List[str]:
        """Generate performance recommendations"""
        recommendations = []
        
        if not self.results:
            return ["No test results available for analysis"]
        
        successful = [r for r in self.results if r.error_message is None]
        if not successful:
            return ["All tests failed - check configuration and implementation"]
        
        # Storage efficiency recommendations
        high_efficiency = [r for r in successful if r.storage_efficiency > 95.0]
        if high_efficiency:
            best_pattern = max(high_efficiency, key=lambda x: x.storage_efficiency)
            recommendations.append(
                f"Best storage efficiency: {best_pattern.pattern_used} "
                f"({best_pattern.storage_efficiency:.1f}% efficiency)"
            )
        
        # Throughput recommendations
        high_throughput = [r for r in successful if r.throughput > 0]
        if high_throughput:
            fastest_pattern = max(high_throughput, key=lambda x: x.throughput)
            recommendations.append(
                f"Highest throughput: {fastest_pattern.pattern_used} "
                f"({fastest_pattern.throughput:.2f} ops/sec)"
            )
        
        # SQ contention recommendations
        low_contention = [r for r in successful if r.sq_contention_rate < 0.05]
        if low_contention and len(low_contention) < len(successful):
            recommendations.append(
                "Consider increasing SQ count to reduce doorbell contention"
            )
        
        # Pattern-specific recommendations
        thread_0_results = [r for r in successful if 'thread_0' in r.pattern_used.lower()]
        if thread_0_results:
            avg_efficiency = sum(r.storage_efficiency for r in thread_0_results) / len(thread_0_results)
            if avg_efficiency > 90:
                recommendations.append(
                    "Thread 0 pattern shows excellent storage efficiency - "
                    "recommend for storage-bandwidth-constrained workloads"
                )
        
        return recommendations
    
    def print_summary(self):
        """Print performance summary to console"""
        summary = self.calculate_summary()
        
        print("\n" + "="*80)
        print("🎯 GNN PERFORMANCE ANALYSIS SUMMARY")
        print("="*80)
        
        print(f"📊 Test Results:")
        print(f"   Total tests: {summary.total_tests}")
        print(f"   Successful: {summary.successful_tests}")
        print(f"   Failed: {summary.failed_tests}")
        print(f"   Success rate: {(summary.successful_tests/summary.total_tests)*100:.1f}%")
        
        if summary.successful_tests > 0:
            print(f"\n🏆 Best Performance:")
            print(f"   Pattern: {summary.best_pattern}")
            print(f"   Avg Storage Efficiency: {summary.avg_storage_efficiency:.1f}%")
            print(f"   Avg Throughput: {summary.avg_throughput:.2f} ops/sec")
        
        print(f"\n⏱️  Total Execution Time: {summary.total_execution_time:.2f} seconds")
        
        # Pattern comparison
        comparison = self._generate_pattern_comparison()
        if comparison:
            print(f"\n📈 Pattern Comparison:")
            for pattern, stats in comparison.items():
                print(f"   {pattern}:")
                print(f"     Tests: {stats['test_count']}")
                print(f"     Avg Efficiency: {stats['avg_storage_efficiency']:.1f}%")
                print(f"     Avg Latency: {stats['avg_latency']:.2f} cycles")
        
        # Recommendations
        recommendations = self._generate_recommendations()
        if recommendations:
            print(f"\n💡 Recommendations:")
            for i, rec in enumerate(recommendations, 1):
                print(f"   {i}. {rec}")
        
        print("="*80)
    
    def export_csv(self, filename: str = None) -> str:
        """Export results to CSV format"""
        if filename is None:
            timestamp = int(time.time())
            filename = f"gnn_performance_results_{timestamp}.csv"
        
        output_path = self.results_dir / filename
        
        import csv
        with open(output_path, 'w', newline='') as csvfile:
            fieldnames = [
                'test_name', 'pattern_used', 'execution_time', 
                'storage_efficiency', 'throughput', 'latency', 
                'sq_contention_rate', 'error_message'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for result in self.results:
                writer.writerow(result.to_dict())
        
        return str(output_path)