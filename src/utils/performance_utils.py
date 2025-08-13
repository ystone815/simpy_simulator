"""Performance tracking and metrics utilities"""
import time
import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from pathlib import Path

@dataclass
class PerformanceMetric:
    """Individual performance metric"""
    name: str
    value: float
    unit: str
    timestamp: float
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class PerformanceTracker:
    """Tracks performance metrics during simulation execution"""
    
    def __init__(self, name: str = "simulation"):
        self.name = name
        self.metrics: List[PerformanceMetric] = []
        self.start_time = time.time()
        self.timers: Dict[str, float] = {}
        self.counters: Dict[str, int] = {}
        
    def start_timer(self, timer_name: str):
        """Start a named timer"""
        self.timers[timer_name] = time.time()
    
    def stop_timer(self, timer_name: str, unit: str = "seconds") -> float:
        """Stop a named timer and record the duration"""
        if timer_name not in self.timers:
            raise ValueError(f"Timer '{timer_name}' was not started")
        
        duration = time.time() - self.timers[timer_name]
        self.record_metric(timer_name, duration, unit)
        del self.timers[timer_name]
        return duration
    
    def increment_counter(self, counter_name: str, increment: int = 1):
        """Increment a named counter"""
        if counter_name not in self.counters:
            self.counters[counter_name] = 0
        self.counters[counter_name] += increment
        
        # Record current counter value as metric
        self.record_metric(counter_name, self.counters[counter_name], "count")
    
    def record_metric(self, name: str, value: float, unit: str, metadata: Dict[str, Any] = None):
        """Record a performance metric"""
        metric = PerformanceMetric(
            name=name,
            value=value,
            unit=unit,
            timestamp=time.time(),
            metadata=metadata or {}
        )
        self.metrics.append(metric)
    
    def record_throughput(self, operations: int, duration: float, operation_type: str = "operations"):
        """Record throughput metric"""
        if duration > 0:
            throughput = operations / duration
            self.record_metric(
                f"{operation_type}_throughput",
                throughput,
                f"{operation_type}/second",
                {"operations": operations, "duration": duration}
            )
    
    def record_efficiency(self, actual: float, theoretical: float, metric_type: str = "efficiency"):
        """Record efficiency percentage"""
        if theoretical > 0:
            efficiency = (actual / theoretical) * 100
            self.record_metric(
                metric_type,
                efficiency,
                "percentage",
                {"actual": actual, "theoretical": theoretical}
            )
    
    def get_metric_summary(self) -> Dict[str, Any]:
        """Get summary of all recorded metrics"""
        if not self.metrics:
            return {"total_metrics": 0, "duration": time.time() - self.start_time}
        
        # Group metrics by name
        metric_groups = {}
        for metric in self.metrics:
            if metric.name not in metric_groups:
                metric_groups[metric.name] = []
            metric_groups[metric.name].append(metric.value)
        
        # Calculate statistics for each metric
        summary = {
            "tracker_name": self.name,
            "total_metrics": len(self.metrics),
            "total_duration": time.time() - self.start_time,
            "metric_statistics": {}
        }
        
        for name, values in metric_groups.items():
            if values:
                summary["metric_statistics"][name] = {
                    "count": len(values),
                    "min": min(values),
                    "max": max(values),
                    "avg": sum(values) / len(values),
                    "total": sum(values),
                    "unit": next((m.unit for m in self.metrics if m.name == name), "unknown")
                }
        
        return summary
    
    def export_metrics(self, output_path: str) -> str:
        """Export metrics to JSON file"""
        output_file = Path(output_path)
        
        export_data = {
            "tracker_name": self.name,
            "export_timestamp": time.time(),
            "total_duration": time.time() - self.start_time,
            "summary": self.get_metric_summary(),
            "detailed_metrics": [metric.to_dict() for metric in self.metrics]
        }
        
        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        return str(output_file)
    
    def reset(self):
        """Reset all metrics and timers"""
        self.metrics.clear()
        self.timers.clear()
        self.counters.clear()
        self.start_time = time.time()

class MetricsCollector:
    """Collects metrics from multiple sources and provides aggregated analysis"""
    
    def __init__(self):
        self.trackers: Dict[str, PerformanceTracker] = {}
        self.collection_start = time.time()
    
    def add_tracker(self, name: str, tracker: PerformanceTracker):
        """Add a performance tracker to the collection"""
        self.trackers[name] = tracker
    
    def create_tracker(self, name: str) -> PerformanceTracker:
        """Create and add a new performance tracker"""
        tracker = PerformanceTracker(name)
        self.add_tracker(name, tracker)
        return tracker
    
    def get_tracker(self, name: str) -> Optional[PerformanceTracker]:
        """Get a tracker by name"""
        return self.trackers.get(name)
    
    def aggregate_metrics(self) -> Dict[str, Any]:
        """Aggregate metrics from all trackers"""
        aggregated = {
            "collection_duration": time.time() - self.collection_start,
            "total_trackers": len(self.trackers),
            "tracker_summaries": {},
            "cross_tracker_analysis": {}
        }
        
        # Get summaries from all trackers
        all_metrics = []
        for name, tracker in self.trackers.items():
            summary = tracker.get_metric_summary()
            aggregated["tracker_summaries"][name] = summary
            all_metrics.extend(tracker.metrics)
        
        # Cross-tracker analysis
        if all_metrics:
            # Find common metrics across trackers
            metric_names = set()
            for metric in all_metrics:
                metric_names.add(metric.name)
            
            cross_analysis = {}
            for metric_name in metric_names:
                values = [m.value for m in all_metrics if m.name == metric_name]
                if values:
                    cross_analysis[metric_name] = {
                        "total_recordings": len(values),
                        "min": min(values),
                        "max": max(values),
                        "avg": sum(values) / len(values),
                        "range": max(values) - min(values)
                    }
            
            aggregated["cross_tracker_analysis"] = cross_analysis
        
        return aggregated
    
    def compare_trackers(self, tracker_names: List[str] = None) -> Dict[str, Any]:
        """Compare performance between specified trackers"""
        if tracker_names is None:
            tracker_names = list(self.trackers.keys())
        
        comparison = {
            "compared_trackers": tracker_names,
            "comparison_timestamp": time.time(),
            "tracker_comparison": {}
        }
        
        for name in tracker_names:
            if name in self.trackers:
                tracker = self.trackers[name]
                summary = tracker.get_metric_summary()
                
                comparison["tracker_comparison"][name] = {
                    "total_metrics": summary["total_metrics"],
                    "duration": summary["total_duration"],
                    "metrics_per_second": summary["total_metrics"] / max(summary["total_duration"], 0.001),
                    "key_metrics": {}
                }
                
                # Extract key performance metrics
                stats = summary.get("metric_statistics", {})
                for metric_name, metric_stats in stats.items():
                    if any(keyword in metric_name.lower() for keyword in 
                          ["throughput", "efficiency", "latency", "bandwidth"]):
                        comparison["tracker_comparison"][name]["key_metrics"][metric_name] = {
                            "avg": metric_stats["avg"],
                            "max": metric_stats["max"],
                            "unit": metric_stats["unit"]
                        }
        
        return comparison
    
    def generate_performance_report(self, output_path: str = None) -> str:
        """Generate comprehensive performance report"""
        if output_path is None:
            timestamp = int(time.time())
            output_path = f"performance_report_{timestamp}.json"
        
        report = {
            "report_metadata": {
                "generation_timestamp": time.time(),
                "collection_duration": time.time() - self.collection_start,
                "report_version": "1.0"
            },
            "aggregated_metrics": self.aggregate_metrics(),
            "tracker_comparison": self.compare_trackers(),
            "detailed_tracker_data": {}
        }
        
        # Include detailed data from each tracker
        for name, tracker in self.trackers.items():
            report["detailed_tracker_data"][name] = {
                "summary": tracker.get_metric_summary(),
                "metrics": [metric.to_dict() for metric in tracker.metrics[-100:]]  # Last 100 metrics
            }
        
        output_file = Path(output_path)
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        return str(output_file)
    
    def print_summary(self):
        """Print performance summary to console"""
        aggregated = self.aggregate_metrics()
        
        print("\n" + "="*80)
        print("📊 PERFORMANCE METRICS SUMMARY")
        print("="*80)
        
        print(f"Collection Duration: {aggregated['collection_duration']:.2f} seconds")
        print(f"Total Trackers: {aggregated['total_trackers']}")
        
        if aggregated["tracker_summaries"]:
            print(f"\n📈 Tracker Performance:")
            for name, summary in aggregated["tracker_summaries"].items():
                print(f"  {name}:")
                print(f"    Metrics recorded: {summary['total_metrics']}")
                print(f"    Duration: {summary['total_duration']:.2f}s")
                if summary['total_duration'] > 0:
                    mps = summary['total_metrics'] / summary['total_duration']
                    print(f"    Metrics/second: {mps:.2f}")
        
        if aggregated["cross_tracker_analysis"]:
            print(f"\n🔍 Cross-Tracker Analysis:")
            for metric_name, analysis in aggregated["cross_tracker_analysis"].items():
                print(f"  {metric_name}:")
                print(f"    Recordings: {analysis['total_recordings']}")
                print(f"    Range: {analysis['min']:.3f} - {analysis['max']:.3f}")
                print(f"    Average: {analysis['avg']:.3f}")
        
        print("="*80)