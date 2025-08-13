"""GNN (Graph Neural Network) workload modules"""
from .graph_engine import UnifiedGNNEngine
from .access_patterns import AccessPattern, GraphFormat, ExecutionMode
from .cugraph_integration import CuGraphInspiredGNN
from .performance_analyzer import GNNPerformanceAnalyzer

__all__ = [
    'UnifiedGNNEngine',
    'AccessPattern', 'GraphFormat', 'ExecutionMode',
    'CuGraphInspiredGNN',
    'GNNPerformanceAnalyzer'
]