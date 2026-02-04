# HHC-MOVRPTW: Home Health Care Multi-Objective Vehicle Routing Problem with Time Windows
# K-NSGA-II Hybrid Implementation

__version__ = "1.0.0"
__author__ = "HHC Research Project"

from src.data_parser import Customer, ProblemInstance, load_instance, parse_solomon_instance
from src.problem import HHCProblem, Route, Solution
from src.kmeans import KMeansClustering, Cluster
from src.nsga2 import NSGAII, Individual
from src.hybrid_knsga2 import KNSGAII, KNSGAIIResult, ParetoSubset

__all__ = [
    # Data Parser
    'Customer', 'ProblemInstance', 'load_instance', 'parse_solomon_instance',
    # Problem Definition
    'HHCProblem', 'Route', 'Solution',
    # K-means Clustering
    'KMeansClustering', 'Cluster',
    # NSGA-II
    'NSGAII', 'Individual',
    # K-NSGA-II Hybrid
    'KNSGAII', 'KNSGAIIResult', 'ParetoSubset'
]