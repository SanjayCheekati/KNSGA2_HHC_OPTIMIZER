"""
Quick Demo Script
=================
Fast demonstration of K-NSGA-II on a single instance.
"""

import sys
import os
import time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_parser import load_instance
from src.hybrid_knsga2 import KNSGAII


def quick_demo(instance_name: str = "C101.25"):
    """Run a quick demonstration of K-NSGA-II."""
    
    print("=" * 65)
    print(f"K-NSGA-II Quick Demo: {instance_name}")
    print("=" * 65)
    
    # Load instance
    instance = load_instance(instance_name)
    print(f"\nProblem: {instance.num_customers} customers, {instance.num_vehicles} vehicles")
    
    # Run optimization
    print("\nRunning optimization...")
    start = time.time()
    
    knsga2 = KNSGAII(
        instance=instance,
        population_size=50,
        max_generations=100,
        crossover_rate=0.7,
        mutation_rate=0.2,
        random_state=42
    )
    
    knsga2.run(verbose=True)
    elapsed = time.time() - start
    
    # Results
    metrics = knsga2.get_performance_metrics()
    
    print("\n" + "=" * 65)
    print("OPTIMIZATION RESULTS")
    print("=" * 65)
    print(f"  Hypervolume:     {metrics['hypervolume']:.4f}")
    print(f"  Spacing:         {metrics['spacing']:.4f}")
    print(f"  Pareto Size:     {metrics['pareto_size']} solutions")
    print(f"  Execution Time:  {elapsed:.2f} seconds")
    print("=" * 65)


if __name__ == "__main__":
    instance = sys.argv[1] if len(sys.argv) > 1 else "C101.25"
    quick_demo(instance)
