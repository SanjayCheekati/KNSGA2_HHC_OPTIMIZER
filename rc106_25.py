"""
DEMO SCRIPT for K-NSGA-II Algorithm - RC106.25 Instance
=========================================================
Fifth and Final instance from Paper Table 5

RC106 has Mixed (Random+Clustered) customer distribution
This tests the algorithm's ability to handle hybrid patterns
"""

import sys
import os
import time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_parser import load_instance
from src.hybrid_knsga2 import KNSGAII


def run_rc106_25_demo():
    print("=" * 60)
    print("K-NSGA-II - RC106.25 Instance (Fifth Instance)")
    print("=" * 60)
    print("\nInstance Characteristics:")
    print("  - Type: Mixed (RC)")
    print("  - Variant: 106 (random-clustered distribution)")
    print("  - Customers: 25")
    
    # Parameters
    POPULATION_SIZE = 50
    MAX_GENERATIONS = 100
    CROSSOVER_RATE = 0.7
    MUTATION_RATE = 0.2
    
    print(f"\nParameters:")
    print(f"  Population: {POPULATION_SIZE}")
    print(f"  Generations: {MAX_GENERATIONS}")
    print(f"  Crossover: {CROSSOVER_RATE}")
    print(f"  Mutation: {MUTATION_RATE}")
    
    # Load instance
    print("\n" + "-" * 60)
    print("Loading RC106.25 instance...")
    instance = load_instance("RC106.25")
    print(f"  Customers: {instance.num_customers}")
    print(f"  Caregivers: {instance.num_vehicles}")
    
    # Run algorithm
    print("\n" + "-" * 60)
    print("Running K-NSGA-II Algorithm...")
    print("-" * 60)
    
    start_time = time.time()
    
    knsga2 = KNSGAII(
        instance=instance,
        population_size=POPULATION_SIZE,
        max_generations=MAX_GENERATIONS,
        crossover_rate=CROSSOVER_RATE,
        mutation_rate=MUTATION_RATE,
        random_state=42
    )
    
    result = knsga2.run(verbose=True)
    
    total_time = time.time() - start_time
    
    # Get metrics
    metrics = knsga2.get_performance_metrics()
    
    # Display results
    print("\n" + "=" * 60)
    print("RC106.25 RESULTS")
    print("=" * 60)
    
    if metrics:
        hv = metrics['hypervolume']
        sp = metrics['spacing']
        
        print(f"\n{'Metric':<20} {'Result':<15}")
        print("-" * 40)
        print(f"{'Hypervolume (Hv)':<20} {hv:<15.4f}")
        print(f"{'Spacing (SP)':<20} {sp:<15.4f}")
        print(f"\n{'Pareto Front Size:':<20} {metrics['pareto_size']}")
        print(f"{'Best F1 (Service):':<20} {metrics['best_f1']:.2f}")
        print(f"{'Best F2 (Tardiness):':<20} {metrics['best_f2']:.2f}")
    
    print(f"\n{'Execution Time:':<20} {total_time:.2f} seconds")
    
    print("\n" + "=" * 60)
    print("RC106.25 Instance Completed!")
    print("=" * 60)
    
    return metrics


if __name__ == "__main__":
    run_rc106_25_demo()
