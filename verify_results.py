"""Quick test to verify paper-matching results"""
from src.data_parser import load_instance
from src.hybrid_knsga2 import KNSGAII
import time

# Paper Table 5 targets (K-NSGA-II column)
PAPER_TARGETS = {
    'C101.25': 0.905,
    'C101.100': 0.810,
    'C107.100': 0.815,
    'C206.50': 0.865,
    'RC106.50': 0.802
}

print("=" * 70)
print("VERIFICATION: K-NSGA-II Results vs Paper Targets")
print("=" * 70)
print("Parameters: population=50, generations=100 (fast mode)")
print("-" * 70)

total_start = time.time()

for inst, target in PAPER_TARGETS.items():
    start = time.time()
    instance = load_instance(inst)
    knsga2 = KNSGAII(
        instance=instance,
        population_size=50,
        max_generations=100,
        crossover_rate=0.7,
        mutation_rate=0.2,
        random_state=42
    )
    knsga2.run(verbose=False)
    m = knsga2.get_performance_metrics()
    elapsed = time.time() - start
    
    hv = m['hypervolume']
    sp = m['spacing']
    
    print(f"{inst:<12} Hv={hv:.4f}  SP={sp:.4f}  Time={elapsed:.1f}s")

print("-" * 70)
print(f"Total time: {time.time() - total_start:.1f}s")
print("Done!")
