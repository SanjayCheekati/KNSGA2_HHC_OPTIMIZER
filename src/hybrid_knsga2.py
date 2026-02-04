"""
K-NSGA-II: Hybrid K-means + NSGA-II Algorithm
Combines clustering with evolutionary optimization for HHC-MOVRPTW

Based on the paper's approach:
1. Decomposition Stage: Divide patients into K clusters using K-means
2. Optimization Stage: Run NSGA-II on each cluster
3. Combination Stage: Combine Pareto subsets to form global Pareto front
"""

import numpy as np
from typing import List, Tuple, Dict
from dataclasses import dataclass
from copy import deepcopy
import time

from src.data_parser import ProblemInstance, load_instance
from src.problem import HHCProblem, Solution, Route
from src.kmeans import KMeansClustering, Cluster
from src.nsga2 import NSGAII, Individual


@dataclass
class ParetoSubset:
    """Represents a Pareto subset for one cluster/caregiver"""
    cluster_id: int
    solutions: List[Individual]
    customer_ids: List[int]
    
    @property
    def size(self) -> int:
        return len(self.solutions)
    
    def get_objective_values(self) -> List[Tuple[float, float]]:
        return [sol.objectives for sol in self.solutions]


@dataclass
class KNSGAIIResult:
    """Result container for K-NSGA-II algorithm"""
    global_pareto_front: List[Tuple[float, float]]
    pareto_subsets: List[ParetoSubset]
    clusters: List[Cluster]
    total_time: float
    decomposition_time: float
    optimization_time: float
    combination_time: float


class KNSGAII:
    """
    K-NSGA-II: Hybrid approach combining K-means clustering with NSGA-II.
    
    The algorithm works in three stages:
    
    1. DECOMPOSITION STAGE:
       - Divide patients into K clusters using K-means
       - K = number of caregivers
       - Clusters based on geographical coordinates and time preferences
    
    2. OPTIMIZATION STAGE:
       - Run NSGA-II independently on each cluster
       - Each cluster corresponds to one caregiver's route
       - Output: K Pareto subsets (one per cluster)
    
    3. COMBINATION STAGE:
       - Combine K Pareto subsets into global Pareto front
       - Sum objective values across corresponding solutions
       - Remove dominated solutions
    """
    
    def __init__(self,
                 instance: ProblemInstance,
                 population_size: int = 100,
                 max_generations: int = 1000,
                 crossover_rate: float = 0.7,
                 mutation_rate: float = 0.2,
                 use_time_features: bool = True,
                 balance_clusters: bool = True,
                 random_state: int = None):
        """
        Initialize K-NSGA-II algorithm.
        
        Args:
            instance: Problem instance
            population_size: NSGA-II population size
            max_generations: NSGA-II max generations
            crossover_rate: NSGA-II crossover rate
            mutation_rate: NSGA-II mutation rate
            use_time_features: Include time windows in clustering
            balance_clusters: Balance cluster sizes
            random_state: Random seed
        """
        self.instance = instance
        self.population_size = population_size
        self.max_generations = max_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.use_time_features = use_time_features
        self.balance_clusters = balance_clusters
        self.random_state = random_state
        
        # Number of clusters = number of caregivers
        self.n_clusters = instance.num_vehicles
        
        # Results
        self.clusters: List[Cluster] = None
        self.pareto_subsets: List[ParetoSubset] = None
        self.global_pareto_front: List[Tuple[float, float]] = None
    
    def _decomposition_stage(self, verbose: bool = True) -> List[Cluster]:
        """
        Stage 1: Decompose problem using K-means clustering.
        """
        if verbose:
            print("\n" + "=" * 60)
            print("STAGE 1: DECOMPOSITION (K-means Clustering)")
            print("=" * 60)
        
        kmeans = KMeansClustering(
            n_clusters=self.n_clusters,
            max_iterations=100,
            use_time_features=self.use_time_features,
            random_state=self.random_state
        )
        
        clusters = kmeans.fit(self.instance)
        
        if self.balance_clusters:
            clusters = kmeans.get_balanced_clusters(self.instance)
        
        if verbose:
            print(f"\nClusters created: {len(clusters)}")
            for cluster in clusters:
                customer_ids = kmeans.get_cluster_customer_ids(cluster.id)
                print(f"  Cluster {cluster.id}: {cluster.size} customers - IDs: {customer_ids}")
        
        self.clusters = clusters
        self.kmeans = kmeans
        return clusters
    
    def _optimization_stage(self, verbose: bool = True) -> List[ParetoSubset]:
        """
        Stage 2: Run NSGA-II on each cluster.
        """
        if verbose:
            print("\n" + "=" * 60)
            print("STAGE 2: OPTIMIZATION (NSGA-II per cluster)")
            print("=" * 60)
        
        pareto_subsets = []
        
        for cluster in self.clusters:
            if verbose:
                print(f"\n--- Optimizing Cluster {cluster.id} ({cluster.size} customers) ---")
            
            # Get customer IDs for this cluster
            customer_ids = self.kmeans.get_cluster_customer_ids(cluster.id)
            
            if len(customer_ids) == 0:
                if verbose:
                    print("  Empty cluster, skipping...")
                continue
            
            # Create sub-problem for this cluster
            sub_problem = self._create_sub_problem(customer_ids)
            
            # Run NSGA-II on sub-problem
            nsga2 = NSGAII(
                problem=sub_problem,
                population_size=self.population_size,
                max_generations=self.max_generations,
                crossover_rate=self.crossover_rate,
                mutation_rate=self.mutation_rate,
                random_state=self.random_state + cluster.id if self.random_state else None
            )
            
            # Set customer subset
            nsga2.set_customer_subset(customer_ids)
            
            # Run optimization
            pareto_front = nsga2.run(verbose=False)
            
            # Create Pareto subset
            pareto_subset = ParetoSubset(
                cluster_id=cluster.id,
                solutions=pareto_front,
                customer_ids=customer_ids
            )
            pareto_subsets.append(pareto_subset)
            
            if verbose:
                print(f"  Pareto front size: {len(pareto_front)}")
                if pareto_front:
                    best_f1 = min(s.objectives[0] for s in pareto_front)
                    best_f2 = min(s.objectives[1] for s in pareto_front)
                    print(f"  Best F1: {best_f1:.2f}, Best F2: {best_f2:.2f}")
        
        self.pareto_subsets = pareto_subsets
        return pareto_subsets
    
    def _create_sub_problem(self, customer_ids: List[int]) -> HHCProblem:
        """Create a sub-problem for a specific cluster of customers."""
        # Create a modified problem with only the cluster's customers
        # The sub-problem uses 1 caregiver
        
        sub_problem = HHCProblem(
            self.instance,
            max_workload=self.instance.depot.due_date
        )
        
        # Override number of caregivers for sub-problem
        sub_problem.num_caregivers = 1
        sub_problem.num_customers = len(customer_ids)
        sub_problem.max_patients_per_caregiver = len(customer_ids)
        
        return sub_problem
    
    def _combination_stage(self, verbose: bool = True) -> List[Tuple[float, float]]:
        """
        Stage 3: Combine Pareto subsets into global Pareto front.
        
        Algorithm:
        1. Generate diverse combinations from all Pareto subsets
        2. Sum objective values across corresponding solutions
        3. Remove dominated solutions
        
        Enhanced with multiple combination strategies for better coverage.
        """
        if verbose:
            print("\n" + "=" * 60)
            print("STAGE 3: COMBINATION (Global Pareto Front)")
            print("=" * 60)
        
        if not self.pareto_subsets:
            return []
        
        # Generate many combinations
        global_solutions = []
        
        # Collect non-empty subsets
        valid_subsets = [ps for ps in self.pareto_subsets if ps.size > 0]
        
        if not valid_subsets:
            return []
        
        # Sort each Pareto subset by F1
        sorted_by_f1 = [sorted(ps.solutions, key=lambda s: s.objectives[0]) for ps in valid_subsets]
        
        # Sort each by F2
        sorted_by_f2 = [sorted(ps.solutions, key=lambda s: s.objectives[1]) for ps in valid_subsets]
        
        # Find minimum subset size
        T = min(len(ss) for ss in sorted_by_f1)
        
        if verbose:
            print(f"\nMinimum Pareto subset size (T): {T}")
            print(f"Number of Pareto subsets: {len(valid_subsets)}")
        
        # Method 1: Combine at same positions (sorted by F1)
        for t in range(T):
            total_f1 = sum(ss[t].objectives[0] for ss in sorted_by_f1)
            total_f2 = sum(ss[t].objectives[1] for ss in sorted_by_f1)
            global_solutions.append((total_f1, total_f2))
        
        # Method 2: Combine at same positions (sorted by F2)
        for t in range(T):
            total_f1 = sum(ss[t].objectives[0] for ss in sorted_by_f2)
            total_f2 = sum(ss[t].objectives[1] for ss in sorted_by_f2)
            global_solutions.append((total_f1, total_f2))
        
        # Method 3: Extreme combinations
        # Best F1 overall
        best_f1_total = sum(min(s.objectives[0] for s in ss) for ss in sorted_by_f1)
        best_f2_for_best_f1 = sum(
            min(ss, key=lambda s: s.objectives[0]).objectives[1] 
            for ss in sorted_by_f1
        )
        global_solutions.append((best_f1_total, best_f2_for_best_f1))
        
        # Best F2 overall
        best_f2_total = sum(min(s.objectives[1] for s in ss) for ss in sorted_by_f2)
        best_f1_for_best_f2 = sum(
            min(ss, key=lambda s: s.objectives[1]).objectives[0] 
            for ss in sorted_by_f2
        )
        global_solutions.append((best_f1_for_best_f2, best_f2_total))
        
        # Method 4: Weighted combinations along Pareto front (more granular)
        for alpha in np.linspace(0, 1, 50):  # 50 weight combinations
            total_f1 = 0.0
            total_f2 = 0.0
            for ss in sorted_by_f1:
                # Normalize objectives within this subset
                f1_vals = [s.objectives[0] for s in ss]
                f2_vals = [s.objectives[1] for s in ss]
                f1_min, f1_max = min(f1_vals), max(f1_vals)
                f2_min, f2_max = min(f2_vals), max(f2_vals)
                f1_range = f1_max - f1_min if f1_max != f1_min else 1
                f2_range = f2_max - f2_min if f2_max != f2_min else 1
                
                # Select solution minimizing weighted sum
                best_sol = min(ss, key=lambda s: 
                    alpha * (s.objectives[0] - f1_min) / f1_range + 
                    (1-alpha) * (s.objectives[1] - f2_min) / f2_range
                )
                total_f1 += best_sol.objectives[0]
                total_f2 += best_sol.objectives[1]
            global_solutions.append((total_f1, total_f2))
        
        # Method 5: Random sampling with more samples
        n_samples = max(200, T * len(valid_subsets) * 5)
        for _ in range(n_samples):
            total_f1 = 0.0
            total_f2 = 0.0
            for ss in sorted_by_f1:
                idx = np.random.randint(0, len(ss))
                total_f1 += ss[idx].objectives[0]
                total_f2 += ss[idx].objectives[1]
            global_solutions.append((total_f1, total_f2))
        
        # Method 6: All combinations of extremes
        # Generate all 2^k combinations of best-F1 vs best-F2 from each cluster
        from itertools import product
        extreme_choices = []
        for ss in sorted_by_f1:
            best_f1_sol = min(ss, key=lambda s: s.objectives[0])
            best_f2_sol = min(ss, key=lambda s: s.objectives[1])
            extreme_choices.append([best_f1_sol, best_f2_sol])
        
        # Limit combinations if too many
        if len(valid_subsets) <= 8:  # 2^8 = 256 combinations
            for combo in product(*extreme_choices):
                total_f1 = sum(s.objectives[0] for s in combo)
                total_f2 = sum(s.objectives[1] for s in combo)
                global_solutions.append((total_f1, total_f2))
        
        # Method 7: Cross-combinations at different positions
        for offset in range(T):
            total_f1 = 0.0
            total_f2 = 0.0
            for i, ss in enumerate(sorted_by_f1):
                idx = (offset + i) % len(ss)
                total_f1 += ss[idx].objectives[0]
                total_f2 += ss[idx].objectives[1]
            global_solutions.append((total_f1, total_f2))
        
        # Remove duplicates (round to avoid floating point issues)
        unique_solutions = list(set((round(f1, 4), round(f2, 4)) for f1, f2 in global_solutions))
        
        # Remove dominated solutions
        non_dominated = self._get_non_dominated(unique_solutions)
        
        if verbose:
            print(f"\nTotal combinations generated: {len(global_solutions)}")
            print(f"Unique solutions: {len(unique_solutions)}")
            print(f"Non-dominated solutions: {len(non_dominated)}")
        
        self.global_pareto_front = non_dominated
        return non_dominated
    
    def _get_non_dominated(self, solutions: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Get non-dominated solutions from a list of (F1, F2) tuples."""
        non_dominated = []
        
        for sol in solutions:
            is_dominated = False
            
            for other in solutions:
                if other != sol:
                    # Check if other dominates sol
                    if (other[0] <= sol[0] and other[1] <= sol[1] and
                        (other[0] < sol[0] or other[1] < sol[1])):
                        is_dominated = True
                        break
            
            if not is_dominated:
                non_dominated.append(sol)
        
        return non_dominated
    
    def run(self, verbose: bool = True) -> KNSGAIIResult:
        """
        Run the complete K-NSGA-II algorithm.
        
        Returns:
            KNSGAIIResult containing all results and timing information
        """
        start_time = time.time()
        
        if verbose:
            print("\n" + "=" * 60)
            print("K-NSGA-II: Hybrid K-means + NSGA-II Algorithm")
            print("=" * 60)
            print(f"\nInstance: {self.instance.name}")
            print(f"Customers: {self.instance.num_customers}")
            print(f"Caregivers/Clusters: {self.n_clusters}")
            print(f"Population size: {self.population_size}")
            print(f"Max generations: {self.max_generations}")
        
        # Stage 1: Decomposition
        decomp_start = time.time()
        self._decomposition_stage(verbose)
        decomp_time = time.time() - decomp_start
        
        # Stage 2: Optimization
        opt_start = time.time()
        self._optimization_stage(verbose)
        opt_time = time.time() - opt_start
        
        # Stage 3: Combination
        comb_start = time.time()
        self._combination_stage(verbose)
        comb_time = time.time() - comb_start
        
        total_time = time.time() - start_time
        
        if verbose:
            print("\n" + "=" * 60)
            print("K-NSGA-II RESULTS")
            print("=" * 60)
            print(f"\nGlobal Pareto Front ({len(self.global_pareto_front)} solutions):")
            for i, (f1, f2) in enumerate(sorted(self.global_pareto_front)):
                print(f"  {i+1}. F1={f1:.2f}, F2={f2:.2f}")
            
            print(f"\nTiming:")
            print(f"  Decomposition: {decomp_time:.2f}s")
            print(f"  Optimization:  {opt_time:.2f}s")
            print(f"  Combination:   {comb_time:.2f}s")
            print(f"  Total:         {total_time:.2f}s")
        
        return KNSGAIIResult(
            global_pareto_front=self.global_pareto_front,
            pareto_subsets=self.pareto_subsets,
            clusters=self.clusters,
            total_time=total_time,
            decomposition_time=decomp_time,
            optimization_time=opt_time,
            combination_time=comb_time
        )
    
    def get_performance_metrics(self) -> Dict:
        """Calculate performance metrics for the result."""
        if not self.global_pareto_front:
            return {}
        
        # Pareto front size
        pf_size = len(self.global_pareto_front)
        
        # Spacing metric (SP) - normalized
        sp = self._calculate_spacing()
        
        # Hypervolume (normalized to [0,1]) - as per paper
        hv = self._calculate_hypervolume_normalized()
        
        return {
            'pareto_size': pf_size,
            'spacing': sp,
            'hypervolume': hv,
            'best_f1': min(s[0] for s in self.global_pareto_front),
            'best_f2': min(s[1] for s in self.global_pareto_front)
        }
    
    def _calculate_spacing(self) -> float:
        """Calculate spacing metric (SP) for diversity assessment."""
        if len(self.global_pareto_front) <= 1:
            return 0.0
        
        # Normalize objectives
        f1_vals = [s[0] for s in self.global_pareto_front]
        f2_vals = [s[1] for s in self.global_pareto_front]
        
        f1_min, f1_max = min(f1_vals), max(f1_vals)
        f2_min, f2_max = min(f2_vals), max(f2_vals)
        
        f1_range = f1_max - f1_min if f1_max != f1_min else 1
        f2_range = f2_max - f2_min if f2_max != f2_min else 1
        
        normalized = [((s[0] - f1_min) / f1_range, (s[1] - f2_min) / f2_range)
                      for s in self.global_pareto_front]
        
        # Calculate minimum distances
        distances = []
        for i, sol in enumerate(normalized):
            min_dist = float('inf')
            for j, other in enumerate(normalized):
                if i != j:
                    dist = np.sqrt((sol[0] - other[0])**2 + (sol[1] - other[1])**2)
                    min_dist = min(min_dist, dist)
            distances.append(min_dist)
        
        # Calculate spacing
        mean_dist = np.mean(distances)
        sp = np.sqrt(sum((d - mean_dist)**2 for d in distances) / len(distances))
        
        return sp
    
    def _calculate_hypervolume_normalized(self) -> float:
        """
        Calculate normalized hypervolume indicator as per the paper.
        
        The paper methodology:
        1. Estimate theoretical bounds for F1 and F2 from the problem instance
        2. Normalize all Pareto front points to [0, 1]
        3. Use reference point (1, 1) representing worst case
        4. Calculate hypervolume as area dominated by the Pareto front
        
        Key insight: The paper likely uses consistent bounds derived from
        the problem's theoretical worst-case values, not dynamic bounds.
        """
        if not self.global_pareto_front or len(self.global_pareto_front) < 1:
            return 0.0
        
        # Get objective values from current Pareto front
        f1_vals = [s[0] for s in self.global_pareto_front]
        f2_vals = [s[1] for s in self.global_pareto_front]
        
        # Calculate theoretical bounds based on problem structure
        # These should be consistent across all runs for fair comparison
        
        # F1 (total service time = travel + service):
        # Theoretical minimum: sum of all service times + minimal possible travel
        total_service_time = sum(c.service_time for c in self.instance.customers)
        
        # Estimate minimum possible travel time using nearest neighbor heuristic idea
        # Best case: very short paths between clustered customers
        avg_coord = np.mean([[c.x, c.y] for c in self.instance.customers], axis=0)
        depot_dist = np.sqrt(self.instance.depot.x**2 + self.instance.depot.y**2)
        
        # F1 theoretical minimum: must do all service + some travel
        f1_theoretical_min = total_service_time * 0.8  # Allow some margin
        
        # F1 theoretical maximum: worst case routing
        # Estimate as visiting all in worst order
        max_possible_travel = 0
        for c in self.instance.customers:
            dist_from_depot = np.sqrt((c.x - self.instance.depot.x)**2 + 
                                       (c.y - self.instance.depot.y)**2)
            max_possible_travel += dist_from_depot * 2  # worst case: depot to each customer and back
        f1_theoretical_max = total_service_time + max_possible_travel
        
        # F2 (tardiness):
        # Theoretical minimum: 0 (all customers served in their time windows)
        f2_theoretical_min = 0.0
        
        # Theoretical maximum: sum of maximum possible tardiness for each customer
        f2_theoretical_max = 0.0
        planning_horizon = self.instance.depot.due_date
        for c in self.instance.customers:
            # Max tardiness = if served at end of horizon
            max_late = max(0, planning_horizon - c.due_date)
            max_early = c.ready_time  # If served at time 0
            f2_theoretical_max += max_late + max_early
        f2_theoretical_max = max(f2_theoretical_max, 1.0)  # Ensure non-zero
        
        # Use observed values to adjust bounds (take wider range)
        f1_min = min(f1_theoretical_min, min(f1_vals) * 0.9)
        f1_max = max(f1_theoretical_max, max(f1_vals) * 1.1)
        f2_min = f2_theoretical_min
        f2_max = max(f2_theoretical_max, max(f2_vals) * 1.1) if max(f2_vals) > 0 else f2_theoretical_max
        
        # Ensure ranges are valid
        f1_range = f1_max - f1_min if f1_max > f1_min else 1.0
        f2_range = f2_max - f2_min if f2_max > f2_min else 1.0
        
        # Normalize to [0, 1] using global bounds
        normalized = []
        for s in self.global_pareto_front:
            norm_f1 = (s[0] - f1_min) / f1_range
            norm_f2 = (s[1] - f2_min) / f2_range
            # Clamp to [0, 1]
            norm_f1 = max(0.0, min(1.0, norm_f1))
            norm_f2 = max(0.0, min(1.0, norm_f2))
            normalized.append((norm_f1, norm_f2))
        
        # Reference point is (1, 1) in normalized space
        ref_point = (1.0, 1.0)
        
        # Sort by first objective (ascending), then by second (descending)
        sorted_front = sorted(normalized, key=lambda x: (x[0], -x[1]))
        
        # Remove dominated points in normalized space
        non_dominated = []
        for p in sorted_front:
            if not non_dominated or p[1] < non_dominated[-1][1]:
                non_dominated.append(p)
        
        # Calculate hypervolume using standard 2D algorithm
        # Compute area dominated by the Pareto front (area between front and reference)
        hv = 0.0
        prev_f1 = 0.0
        
        for (f1, f2) in non_dominated:
            if f1 < ref_point[0] and f2 < ref_point[1]:
                # Add rectangle: width from prev_f1 to f1, height from f2 to ref_point[1]
                width = f1 - prev_f1
                height = ref_point[1] - f2
                hv += width * height
                prev_f1 = f1
        
        # Add final rectangle from last point to reference point
        if non_dominated:
            last_f1, last_f2 = non_dominated[-1]
            if last_f1 < ref_point[0] and last_f2 < ref_point[1]:
                hv += (ref_point[0] - last_f1) * (ref_point[1] - last_f2)
        
        return hv
    
    def _calculate_hypervolume(self, ref_point: Tuple[float, float] = None) -> float:
        """Calculate hypervolume indicator for convergence assessment."""
        if not self.global_pareto_front:
            return 0.0
        
        # Set reference point as worst values + margin
        if ref_point is None:
            max_f1 = max(s[0] for s in self.global_pareto_front)
            max_f2 = max(s[1] for s in self.global_pareto_front)
            ref_point = (max_f1 * 1.1, max_f2 * 1.1)
        
        # Sort by first objective
        sorted_front = sorted(self.global_pareto_front, key=lambda x: x[0])
        
        # Calculate hypervolume using simple rectangle method
        hv = 0.0
        prev_f1 = 0.0
        
        for f1, f2 in sorted_front:
            if f1 < ref_point[0] and f2 < ref_point[1]:
                width = f1 - prev_f1
                height = ref_point[1] - f2
                hv += width * height
                prev_f1 = f1
        
        # Add final rectangle
        if sorted_front:
            hv += (ref_point[0] - sorted_front[-1][0]) * (ref_point[1] - sorted_front[-1][1])
        
        return hv


if __name__ == "__main__":
    # Test K-NSGA-II
    print("Testing K-NSGA-II Hybrid Algorithm")
    print("=" * 60)
    
    # Load test instance
    instance = load_instance("C101.25")
    
    # Run K-NSGA-II with reduced parameters for testing
    knsga2 = KNSGAII(
        instance=instance,
        population_size=50,
        max_generations=100,
        crossover_rate=0.7,
        mutation_rate=0.2,
        use_time_features=True,
        balance_clusters=True,
        random_state=42
    )
    
    result = knsga2.run(verbose=True)
    
    # Calculate metrics
    metrics = knsga2.get_performance_metrics()
    print("\nPerformance Metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
