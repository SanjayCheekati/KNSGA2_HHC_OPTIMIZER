"""
K-NSGA-II: Hybrid K-means + NSGA-II Algorithm
===============================================
For solving HHC-MOVRPTW (Home Health Care Multi-Objective VRP with Time Windows)

Three-stage hybrid optimization approach:
1. Decomposition (K-means clustering)
2. Optimization (NSGA-II per cluster)
3. Combination (Merge Pareto fronts)

This decomposition-based method reduces computational complexity by partitioning
the problem into smaller subproblems that can be solved independently.
"""

import random
import math
import time
from typing import List, Tuple, Optional, Dict
from .problem import HHCInstance, Customer, Solution
from .kmeans import KMeans
from .nsga2 import NSGA2


class KNSGAII:
    """
    K-NSGA-II: Hybrid algorithm combining K-means clustering with NSGA-II
    
    This algorithm uses a decomposition-based approach where:
    - Stage 1 (Decomposition): K-means clusters customers geographically
    - Stage 2 (Optimization): NSGA-II optimizes each cluster independently
    - Stage 3 (Combination): Cluster Pareto fronts merge into global front
    
    The hybrid approach offers several advantages:
    - Reduced search space per subproblem
    - Faster convergence
    - Natural parallelization potential
    """
    
    def __init__(
        self,
        instance: HHCInstance,
        population_size: int = 100,
        max_generations: int = 1000,
        crossover_rate: float = 0.9,
        mutation_rate: float = 0.1,
        random_state: Optional[int] = None
    ):
        """
        Initialize K-NSGA-II
        
        Args:
            instance: HHC problem instance
            population_size: Population size for NSGA-II
            max_generations: Max generations for NSGA-II
            crossover_rate: Crossover probability
            mutation_rate: Mutation probability
            random_state: Random seed for reproducibility
        """
        self.instance = instance
        self.population_size = population_size
        self.max_generations = max_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.random_state = random_state
        
        if random_state is not None:
            random.seed(random_state)
        
        # Results storage
        self.clusters: List[List[Customer]] = []
        self.cluster_pareto_fronts: List[List[Solution]] = []
        self.global_pareto_front: List[Solution] = []
        
        # Timing
        self.decomposition_time: float = 0.0
        self.optimization_time: float = 0.0
        self.combination_time: float = 0.0
    
    def _decomposition_stage(self, verbose: bool = False) -> List[List[Customer]]:
        """
        Stage 1: Decomposition using K-means clustering
        
        Clusters customers based on geographic location.
        Number of clusters equals number of caregivers.
        """
        start_time = time.time()
        
        k = self.instance.num_vehicles  # Number of clusters = number of caregivers
        
        if verbose:
            print("\n" + "=" * 60)
            print("STAGE 1: DECOMPOSITION (K-means Clustering)")
            print("=" * 60)
        
        # Run K-means
        kmeans = KMeans(k=k, max_iterations=100, random_state=self.random_state)
        kmeans.fit(self.instance.customers)
        
        self.clusters = kmeans.get_clusters()
        
        if verbose:
            print(f"\nClusters created: {len(self.clusters)}")
            for i, cluster in enumerate(self.clusters):
                ids = [c.id for c in cluster]
                print(f"  Cluster {i}: {len(cluster)} customers - IDs: {ids}")
        
        self.decomposition_time = time.time() - start_time
        
        return self.clusters
    
    def _optimization_stage(self, verbose: bool = False) -> List[List[Solution]]:
        """
        Stage 2: Optimization using NSGA-II on each cluster
        
        Runs NSGA-II independently on each cluster to find
        Pareto-optimal solutions for that sub-problem.
        """
        start_time = time.time()
        
        if verbose:
            print("\n" + "=" * 60)
            print("STAGE 2: OPTIMIZATION (NSGA-II per cluster)")
            print("=" * 60)
        
        self.cluster_pareto_fronts = []
        
        for i, cluster in enumerate(self.clusters):
            if verbose:
                print(f"\n--- Optimizing Cluster {i} ({len(cluster)} customers) ---")
            
            if not cluster:
                self.cluster_pareto_fronts.append([])
                continue
            
            # Create NSGA-II instance for this cluster
            nsga2 = NSGA2(
                instance=self.instance,
                customers=cluster,
                population_size=self.population_size,
                max_generations=self.max_generations,
                crossover_rate=self.crossover_rate,
                mutation_rate=self.mutation_rate,
                random_state=self.random_state + i if self.random_state else None
            )
            
            # Run optimization
            pareto_front = nsga2.run(verbose=False)
            self.cluster_pareto_fronts.append(pareto_front)
            
            if verbose:
                print(f"  Pareto front size: {len(pareto_front)}")
                if pareto_front:
                    best_f1 = min(s.f1 for s in pareto_front)
                    best_f2 = min(s.f2 for s in pareto_front)
                    print(f"  Best F1: {best_f1:.2f}, Best F2: {best_f2:.2f}")
        
        self.optimization_time = time.time() - start_time
        
        return self.cluster_pareto_fronts
    
    def _combination_stage(self, verbose: bool = False) -> List[Solution]:
        """
        Stage 3: Combination of cluster Pareto fronts
        
        Combines solutions from all cluster Pareto fronts
        to form the global Pareto front.
        
        Uses Cartesian product approach to generate combined solutions.
        """
        start_time = time.time()
        
        if verbose:
            print("\n" + "=" * 60)
            print("STAGE 3: COMBINATION (Global Pareto Front)")
            print("=" * 60)
        
        # Get minimum Pareto front size (T in the paper)
        non_empty_fronts = [f for f in self.cluster_pareto_fronts if f]
        if not non_empty_fronts:
            self.global_pareto_front = []
            self.combination_time = time.time() - start_time
            return self.global_pareto_front
        
        T = min(len(f) for f in non_empty_fronts)
        
        if verbose:
            print(f"\nMinimum Pareto subset size (T): {T}")
            print(f"Number of Pareto subsets: {len(non_empty_fronts)}")
        
        # Generate combined solutions
        combined_solutions = []
        
        # Take top T solutions from each front (sorted by crowding distance)
        pareto_subsets = []
        for front in non_empty_fronts:
            # Sort by a combination of objectives for diversity
            sorted_front = sorted(front, key=lambda s: (s.rank, -s.crowding_distance))[:T]
            pareto_subsets.append(sorted_front)
        
        # Generate combinations (Cartesian product approach)
        # To avoid exponential explosion, we use sampling
        max_combinations = 5000
        num_subsets = len(pareto_subsets)
        
        if num_subsets == 0:
            self.global_pareto_front = []
            self.combination_time = time.time() - start_time
            return self.global_pareto_front
        
        # Generate combinations
        for _ in range(max_combinations):
            # Pick one solution from each Pareto subset
            selected = [random.choice(subset) for subset in pareto_subsets]
            
            # Combine into a global solution
            combined = self._merge_solutions(selected)
            if combined:
                combined_solutions.append(combined)
        
        if verbose:
            print(f"\nTotal combinations generated: {len(combined_solutions)}")
        
        # Remove duplicates and find non-dominated solutions
        unique_solutions = self._remove_duplicates(combined_solutions)
        
        if verbose:
            print(f"Unique solutions: {len(unique_solutions)}")
        
        # Extract global Pareto front (non-dominated solutions)
        self.global_pareto_front = self._extract_pareto_front(unique_solutions)
        
        if verbose:
            print(f"Non-dominated solutions: {len(self.global_pareto_front)}")
        
        self.combination_time = time.time() - start_time
        
        return self.global_pareto_front
    
    def _merge_solutions(self, solutions: List[Solution]) -> Optional[Solution]:
        """Merge solutions from different clusters into one global solution"""
        merged = Solution(self.instance)
        merged.routes = []
        total_f1 = 0.0
        total_f2 = 0.0
        
        for sol in solutions:
            for route in sol.routes:
                if route:
                    merged.routes.append(route.copy())
            total_f1 += sol.f1
            total_f2 += sol.f2
        
        merged.f1 = total_f1
        merged.f2 = total_f2
        
        return merged
    
    def _remove_duplicates(self, solutions: List[Solution]) -> List[Solution]:
        """Remove duplicate solutions based on objective values"""
        unique = []
        seen = set()
        
        for sol in solutions:
            key = (round(sol.f1, 2), round(sol.f2, 2))
            if key not in seen:
                seen.add(key)
                unique.append(sol)
        
        return unique
    
    def _extract_pareto_front(self, solutions: List[Solution]) -> List[Solution]:
        """Extract non-dominated solutions (Pareto front)"""
        if not solutions:
            return []
        
        pareto_front = []
        
        for sol in solutions:
            is_dominated = False
            to_remove = []
            
            for pf_sol in pareto_front:
                if pf_sol.dominates(sol):
                    is_dominated = True
                    break
                elif sol.dominates(pf_sol):
                    to_remove.append(pf_sol)
            
            if not is_dominated:
                pareto_front = [s for s in pareto_front if s not in to_remove]
                pareto_front.append(sol)
        
        # Sort by F1 for consistent output
        pareto_front.sort(key=lambda s: s.f1)
        
        return pareto_front
    
    def run(self, verbose: bool = True) -> List[Solution]:
        """
        Run the complete K-NSGA-II algorithm
        
        Args:
            verbose: Print progress information
        
        Returns:
            Global Pareto front
        """
        if verbose:
            print("\n" + "=" * 60)
            print("K-NSGA-II: Hybrid K-means + NSGA-II Algorithm")
            print("=" * 60)
            print(f"\nInstance: {self.instance.name}")
            print(f"Customers: {self.instance.num_customers}")
            print(f"Caregivers/Clusters: {self.instance.num_vehicles}")
            print(f"Population size: {self.population_size}")
            print(f"Max generations: {self.max_generations}")
        
        # Stage 1: Decomposition
        self._decomposition_stage(verbose)
        
        # Stage 2: Optimization
        self._optimization_stage(verbose)
        
        # Stage 3: Combination
        self._combination_stage(verbose)
        
        # Print results
        if verbose:
            self._print_results()
        
        return self.global_pareto_front
    
    def _print_results(self):
        """Print final results"""
        print("\n" + "=" * 60)
        print("K-NSGA-II RESULTS")
        print("=" * 60)
        
        print(f"\nGlobal Pareto Front ({len(self.global_pareto_front)} solutions):")
        for i, sol in enumerate(self.global_pareto_front[:20]):  # Show first 20
            print(f"  {i+1}. F1={sol.f1:.2f}, F2={sol.f2:.2f}")
        
        if len(self.global_pareto_front) > 20:
            print(f"  ... and {len(self.global_pareto_front) - 20} more solutions")
        
        total_time = self.decomposition_time + self.optimization_time + self.combination_time
        print(f"\nTiming:")
        print(f"  Decomposition: {self.decomposition_time:.2f}s")
        print(f"  Optimization:  {self.optimization_time:.2f}s")
        print(f"  Combination:   {self.combination_time:.2f}s")
        print(f"  Total:         {total_time:.2f}s")
    
    def get_pareto_front(self) -> List[Solution]:
        """Return the global Pareto front"""
        return self.global_pareto_front
    
    def get_performance_metrics(self) -> Dict:
        """
        Calculate performance metrics for the Pareto front
        
        Returns:
            Dictionary with hypervolume, spacing, and other metrics
        """
        if not self.global_pareto_front:
            return {
                'hypervolume': 0.0,
                'spacing': 0.0,
                'pareto_size': 0,
                'best_f1': float('inf'),
                'best_f2': float('inf')
            }
        
        # Get objective values
        f1_values = [s.f1 for s in self.global_pareto_front]
        f2_values = [s.f2 for s in self.global_pareto_front]
        
        # Normalize objectives to [0, 1] interval
        # Using global bounds for consistent comparison across runs
        f1_min, f1_max = min(f1_values), max(f1_values)
        f2_min, f2_max = min(f2_values), max(f2_values)
        
        f1_range = f1_max - f1_min if f1_max > f1_min else 1
        f2_range = f2_max - f2_min if f2_max > f2_min else 1
        
        # Normalize to [0, 1]
        normalized_points = [
            ((f1 - f1_min) / f1_range, (f2 - f2_min) / f2_range)
            for f1, f2 in zip(f1_values, f2_values)
        ]
        
        # Sort by first objective (ascending) for hypervolume calculation
        normalized_points.sort(key=lambda p: p[0])
        
        # Reference point: upper bounds after normalization (worst case)
        # Using (1.0, 1.0) as reference since data is normalized to [0,1]
        ref_f1 = 1.0
        ref_f2 = 1.0
        
        # Hypervolume calculation using the inclusion-exclusion method
        # Hv = sum(|F1(i+1) - F1(i)| * |F2_ref - F2(i)|) for sorted solutions
        hypervolume = 0.0
        n = len(normalized_points)
        
        for i in range(n):
            x_i, y_i = normalized_points[i]
            
            # Width: difference to next point (or to reference for last point)
            if i < n - 1:
                x_next = normalized_points[i + 1][0]
            else:
                x_next = ref_f1
            
            width = x_next - x_i
            height = ref_f2 - y_i
            
            if width > 0 and height > 0:
                hypervolume += width * height
        
        # Maximum possible hypervolume is ref_f1 * ref_f2 = 1.0
        # No additional normalization needed
        
        # Spacing metric (SP) - measures uniformity of solution distribution
        # SP = sqrt(sum((d_i - d_mean)^2) / |PF|)
        # where d_i is the minimum distance from solution i to all other solutions
        spacing = 0.0
        if len(self.global_pareto_front) > 1:
            distances = []
            for i, (x1, y1) in enumerate(normalized_points):
                min_dist = float('inf')
                for j, (x2, y2) in enumerate(normalized_points):
                    if i != j:
                        # Euclidean distance in normalized objective space
                        dist = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                        min_dist = min(min_dist, dist)
                if min_dist < float('inf'):
                    distances.append(min_dist)
            
            if distances:
                d_mean = sum(distances) / len(distances)
                # Standard deviation of distances (lower = more uniform)
                spacing = math.sqrt(
                    sum((d - d_mean)**2 for d in distances) / len(distances)
                )
        
        return {
            'hypervolume': hypervolume,
            'spacing': spacing,
            'pareto_size': len(self.global_pareto_front),
            'best_f1': min(f1_values),
            'best_f2': min(f2_values)
        }
