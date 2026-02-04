"""
NSGA-II (Non-dominated Sorting Genetic Algorithm II) Implementation
For solving the HHC-MOVRPTW bi-objective optimization problem
"""

import numpy as np
import random
from typing import List, Tuple, Optional
from dataclasses import dataclass, field
from copy import deepcopy

from src.data_parser import ProblemInstance
from src.problem import HHCProblem, Solution, Route


@dataclass
class Individual:
    """
    Represents an individual (chromosome) in the NSGA-II population.
    The chromosome is represented as a permutation of customer IDs.
    """
    chromosome: List[int]  # Permutation of customer IDs
    objectives: Tuple[float, float] = None  # (F1, F2)
    rank: int = 0  # Pareto front rank
    crowding_distance: float = 0.0
    
    def __repr__(self):
        if self.objectives:
            return f"Individual(F1={self.objectives[0]:.2f}, F2={self.objectives[1]:.2f}, rank={self.rank})"
        return f"Individual(chromosome_len={len(self.chromosome)})"


class NSGAII:
    """
    NSGA-II Algorithm for Multi-Objective Optimization.
    
    Based on: Deb et al., "A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II"
    
    Key Features:
    - Fast non-dominated sorting O(MN²)
    - Crowding distance for diversity preservation
    - Elitist selection strategy
    """
    
    def __init__(self,
                 problem: HHCProblem,
                 population_size: int = 100,
                 max_generations: int = 1000,
                 crossover_rate: float = 0.7,
                 mutation_rate: float = 0.2,
                 random_state: int = None):
        """
        Initialize NSGA-II algorithm.
        
        Args:
            problem: HHC problem instance
            population_size: Size of population (N)
            max_generations: Maximum number of generations
            crossover_rate: Probability of crossover
            mutation_rate: Probability of mutation
            random_state: Random seed for reproducibility
        """
        self.problem = problem
        self.population_size = population_size
        self.max_generations = max_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        
        if random_state is not None:
            random.seed(random_state)
            np.random.seed(random_state)
        
        self.population: List[Individual] = []
        self.pareto_front: List[Individual] = []
        self.generation_history = []
        
        # Customer IDs to include (1-indexed)
        # By default, all customers
        self.customer_ids = list(range(1, problem.num_customers + 1))
    
    def set_customer_subset(self, customer_ids: List[int]):
        """Set a subset of customers for optimization (used in K-NSGA-II)"""
        self.customer_ids = customer_ids
    
    def _create_random_individual(self) -> Individual:
        """Create a random individual (permutation of customer IDs)"""
        chromosome = self.customer_ids.copy()
        random.shuffle(chromosome)
        return Individual(chromosome=chromosome)
    
    def _create_nearest_neighbor_individual(self, start_idx: int = None) -> Individual:
        """Create individual using nearest neighbor heuristic (for service time)"""
        customers = self.customer_ids.copy()
        if not customers:
            return Individual(chromosome=[])
        
        if start_idx is None:
            start_idx = random.randint(0, len(customers) - 1)
        start_idx = start_idx % len(customers)
        
        chromosome = [customers[start_idx]]
        remaining = customers.copy()
        remaining.remove(customers[start_idx])
        
        while remaining:
            last = chromosome[-1]
            # Customer IDs are 1-indexed, list is 0-indexed
            last_customer = self.problem.instance.customers[last - 1]
            
            # Find nearest unvisited customer
            best_dist = float('inf')
            best_cust = remaining[0]
            for cust_id in remaining:
                cust = self.problem.instance.customers[cust_id - 1]
                dist = ((last_customer.x - cust.x) ** 2 + (last_customer.y - cust.y) ** 2) ** 0.5
                if dist < best_dist:
                    best_dist = dist
                    best_cust = cust_id
            
            chromosome.append(best_cust)
            remaining.remove(best_cust)
        
        return Individual(chromosome=chromosome)
    
    def _create_earliest_deadline_individual(self) -> Individual:
        """Create individual sorted by due date (for tardiness)"""
        customers = self.customer_ids.copy()
        customers.sort(key=lambda cid: self.problem.instance.customers[cid - 1].due_date)
        return Individual(chromosome=customers)
    
    def _create_earliest_ready_individual(self) -> Individual:
        """Create individual sorted by ready time"""
        customers = self.customer_ids.copy()
        customers.sort(key=lambda cid: self.problem.instance.customers[cid - 1].ready_time)
        return Individual(chromosome=customers)
    
    def _decode_chromosome(self, chromosome: List[int]) -> Solution:
        """
        Decode chromosome to solution (routes).
        Splits the permutation into routes for each caregiver.
        """
        n_customers = len(chromosome)
        n_caregivers = self.problem.num_caregivers
        
        # For sub-problems (clusters), use single caregiver
        if n_customers <= self.problem.max_patients_per_caregiver:
            n_caregivers = 1
        
        # Calculate customers per route
        customers_per_route = n_customers // n_caregivers
        remainder = n_customers % n_caregivers
        
        routes = []
        start_idx = 0
        
        for i in range(n_caregivers):
            # Distribute remainder among first routes
            route_size = customers_per_route + (1 if i < remainder else 0)
            end_idx = start_idx + route_size
            
            if start_idx < n_customers:
                route_customers = chromosome[start_idx:end_idx]
                routes.append(Route(caregiver_id=i, customer_ids=route_customers))
            
            start_idx = end_idx
        
        return Solution(routes=routes)
    
    def _evaluate_individual(self, individual: Individual) -> Tuple[float, float]:
        """Evaluate an individual and return objective values"""
        solution = self._decode_chromosome(individual.chromosome)
        
        # Always pass customer_ids for sub-problem evaluation
        f1, f2, is_feasible = self.problem.evaluate_solution(solution, self.customer_ids)
        
        # Penalize infeasible solutions
        if not is_feasible:
            penalty = 1000000
            f1 += penalty
            f2 += penalty
        
        individual.objectives = (f1, f2)
        return f1, f2
    
    def _fast_non_dominated_sort(self, population: List[Individual]) -> List[List[Individual]]:
        """
        Fast non-dominated sorting algorithm.
        Returns list of fronts, where front[0] is the Pareto front.
        
        Complexity: O(MN²) where M is number of objectives, N is population size
        """
        n = len(population)
        
        # S[p] = set of solutions dominated by p
        # n[p] = number of solutions that dominate p
        S = [[] for _ in range(n)]
        n_dominated = [0] * n
        
        fronts = [[]]
        
        for p in range(n):
            for q in range(n):
                if p != q:
                    if self._dominates(population[p], population[q]):
                        S[p].append(q)
                    elif self._dominates(population[q], population[p]):
                        n_dominated[p] += 1
            
            if n_dominated[p] == 0:
                population[p].rank = 0
                fronts[0].append(population[p])
        
        i = 0
        while fronts[i]:
            next_front = []
            for p_ind in fronts[i]:
                p = population.index(p_ind)
                for q in S[p]:
                    n_dominated[q] -= 1
                    if n_dominated[q] == 0:
                        population[q].rank = i + 1
                        next_front.append(population[q])
            i += 1
            fronts.append(next_front)
        
        # Remove empty last front
        if not fronts[-1]:
            fronts.pop()
        
        return fronts
    
    def _dominates(self, ind1: Individual, ind2: Individual) -> bool:
        """Check if ind1 dominates ind2 (for minimization)"""
        if ind1.objectives is None or ind2.objectives is None:
            return False
        
        f1_1, f2_1 = ind1.objectives
        f1_2, f2_2 = ind2.objectives
        
        no_worse = (f1_1 <= f1_2) and (f2_1 <= f2_2)
        strictly_better = (f1_1 < f1_2) or (f2_1 < f2_2)
        
        return no_worse and strictly_better
    
    def _calculate_crowding_distance(self, front: List[Individual]):
        """
        Calculate crowding distance for individuals in a front.
        
        Crowding distance measures how close an individual is to its neighbors.
        Higher distance = more isolated = more preferred for diversity.
        """
        n = len(front)
        if n == 0:
            return
        
        for ind in front:
            ind.crowding_distance = 0.0
        
        if n <= 2:
            for ind in front:
                ind.crowding_distance = float('inf')
            return
        
        # For each objective
        for m in range(2):  # 2 objectives
            # Sort by objective m
            front.sort(key=lambda x: x.objectives[m])
            
            # Boundary individuals get infinite distance
            front[0].crowding_distance = float('inf')
            front[-1].crowding_distance = float('inf')
            
            # Calculate distance for intermediate individuals
            obj_range = front[-1].objectives[m] - front[0].objectives[m]
            if obj_range == 0:
                continue
            
            for i in range(1, n - 1):
                front[i].crowding_distance += (
                    (front[i + 1].objectives[m] - front[i - 1].objectives[m]) / obj_range
                )
    
    def _crowded_comparison(self, ind1: Individual, ind2: Individual) -> Individual:
        """
        Crowded comparison operator.
        Returns the better individual based on rank and crowding distance.
        """
        if ind1.rank < ind2.rank:
            return ind1
        elif ind2.rank < ind1.rank:
            return ind2
        elif ind1.crowding_distance > ind2.crowding_distance:
            return ind1
        else:
            return ind2
    
    def _tournament_selection(self, population: List[Individual], k: int = 2) -> Individual:
        """Binary tournament selection using crowded comparison"""
        candidates = random.sample(population, k)
        winner = candidates[0]
        for candidate in candidates[1:]:
            winner = self._crowded_comparison(winner, candidate)
        return winner
    
    def _order_crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """
        Order Crossover (OX) for permutation representation.
        Preserves relative order of elements.
        """
        size = len(parent1.chromosome)
        
        # Select crossover points
        point1, point2 = sorted(random.sample(range(size), 2))
        
        # Create offspring
        def create_child(p1, p2):
            child = [None] * size
            
            # Copy segment from p1
            child[point1:point2] = p1.chromosome[point1:point2]
            
            # Fill remaining from p2 in order
            p2_elements = [x for x in p2.chromosome if x not in child]
            
            idx = 0
            for i in range(size):
                if child[i] is None:
                    child[i] = p2_elements[idx]
                    idx += 1
            
            return Individual(chromosome=child)
        
        child1 = create_child(parent1, parent2)
        child2 = create_child(parent2, parent1)
        
        return child1, child2
    
    def _swap_mutation(self, individual: Individual) -> Individual:
        """Swap mutation: swap two random positions"""
        chromosome = individual.chromosome.copy()
        
        if len(chromosome) > 1:
            i, j = random.sample(range(len(chromosome)), 2)
            chromosome[i], chromosome[j] = chromosome[j], chromosome[i]
        
        return Individual(chromosome=chromosome)
    
    def _inversion_mutation(self, individual: Individual) -> Individual:
        """Inversion mutation: reverse a random segment"""
        chromosome = individual.chromosome.copy()
        
        if len(chromosome) > 2:
            i, j = sorted(random.sample(range(len(chromosome)), 2))
            chromosome[i:j+1] = chromosome[i:j+1][::-1]
        
        return Individual(chromosome=chromosome)
    
    def _insertion_mutation(self, individual: Individual) -> Individual:
        """Insertion mutation: remove element and insert at random position"""
        chromosome = individual.chromosome.copy()
        
        if len(chromosome) > 2:
            # Remove random element
            i = random.randint(0, len(chromosome) - 1)
            element = chromosome.pop(i)
            # Insert at random position
            j = random.randint(0, len(chromosome))
            chromosome.insert(j, element)
        
        return Individual(chromosome=chromosome)
    
    def _or_opt_mutation(self, individual: Individual) -> Individual:
        """Or-opt: move a sequence of 1-3 customers to another position"""
        chromosome = individual.chromosome.copy()
        n = len(chromosome)
        
        if n > 3:
            # Length of segment to move (1, 2, or 3)
            seg_len = random.randint(1, min(3, n - 1))
            # Starting position of segment
            start = random.randint(0, n - seg_len)
            # Extract segment
            segment = chromosome[start:start + seg_len]
            # Remove segment
            remaining = chromosome[:start] + chromosome[start + seg_len:]
            # Insert at random position
            insert_pos = random.randint(0, len(remaining))
            chromosome = remaining[:insert_pos] + segment + remaining[insert_pos:]
        
        return Individual(chromosome=chromosome)
    
    def _2opt_local_search(self, individual: Individual) -> Individual:
        """Apply 2-opt local search to improve a solution (lightweight version)"""
        chromosome = individual.chromosome.copy()
        n = len(chromosome)
        
        if n <= 3:
            return individual
        
        # Evaluate current solution
        test_ind = Individual(chromosome=chromosome)
        self._evaluate_individual(test_ind)
        best_obj = test_ind.objectives[0] + test_ind.objectives[1]  # Sum objectives
        
        # Limited 2-opt: try only a few random swaps
        for _ in range(min(10, n)):
            i = random.randint(0, n - 3)
            j = random.randint(i + 2, n - 1)
            
            # Try 2-opt swap (reverse segment between i+1 and j)
            new_chromosome = chromosome[:i+1] + chromosome[i+1:j+1][::-1] + chromosome[j+1:]
            test_ind = Individual(chromosome=new_chromosome)
            self._evaluate_individual(test_ind)
            new_obj = test_ind.objectives[0] + test_ind.objectives[1]
            
            if new_obj < best_obj:
                chromosome = new_chromosome
                best_obj = new_obj
        
        return Individual(chromosome=chromosome)
    
    def _create_offspring(self, population: List[Individual]) -> List[Individual]:
        """Create offspring population using genetic operators"""
        offspring = []
        
        while len(offspring) < self.population_size:
            # Selection
            parent1 = self._tournament_selection(population)
            parent2 = self._tournament_selection(population)
            
            # Crossover
            if random.random() < self.crossover_rate:
                child1, child2 = self._order_crossover(parent1, parent2)
            else:
                child1 = Individual(chromosome=parent1.chromosome.copy())
                child2 = Individual(chromosome=parent2.chromosome.copy())
            
            # Mutation - use multiple operators with different probabilities
            if random.random() < self.mutation_rate:
                mutation_type = random.random()
                if mutation_type < 0.3:
                    child1 = self._swap_mutation(child1)
                elif mutation_type < 0.6:
                    child1 = self._inversion_mutation(child1)
                elif mutation_type < 0.8:
                    child1 = self._insertion_mutation(child1)
                else:
                    child1 = self._or_opt_mutation(child1)
            
            if random.random() < self.mutation_rate:
                mutation_type = random.random()
                if mutation_type < 0.3:
                    child2 = self._swap_mutation(child2)
                elif mutation_type < 0.6:
                    child2 = self._inversion_mutation(child2)
                elif mutation_type < 0.8:
                    child2 = self._insertion_mutation(child2)
                else:
                    child2 = self._or_opt_mutation(child2)
            
            # Apply local search with very small probability
            if random.random() < 0.02:  # 2% chance
                child1 = self._2opt_local_search(child1)
            
            offspring.append(child1)
            if len(offspring) < self.population_size:
                offspring.append(child2)
        
        return offspring
    
    def _initialize_population(self):
        """Initialize population with a mix of heuristic and random individuals"""
        self.population = []
        
        # Seed with heuristic individuals for better initial coverage
        # 10% nearest neighbor from different starts
        n_nn = max(1, self.population_size // 10)
        for i in range(n_nn):
            individual = self._create_nearest_neighbor_individual(start_idx=i)
            self._evaluate_individual(individual)
            self.population.append(individual)
        
        # Add deadline-sorted individual
        ind_deadline = self._create_earliest_deadline_individual()
        self._evaluate_individual(ind_deadline)
        self.population.append(ind_deadline)
        
        # Add ready-time-sorted individual
        ind_ready = self._create_earliest_ready_individual()
        self._evaluate_individual(ind_ready)
        self.population.append(ind_ready)
        
        # Fill rest with random individuals
        while len(self.population) < self.population_size:
            individual = self._create_random_individual()
            self._evaluate_individual(individual)
            self.population.append(individual)
    
    def run(self, verbose: bool = True) -> List[Individual]:
        """
        Run NSGA-II algorithm.
        
        Returns:
            List of non-dominated individuals (Pareto front)
        """
        if verbose:
            print(f"\nRunning NSGA-II")
            print(f"  Population size: {self.population_size}")
            print(f"  Max generations: {self.max_generations}")
            print(f"  Customers: {len(self.customer_ids)}")
        
        # Initialize population
        self._initialize_population()
        
        # Initial non-dominated sort
        fronts = self._fast_non_dominated_sort(self.population)
        for front in fronts:
            self._calculate_crowding_distance(front)
        
        # Main loop
        for generation in range(self.max_generations):
            # Create offspring
            offspring = self._create_offspring(self.population)
            
            # Evaluate offspring
            for ind in offspring:
                self._evaluate_individual(ind)
            
            # Combine parent and offspring
            combined = self.population + offspring
            
            # Non-dominated sorting
            fronts = self._fast_non_dominated_sort(combined)
            
            # Select next generation
            new_population = []
            front_idx = 0
            
            while len(new_population) + len(fronts[front_idx]) <= self.population_size:
                # Add entire front
                self._calculate_crowding_distance(fronts[front_idx])
                new_population.extend(fronts[front_idx])
                front_idx += 1
                
                if front_idx >= len(fronts):
                    break
            
            # Fill remaining slots using crowding distance
            if len(new_population) < self.population_size and front_idx < len(fronts):
                self._calculate_crowding_distance(fronts[front_idx])
                fronts[front_idx].sort(key=lambda x: x.crowding_distance, reverse=True)
                
                remaining = self.population_size - len(new_population)
                new_population.extend(fronts[front_idx][:remaining])
            
            self.population = new_population
            
            # Track progress
            self.pareto_front = fronts[0] if fronts else []
            
            if verbose and (generation + 1) % 100 == 0:
                best_f1 = min(ind.objectives[0] for ind in self.pareto_front)
                best_f2 = min(ind.objectives[1] for ind in self.pareto_front)
                print(f"  Generation {generation + 1}: PF size={len(self.pareto_front)}, "
                      f"Best F1={best_f1:.2f}, Best F2={best_f2:.2f}")
            
            self.generation_history.append({
                'generation': generation + 1,
                'pareto_size': len(self.pareto_front),
                'best_f1': min(ind.objectives[0] for ind in self.pareto_front),
                'best_f2': min(ind.objectives[1] for ind in self.pareto_front)
            })
        
        # Final non-dominated sort
        fronts = self._fast_non_dominated_sort(self.population)
        self.pareto_front = fronts[0] if fronts else []
        
        if verbose:
            print(f"\nFinal Pareto front size: {len(self.pareto_front)}")
        
        return self.pareto_front
    
    def get_pareto_solutions(self) -> List[Tuple[float, float]]:
        """Get objective values of Pareto front solutions"""
        return [ind.objectives for ind in self.pareto_front]
    
    def get_best_solution(self, objective: int = 0) -> Individual:
        """Get the best solution for a specific objective (0=F1, 1=F2)"""
        if not self.pareto_front:
            return None
        return min(self.pareto_front, key=lambda x: x.objectives[objective])


if __name__ == "__main__":
    # Test NSGA-II
    from src.data_parser import load_instance
    
    print("Testing NSGA-II Algorithm")
    print("=" * 50)
    
    # Load test instance
    instance = load_instance("C101.25")
    problem = HHCProblem(instance)
    
    print(f"\nInstance: {instance.name}")
    print(f"Customers: {problem.num_customers}")
    print(f"Caregivers: {problem.num_caregivers}")
    
    # Run NSGA-II with smaller parameters for testing
    nsga2 = NSGAII(
        problem=problem,
        population_size=50,
        max_generations=100,
        crossover_rate=0.7,
        mutation_rate=0.2,
        random_state=42
    )
    
    pareto_front = nsga2.run(verbose=True)
    
    # Print Pareto front
    print("\nPareto Front Solutions:")
    for i, ind in enumerate(pareto_front[:10]):  # Show first 10
        print(f"  {i+1}. F1={ind.objectives[0]:.2f}, F2={ind.objectives[1]:.2f}")
    
    if len(pareto_front) > 10:
        print(f"  ... ({len(pareto_front) - 10} more solutions)")
