"""
NSGA-II (Non-dominated Sorting Genetic Algorithm II)
======================================================
Implementation for HHC-MOVRPTW optimization

A fast and elitist multi-objective genetic algorithm featuring:
- Fast non-dominated sorting with O(MN^2) complexity
- Crowding distance for diversity preservation
- Elitist selection mechanism

This is used as the core optimizer in Stage 2 of K-NSGA-II.
"""

import random
import math
from typing import List, Tuple, Optional, Dict
from .problem import HHCInstance, Customer, Solution


class NSGA2:
    """
    NSGA-II Algorithm for Multi-Objective Optimization
    
    A population-based evolutionary algorithm that maintains a diverse
    set of Pareto-optimal solutions through:
    - Non-dominated sorting (rank-based selection)
    - Crowding distance (diversity preservation)
    - Elitism (preserving best solutions)
    
    Optimizes:
        F1: Total service time (minimize)
        F2: Total tardiness (minimize)
    """
    
    def __init__(
        self,
        instance: HHCInstance,
        customers: Optional[List[Customer]] = None,
        population_size: int = 100,
        max_generations: int = 1000,
        crossover_rate: float = 0.9,
        mutation_rate: float = 0.1,
        random_state: Optional[int] = None
    ):
        """
        Initialize NSGA-II
        
        Args:
            instance: HHC problem instance
            customers: Subset of customers for this cluster (None = all)
            population_size: Size of population
            max_generations: Number of generations
            crossover_rate: Probability of crossover
            mutation_rate: Probability of mutation
            random_state: Random seed for reproducibility
        """
        self.instance = instance
        self.customers = customers if customers else instance.customers
        self.population_size = population_size
        self.max_generations = max_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        
        if random_state is not None:
            random.seed(random_state)
        
        self.population: List[Solution] = []
        self.pareto_front: List[Solution] = []
    
    def _create_random_solution(self) -> Solution:
        """Create a random solution (chromosome)"""
        solution = Solution(self.instance)
        
        # Get customer IDs to route
        customer_ids = [c.id for c in self.customers]
        random.shuffle(customer_ids)
        
        # Simple: single route with all customers
        # This represents a permutation-based encoding
        solution.routes = [customer_ids]
        solution.evaluate()
        
        return solution
    
    def _create_nearest_neighbor_solution(self) -> Solution:
        """Create a solution using nearest neighbor heuristic"""
        solution = Solution(self.instance)
        
        unvisited = set(c.id for c in self.customers)
        route = []
        
        if not unvisited:
            solution.routes = [route]
            solution.evaluate()
            return solution
        
        # Start from customer nearest to depot
        depot = self.instance.depot
        current = min(self.customers, key=lambda c: 
                     math.sqrt((c.x - depot.x)**2 + (c.y - depot.y)**2))
        route.append(current.id)
        unvisited.remove(current.id)
        
        # Build route using nearest neighbor
        while unvisited:
            # Find nearest unvisited customer
            nearest = None
            min_dist = float('inf')
            
            for cid in unvisited:
                for c in self.customers:
                    if c.id == cid:
                        dist = math.sqrt((c.x - current.x)**2 + (c.y - current.y)**2)
                        if dist < min_dist:
                            min_dist = dist
                            nearest = c
                        break
            
            if nearest:
                route.append(nearest.id)
                unvisited.remove(nearest.id)
                current = nearest
        
        solution.routes = [route]
        solution.evaluate()
        return solution
    
    def _initialize_population(self) -> List[Solution]:
        """Initialize population with random and heuristic solutions"""
        population = []
        
        # Add some heuristic solutions
        for _ in range(min(5, self.population_size // 4)):
            sol = self._create_nearest_neighbor_solution()
            population.append(sol)
        
        # Fill rest with random solutions
        while len(population) < self.population_size:
            sol = self._create_random_solution()
            population.append(sol)
        
        return population
    
    def _fast_non_dominated_sort(self, population: List[Solution]) -> List[List[Solution]]:
        """
        Fast non-dominated sorting (NSGA-II)
        Returns list of fronts, each front is a list of solutions
        """
        n = len(population)
        S: Dict[int, List[int]] = {i: [] for i in range(n)}  # Solutions dominated by i
        n_p: Dict[int, int] = {i: 0 for i in range(n)}        # Domination count for i
        fronts: List[List[Solution]] = [[]]
        rank: Dict[int, int] = {}
        
        # Calculate domination for each pair
        for p in range(n):
            for q in range(n):
                if p == q:
                    continue
                
                if population[p].dominates(population[q]):
                    S[p].append(q)
                elif population[q].dominates(population[p]):
                    n_p[p] += 1
            
            if n_p[p] == 0:
                rank[p] = 0
                population[p].rank = 0
                fronts[0].append(population[p])
        
        # Build subsequent fronts
        i = 0
        while fronts[i]:
            next_front = []
            for p_idx, p in enumerate(population):
                if p in fronts[i]:
                    for q in S[p_idx]:
                        n_p[q] -= 1
                        if n_p[q] == 0:
                            rank[q] = i + 1
                            population[q].rank = i + 1
                            next_front.append(population[q])
            
            i += 1
            if next_front:
                fronts.append(next_front)
            else:
                break
        
        return fronts
    
    def _calculate_crowding_distance(self, front: List[Solution]) -> None:
        """Calculate crowding distance for solutions in a front"""
        n = len(front)
        if n == 0:
            return
        
        # Initialize distances
        for sol in front:
            sol.crowding_distance = 0.0
        
        # For each objective
        for obj in ['f1', 'f2']:
            # Sort by objective
            front.sort(key=lambda x: getattr(x, obj))
            
            # Boundary solutions get infinite distance
            front[0].crowding_distance = float('inf')
            front[-1].crowding_distance = float('inf')
            
            # Get objective range
            obj_min = getattr(front[0], obj)
            obj_max = getattr(front[-1], obj)
            obj_range = obj_max - obj_min
            
            if obj_range == 0:
                continue
            
            # Calculate crowding distance
            for i in range(1, n - 1):
                front[i].crowding_distance += (
                    getattr(front[i + 1], obj) - getattr(front[i - 1], obj)
                ) / obj_range
    
    def _tournament_selection(self, population: List[Solution]) -> Solution:
        """Binary tournament selection"""
        i1 = random.randint(0, len(population) - 1)
        i2 = random.randint(0, len(population) - 1)
        
        p1 = population[i1]
        p2 = population[i2]
        
        # Select based on rank and crowding distance
        if p1.rank < p2.rank:
            return p1
        elif p2.rank < p1.rank:
            return p2
        elif p1.crowding_distance > p2.crowding_distance:
            return p1
        else:
            return p2
    
    def _crossover(self, parent1: Solution, parent2: Solution) -> Tuple[Solution, Solution]:
        """Order crossover (OX) for permutation representation"""
        if random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        route1 = parent1.routes[0] if parent1.routes else []
        route2 = parent2.routes[0] if parent2.routes else []
        
        if len(route1) < 2 or len(route2) < 2:
            return parent1.copy(), parent2.copy()
        
        size = len(route1)
        
        # Select crossover points
        point1 = random.randint(0, size - 1)
        point2 = random.randint(0, size - 1)
        if point1 > point2:
            point1, point2 = point2, point1
        
        # Create offspring
        child1 = Solution(self.instance)
        child2 = Solution(self.instance)
        
        # Initialize with -1
        offspring1 = [-1] * size
        offspring2 = [-1] * size
        
        # Copy segment from parents
        for i in range(point1, point2 + 1):
            offspring1[i] = route1[i]
            offspring2[i] = route2[i]
        
        # Fill remaining positions
        def fill_remaining(offspring, other_parent, p1, p2):
            current_pos = (p2 + 1) % size
            parent_pos = (p2 + 1) % size
            
            while -1 in offspring:
                gene = other_parent[parent_pos]
                if gene not in offspring:
                    offspring[current_pos] = gene
                    current_pos = (current_pos + 1) % size
                parent_pos = (parent_pos + 1) % size
        
        fill_remaining(offspring1, route2, point1, point2)
        fill_remaining(offspring2, route1, point1, point2)
        
        child1.routes = [offspring1]
        child2.routes = [offspring2]
        
        child1.evaluate()
        child2.evaluate()
        
        return child1, child2
    
    def _mutate(self, solution: Solution) -> Solution:
        """Swap mutation for permutation representation"""
        if random.random() > self.mutation_rate:
            return solution
        
        mutated = solution.copy()
        
        if not mutated.routes or not mutated.routes[0]:
            return mutated
        
        route = mutated.routes[0]
        if len(route) < 2:
            return mutated
        
        # Swap two random positions
        i = random.randint(0, len(route) - 1)
        j = random.randint(0, len(route) - 1)
        route[i], route[j] = route[j], route[i]
        
        mutated.routes = [route]
        mutated.evaluate()
        
        return mutated
    
    def _create_offspring(self, population: List[Solution]) -> List[Solution]:
        """Create offspring population through selection, crossover, mutation"""
        offspring = []
        
        while len(offspring) < self.population_size:
            # Select parents
            parent1 = self._tournament_selection(population)
            parent2 = self._tournament_selection(population)
            
            # Crossover
            child1, child2 = self._crossover(parent1, parent2)
            
            # Mutation
            child1 = self._mutate(child1)
            child2 = self._mutate(child2)
            
            offspring.extend([child1, child2])
        
        return offspring[:self.population_size]
    
    def _select_next_generation(self, combined: List[Solution]) -> List[Solution]:
        """Select next generation from combined parent+offspring population"""
        # Non-dominated sorting
        fronts = self._fast_non_dominated_sort(combined)
        
        # Calculate crowding distance for each front
        for front in fronts:
            self._calculate_crowding_distance(front)
        
        # Select solutions for next generation
        next_gen = []
        front_idx = 0
        
        while len(next_gen) + len(fronts[front_idx]) <= self.population_size:
            next_gen.extend(fronts[front_idx])
            front_idx += 1
            if front_idx >= len(fronts):
                break
        
        # If we need more solutions, select from next front by crowding distance
        if len(next_gen) < self.population_size and front_idx < len(fronts):
            remaining = self.population_size - len(next_gen)
            fronts[front_idx].sort(key=lambda x: x.crowding_distance, reverse=True)
            next_gen.extend(fronts[front_idx][:remaining])
        
        return next_gen
    
    def run(self, verbose: bool = False) -> List[Solution]:
        """
        Run NSGA-II optimization
        
        Args:
            verbose: Print progress information
        
        Returns:
            Pareto front (list of non-dominated solutions)
        """
        # Initialize population
        self.population = self._initialize_population()
        
        # Initial sorting
        fronts = self._fast_non_dominated_sort(self.population)
        for front in fronts:
            self._calculate_crowding_distance(front)
        
        # Main loop
        for gen in range(self.max_generations):
            # Create offspring
            offspring = self._create_offspring(self.population)
            
            # Combine parent and offspring
            combined = self.population + offspring
            
            # Select next generation
            self.population = self._select_next_generation(combined)
        
        # Extract final Pareto front
        fronts = self._fast_non_dominated_sort(self.population)
        self.pareto_front = fronts[0] if fronts else []
        
        return self.pareto_front
    
    def get_pareto_front(self) -> List[Solution]:
        """Return the Pareto front"""
        return self.pareto_front
    
    def get_best_f1(self) -> Optional[Solution]:
        """Get solution with best F1 value"""
        if not self.pareto_front:
            return None
        return min(self.pareto_front, key=lambda x: x.f1)
    
    def get_best_f2(self) -> Optional[Solution]:
        """Get solution with best F2 value"""
        if not self.pareto_front:
            return None
        return min(self.pareto_front, key=lambda x: x.f2)
