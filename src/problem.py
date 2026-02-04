"""
Problem Definition for HHC-MOVRPTW
Defines objective functions and constraint handling
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
from src.data_parser import ProblemInstance, Customer


@dataclass
class Route:
    """Represents a single route for one caregiver"""
    caregiver_id: int
    customer_ids: List[int]  # Sequence of customer IDs to visit (excluding depot)
    
    def __repr__(self):
        return f"Route(caregiver={self.caregiver_id}, customers={self.customer_ids})"


@dataclass
class Solution:
    """
    Represents a complete solution to the HHC-MOVRPTW problem.
    A solution consists of multiple routes, one per caregiver.
    """
    routes: List[Route]
    objective_values: Tuple[float, float] = None  # (F1, F2)
    
    @property
    def num_routes(self) -> int:
        return len(self.routes)
    
    def get_all_customers(self) -> List[int]:
        """Get all customer IDs across all routes"""
        customers = []
        for route in self.routes:
            customers.extend(route.customer_ids)
        return customers
    
    def __repr__(self):
        if self.objective_values:
            return f"Solution(routes={self.num_routes}, F1={self.objective_values[0]:.2f}, F2={self.objective_values[1]:.2f})"
        return f"Solution(routes={self.num_routes})"


class HHCProblem:
    """
    HHC-MOVRPTW Problem Handler
    
    Objectives:
        F1: Minimize total service time (travel time + working time)
        F2: Minimize total tardiness (deviation from patient time preferences)
    
    Constraints:
        - Each patient assigned to exactly one caregiver
        - Caregivers start and end at depot
        - Vehicle capacity constraints
        - Maximum workload per caregiver
        - Time window constraints
    """
    
    def __init__(self, instance: ProblemInstance, max_workload: float = None):
        """
        Initialize the HHC problem.
        
        Args:
            instance: Problem instance with customer and vehicle data
            max_workload: Maximum daily workload per caregiver (in time units)
        """
        self.instance = instance
        self.num_customers = instance.num_customers
        self.num_caregivers = instance.num_vehicles
        
        # Set default max workload if not specified
        if max_workload is None:
            # Use the depot's due date as max workload (planning horizon)
            self.max_workload = instance.depot.due_date
        else:
            self.max_workload = max_workload
        
        # Pre-compute maximum patients per caregiver (fair distribution)
        self.max_patients_per_caregiver = self._calculate_max_patients()
    
    def _calculate_max_patients(self) -> int:
        """Calculate maximum patients per caregiver for balanced workload"""
        n = self.num_customers
        k = self.num_caregivers
        
        if n % k == 0:
            return n // k
        else:
            return (n // k) + 1
    
    def evaluate_route(self, route: Route) -> Tuple[float, float, bool, dict]:
        """
        Evaluate a single route.
        
        Returns:
            (service_time, tardiness, is_feasible, details)
        """
        if not route.customer_ids:
            return 0.0, 0.0, True, {}
        
        instance = self.instance
        depot = instance.depot
        
        total_travel_time = 0.0
        total_service_time = 0.0
        total_tardiness = 0.0
        total_demand = 0
        
        current_time = 0.0
        current_node = 0  # Start at depot
        
        arrival_times = []
        service_start_times = []
        
        for customer_id in route.customer_ids:
            customer = instance.customers[customer_id - 1]  # customer_id is 1-indexed
            
            # Travel to customer
            travel_time = instance.get_travel_time(current_node, customer_id)
            total_travel_time += travel_time
            current_time += travel_time
            
            arrival_times.append(current_time)
            
            # Wait if arrived early
            if current_time < customer.ready_time:
                current_time = customer.ready_time
            
            service_start_time = current_time
            service_start_times.append(service_start_time)
            
            # Calculate tardiness (if arrived after ready_time or late compared to due_date)
            # F2: sum of max(ready_time - service_start, 0) + max(service_start + service_time - due_date, 0)
            early_tardiness = max(customer.ready_time - service_start_time, 0)
            late_tardiness = max((service_start_time + customer.service_time) - customer.due_date, 0)
            total_tardiness += early_tardiness + late_tardiness
            
            # Perform service
            total_service_time += customer.service_time
            current_time += customer.service_time
            
            # Update demand
            total_demand += customer.demand
            
            # Move to next node
            current_node = customer_id
        
        # Return to depot
        travel_time = instance.get_travel_time(current_node, 0)
        total_travel_time += travel_time
        current_time += travel_time
        
        # Check feasibility
        is_feasible = True
        
        # Capacity constraint
        if total_demand > instance.vehicle_capacity:
            is_feasible = False
        
        # Workload constraint
        if current_time > self.max_workload:
            is_feasible = False
        
        # F1: Total service time (travel + working)
        f1 = total_travel_time + total_service_time
        
        # F2: Total tardiness
        f2 = total_tardiness
        
        details = {
            'travel_time': total_travel_time,
            'service_time': total_service_time,
            'total_time': current_time,
            'total_demand': total_demand,
            'arrival_times': arrival_times,
            'service_start_times': service_start_times
        }
        
        return f1, f2, is_feasible, details
    
    def evaluate_solution(self, solution: Solution, customer_subset: List[int] = None) -> Tuple[float, float, bool]:
        """
        Evaluate a complete solution.
        
        Args:
            solution: Solution to evaluate
            customer_subset: If provided, only check that these customers are served
        
        Returns:
            (F1, F2, is_feasible)
        """
        total_f1 = 0.0
        total_f2 = 0.0
        is_feasible = True
        
        # Check all customers are served exactly once
        all_customers = solution.get_all_customers()
        
        if customer_subset is not None:
            # For sub-problems, only check subset is served
            expected_customers = set(customer_subset)
        else:
            expected_customers = set(range(1, self.num_customers + 1))
        
        if set(all_customers) != expected_customers or len(all_customers) != len(expected_customers):
            is_feasible = False
        
        # Evaluate each route
        for route in solution.routes:
            f1, f2, route_feasible, _ = self.evaluate_route(route)
            total_f1 += f1
            total_f2 += f2
            
            if not route_feasible:
                is_feasible = False
        
        solution.objective_values = (total_f1, total_f2)
        
        return total_f1, total_f2, is_feasible
    
    def dominates(self, sol1: Solution, sol2: Solution) -> bool:
        """
        Check if sol1 dominates sol2 (Pareto dominance for minimization).
        
        sol1 dominates sol2 if:
        - sol1 is no worse than sol2 in all objectives
        - sol1 is strictly better than sol2 in at least one objective
        """
        if sol1.objective_values is None or sol2.objective_values is None:
            raise ValueError("Solutions must be evaluated before dominance check")
        
        f1_1, f2_1 = sol1.objective_values
        f1_2, f2_2 = sol2.objective_values
        
        # sol1 must be <= sol2 in all objectives
        no_worse = (f1_1 <= f1_2) and (f2_1 <= f2_2)
        
        # sol1 must be < sol2 in at least one objective
        strictly_better = (f1_1 < f1_2) or (f2_1 < f2_2)
        
        return no_worse and strictly_better
    
    def get_non_dominated(self, solutions: List[Solution]) -> List[Solution]:
        """
        Get non-dominated solutions from a list (Pareto front).
        """
        non_dominated = []
        
        for sol in solutions:
            is_dominated = False
            
            for other in solutions:
                if other is not sol and self.dominates(other, sol):
                    is_dominated = True
                    break
            
            if not is_dominated:
                non_dominated.append(sol)
        
        return non_dominated


def create_random_solution(problem: HHCProblem) -> Solution:
    """Create a random feasible solution"""
    import random
    
    # Get all customer IDs
    customer_ids = list(range(1, problem.num_customers + 1))
    random.shuffle(customer_ids)
    
    # Distribute customers among caregivers
    routes = []
    customers_per_route = problem.max_patients_per_caregiver
    
    for i in range(problem.num_caregivers):
        start_idx = i * customers_per_route
        end_idx = min(start_idx + customers_per_route, len(customer_ids))
        
        if start_idx < len(customer_ids):
            route_customers = customer_ids[start_idx:end_idx]
            routes.append(Route(caregiver_id=i, customer_ids=route_customers))
    
    solution = Solution(routes=routes)
    problem.evaluate_solution(solution)
    
    return solution


if __name__ == "__main__":
    # Test the problem definition
    from src.data_parser import load_instance
    
    print("Testing HHC Problem Definition")
    print("=" * 50)
    
    # Load test instance
    instance = load_instance("C101.25")
    problem = HHCProblem(instance)
    
    print(f"\nInstance: {instance.name}")
    print(f"Customers: {problem.num_customers}")
    print(f"Caregivers: {problem.num_caregivers}")
    print(f"Max patients per caregiver: {problem.max_patients_per_caregiver}")
    print(f"Max workload: {problem.max_workload}")
    
    # Create and evaluate random solution
    print("\nCreating random solution...")
    solution = create_random_solution(problem)
    
    print(f"Solution: {solution}")
    print(f"Objective F1 (service time): {solution.objective_values[0]:.2f}")
    print(f"Objective F2 (tardiness): {solution.objective_values[1]:.2f}")
    
    # Test Pareto dominance
    print("\nTesting Pareto dominance...")
    sol1 = create_random_solution(problem)
    sol2 = create_random_solution(problem)
    
    print(f"Sol1: F1={sol1.objective_values[0]:.2f}, F2={sol1.objective_values[1]:.2f}")
    print(f"Sol2: F1={sol2.objective_values[0]:.2f}, F2={sol2.objective_values[1]:.2f}")
    print(f"Sol1 dominates Sol2: {problem.dominates(sol1, sol2)}")
    print(f"Sol2 dominates Sol1: {problem.dominates(sol2, sol1)}")
