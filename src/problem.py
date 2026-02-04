"""
Problem Definition for HHC-MOVRPTW
Home Health Care Multi-Objective Vehicle Routing Problem with Time Windows
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import math


@dataclass
class Customer:
    """Represents a customer/patient in the HHC problem"""
    id: int
    x: float
    y: float
    demand: float
    ready_time: float      # Start of time window
    due_date: float        # End of time window
    service_time: float
    
    def __hash__(self):
        return hash(self.id)
    
    def __eq__(self, other):
        if isinstance(other, Customer):
            return self.id == other.id
        return False


@dataclass
class HHCInstance:
    """
    Home Health Care Instance
    Based on Solomon benchmark format adapted for HHC
    """
    name: str
    num_vehicles: int          # Number of caregivers
    vehicle_capacity: float    # Caregiver capacity (work hours)
    customers: List[Customer]  # List of patients (excluding depot)
    depot: Customer            # Depot/healthcare center
    
    @property
    def num_customers(self) -> int:
        return len(self.customers)
    
    def get_distance(self, c1: Customer, c2: Customer) -> float:
        """Calculate Euclidean distance between two customers"""
        return math.sqrt((c1.x - c2.x)**2 + (c1.y - c2.y)**2)
    
    def get_distance_matrix(self) -> List[List[float]]:
        """Generate full distance matrix including depot"""
        all_nodes = [self.depot] + self.customers
        n = len(all_nodes)
        matrix = [[0.0] * n for _ in range(n)]
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    matrix[i][j] = self.get_distance(all_nodes[i], all_nodes[j])
        
        return matrix
    
    def get_travel_time(self, c1: Customer, c2: Customer) -> float:
        """Travel time equals distance (assuming unit speed)"""
        return self.get_distance(c1, c2)


class Solution:
    """
    Represents a solution to the HHC-MOVRPTW problem
    A solution consists of routes for each caregiver
    """
    def __init__(self, instance: HHCInstance):
        self.instance = instance
        self.routes: List[List[int]] = []  # List of routes, each route is list of customer IDs
        self.f1: float = float('inf')      # Objective 1: Total service time
        self.f2: float = float('inf')      # Objective 2: Total tardiness
        
        # For NSGA-II
        self.rank: int = 0
        self.crowding_distance: float = 0.0
    
    def copy(self) -> 'Solution':
        """Create a deep copy of the solution"""
        new_sol = Solution(self.instance)
        new_sol.routes = [route.copy() for route in self.routes]
        new_sol.f1 = self.f1
        new_sol.f2 = self.f2
        new_sol.rank = self.rank
        new_sol.crowding_distance = self.crowding_distance
        return new_sol
    
    def evaluate(self) -> Tuple[float, float]:
        """
        Evaluate the solution and compute objective values
        
        F1 (Service Time): Sum of all travel times + service times
        F2 (Tardiness): Sum of max(0, arrival_time - due_date) for all customers
        
        Constraints enforced:
        - Vehicle capacity constraint (demand vs capacity)
        - Time window constraints (soft - penalized via tardiness)
        """
        total_service_time = 0.0
        total_tardiness = 0.0
        
        depot = self.instance.depot
        
        for route in self.routes:
            if not route:
                continue
            
            # Start from depot
            current_time = 0.0
            prev_node = depot
            route_demand = 0.0
            
            for cust_id in route:
                # Find customer
                customer = None
                for c in self.instance.customers:
                    if c.id == cust_id:
                        customer = c
                        break
                
                if customer is None:
                    continue
                
                # Accumulate demand for capacity check
                route_demand += customer.demand
                
                # Travel time from previous node
                travel_time = self.instance.get_travel_time(prev_node, customer)
                total_service_time += travel_time
                
                # Arrival time at customer
                arrival_time = current_time + travel_time
                
                # Wait if arriving before ready time (earliest start)
                start_service = max(arrival_time, customer.ready_time)
                
                # Calculate tardiness (late arrival penalty)
                tardiness = max(0, arrival_time - customer.due_date)
                total_tardiness += tardiness
                
                # Add service time
                total_service_time += customer.service_time
                current_time = start_service + customer.service_time
                
                prev_node = customer
            
            # Return to depot
            if route:
                travel_time = self.instance.get_travel_time(prev_node, depot)
                total_service_time += travel_time
            
            # Capacity constraint violation penalty
            if route_demand > self.instance.vehicle_capacity:
                capacity_violation = route_demand - self.instance.vehicle_capacity
                # Add penalty to both objectives
                total_service_time += capacity_violation * 100
                total_tardiness += capacity_violation * 100
        
        self.f1 = total_service_time
        self.f2 = total_tardiness
        
        return self.f1, self.f2
    
    def is_feasible(self) -> bool:
        """Check if solution satisfies all hard constraints"""
        for route in self.routes:
            route_demand = 0.0
            for cust_id in route:
                for c in self.instance.customers:
                    if c.id == cust_id:
                        route_demand += c.demand
                        break
            if route_demand > self.instance.vehicle_capacity:
                return False
        return True
    
    def dominates(self, other: 'Solution') -> bool:
        """Check if this solution dominates another (for minimization)"""
        return (self.f1 <= other.f1 and self.f2 <= other.f2 and 
                (self.f1 < other.f1 or self.f2 < other.f2))
    
    def __lt__(self, other: 'Solution') -> bool:
        """Comparison for sorting (by rank, then crowding distance)"""
        if self.rank != other.rank:
            return self.rank < other.rank
        return self.crowding_distance > other.crowding_distance
    
    def __repr__(self) -> str:
        return f"Solution(F1={self.f1:.2f}, F2={self.f2:.2f}, routes={len(self.routes)})"
