"""
Data Parser for Solomon VRPTW Benchmark Instances
Parses the dataset files and creates structured data for the HHC problem
"""

import os
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple

@dataclass
class Customer:
    """Represents a patient/customer in the HHC problem"""
    id: int
    x: float
    y: float
    demand: int
    ready_time: int      # Earliest service start time
    due_date: int        # Latest service start time
    service_time: int    # Time required to perform service
    
    def __repr__(self):
        return f"Customer(id={self.id}, pos=({self.x},{self.y}), tw=[{self.ready_time},{self.due_date}])"


@dataclass 
class ProblemInstance:
    """Represents a complete HHC-MOVRPTW problem instance"""
    name: str
    num_vehicles: int           # Number of caregivers
    vehicle_capacity: int       # Vehicle capacity
    depot: Customer             # Depot (node 0)
    customers: List[Customer]   # List of patients
    distance_matrix: np.ndarray # Distance/travel time matrix
    
    @property
    def num_customers(self) -> int:
        return len(self.customers)
    
    @property
    def all_nodes(self) -> List[Customer]:
        """Returns depot + all customers"""
        return [self.depot] + self.customers
    
    def get_travel_time(self, from_id: int, to_id: int) -> float:
        """Get travel time between two nodes"""
        return self.distance_matrix[from_id][to_id]
    
    def __repr__(self):
        return f"ProblemInstance(name={self.name}, customers={self.num_customers}, vehicles={self.num_vehicles})"


def calculate_distance_matrix(nodes: List[Customer]) -> np.ndarray:
    """
    Calculate Euclidean distance matrix between all nodes.
    In the paper, travel time is measured in minutes and proportional to distance.
    """
    n = len(nodes)
    matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i != j:
                dx = nodes[i].x - nodes[j].x
                dy = nodes[i].y - nodes[j].y
                # Euclidean distance (can be scaled to represent travel time)
                matrix[i][j] = np.sqrt(dx**2 + dy**2)
    
    return matrix


def parse_solomon_instance(filepath: str) -> ProblemInstance:
    """
    Parse a Solomon VRPTW benchmark file.
    
    File format:
    - Line 1: Instance name
    - Lines 2-4: Vehicle info header
    - Line 5: NUMBER CAPACITY
    - Lines 6-8: Customer info header  
    - Line 9+: Customer data (id, x, y, demand, ready_time, due_date, service_time)
    """
    
    with open(filepath, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    
    # Parse instance name
    name = lines[0]
    
    # Find vehicle info (look for line with two numbers after "VEHICLE" section)
    vehicle_line_idx = None
    for i, line in enumerate(lines):
        if 'NUMBER' in line and 'CAPACITY' in line:
            vehicle_line_idx = i + 1
            break
    
    if vehicle_line_idx is None:
        raise ValueError(f"Could not find vehicle info in {filepath}")
    
    vehicle_parts = lines[vehicle_line_idx].split()
    num_vehicles = int(vehicle_parts[0])
    vehicle_capacity = int(vehicle_parts[1])
    
    # Find customer data start
    customer_start_idx = None
    for i, line in enumerate(lines):
        if 'CUST NO' in line or 'CUST NO.' in line:
            customer_start_idx = i + 1
            break
    
    if customer_start_idx is None:
        raise ValueError(f"Could not find customer data in {filepath}")
    
    # Parse customers
    customers = []
    depot = None
    
    for line in lines[customer_start_idx:]:
        parts = line.split()
        if len(parts) >= 7:
            customer = Customer(
                id=int(parts[0]),
                x=float(parts[1]),
                y=float(parts[2]),
                demand=int(parts[3]),
                ready_time=int(parts[4]),
                due_date=int(parts[5]),
                service_time=int(parts[6])
            )
            
            if customer.id == 0:
                depot = customer
            else:
                customers.append(customer)
    
    if depot is None:
        raise ValueError(f"Could not find depot (node 0) in {filepath}")
    
    # Calculate distance matrix
    all_nodes = [depot] + customers
    distance_matrix = calculate_distance_matrix(all_nodes)
    
    return ProblemInstance(
        name=name,
        num_vehicles=num_vehicles,
        vehicle_capacity=vehicle_capacity,
        depot=depot,
        customers=customers,
        distance_matrix=distance_matrix
    )


def load_instance(instance_name: str, base_path: str = "datasets") -> ProblemInstance:
    """
    Load a problem instance by name.
    
    Args:
        instance_name: e.g., "C101.25", "R109.50", "RC106.100"
        base_path: path to datasets folder
    
    Returns:
        ProblemInstance object
    """
    # Parse instance name
    parts = instance_name.upper().replace('.', '_').split('_')
    base_name = parts[0].lower()  # e.g., "c101"
    
    if len(parts) > 1:
        size = parts[1]  # e.g., "25", "50", "100"
        if size == "100":
            filename = f"{base_name}.txt"
        else:
            filename = f"{base_name}_{size}.txt"
    else:
        filename = f"{base_name}.txt"
    
    # Determine folder based on type
    if base_name.startswith('rc'):
        folder = "RC_type"
    elif base_name.startswith('r'):
        folder = "R_type"
    else:
        folder = "C_type"
    
    filepath = os.path.join(base_path, folder, filename)
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Instance file not found: {filepath}")
    
    return parse_solomon_instance(filepath)


def get_customer_coordinates(instance: ProblemInstance) -> np.ndarray:
    """Extract customer coordinates as numpy array for clustering"""
    coords = np.array([[c.x, c.y] for c in instance.customers])
    return coords


def get_customer_features(instance: ProblemInstance, include_time: bool = True) -> np.ndarray:
    """
    Extract customer features for clustering.
    Features: [x, y] or [x, y, ready_time, due_date] (normalized)
    """
    if include_time:
        features = []
        for c in instance.customers:
            features.append([c.x, c.y, c.ready_time, c.due_date])
        features = np.array(features)
        
        # Normalize features
        min_vals = features.min(axis=0)
        max_vals = features.max(axis=0)
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1  # Avoid division by zero
        
        normalized = (features - min_vals) / range_vals
        return normalized
    else:
        return get_customer_coordinates(instance)


if __name__ == "__main__":
    # Test the parser
    print("Testing Solomon Instance Parser")
    print("=" * 50)
    
    # Test loading instances
    test_instances = ["C101.25", "C101.100", "R109.25", "RC106.50"]
    
    for inst_name in test_instances:
        try:
            instance = load_instance(inst_name)
            print(f"\n{inst_name}:")
            print(f"  Name: {instance.name}")
            print(f"  Customers: {instance.num_customers}")
            print(f"  Vehicles: {instance.num_vehicles}")
            print(f"  Capacity: {instance.vehicle_capacity}")
            print(f"  Depot: ({instance.depot.x}, {instance.depot.y})")
            print(f"  Distance matrix shape: {instance.distance_matrix.shape}")
        except FileNotFoundError as e:
            print(f"\n{inst_name}: {e}")
