"""
Data Parser for Solomon Benchmark Instances
Parses Solomon VRPTW format files for HHC-MOVRPTW
"""

import os
from typing import Optional

try:
    from .problem import Customer, HHCInstance
except ImportError:  # Allows running as a script from the src directory.
    from problem import Customer, HHCInstance
# Experimentally optimized number of clusters (K) for each benchmark instance.
# K includes a depot cluster during fitting, but since our K-means operates
# on customers only (no depot), we store the effective K-1 value here.
OPTIMAL_K_VALUES = {
    'C101.25': 2,
    'C101.50': 3,
    'C101.100': 4,
    'C107.25': 2,
    'C107.50': 3,
    'C107.100': 4,
    'C206.25': 2,
    'C206.50': 3,
    'R109.25': 2,
    'R109.50': 3,
    'RC106.25': 2,
    'RC106.50': 3,
}

# In Home Health Care, service time is standardized to 20 minutes per visit.
HHC_SERVICE_TIME = 20.0


def get_num_clusters(instance_name: str, num_customers: int) -> int:
    """
    Get the optimal number of clusters (caregivers) for a given instance.
    Uses tuned values when available, otherwise applies a size-based heuristic.
    
    Args:
        instance_name: Instance name (e.g., 'C101.25')
        num_customers: Number of customers in the instance
    
    Returns:
        Number of clusters for K-means (customer-only, excluding depot)
    """
    name_upper = instance_name.upper()
    if name_upper in OPTIMAL_K_VALUES:
        return OPTIMAL_K_VALUES[name_upper]
    
    # Also try constructing key as "NAME.SIZE" (instance.name from file is just "C101")
    composite_key = f"{name_upper}.{num_customers}"
    if composite_key in OPTIMAL_K_VALUES:
        return OPTIMAL_K_VALUES[composite_key]
    
    # Heuristic for unknown instances: ~10-13 customers per cluster
    if num_customers <= 30:
        return 2
    elif num_customers <= 60:
        return 3
    elif num_customers <= 120:
        return 4
    else:
        return max(2, num_customers // 25)

def parse_solomon_file(filepath: str) -> HHCInstance:
    """
    Parse a Solomon benchmark file
    
    Format:
    Line 1: Instance name
    Lines 2-4: Empty or headers
    Line 5: VEHICLE section header
    Line 6: NUMBER CAPACITY
    Line 7: vehicle_count vehicle_capacity
    Lines 8-9: Empty or headers
    Line 10: CUSTOMER section header  
    Line 11: Column headers
    Line 12+: CUST_NO. XCOORD. YCOORD. DEMAND READY_TIME DUE_DATE SERVICE_TIME
    """
    customers = []
    depot = None
    name = ""
    num_vehicles = 0
    capacity = 0.0
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Parse instance name
    name = lines[0].strip()
    
    # Find vehicle info (usually around line 4-5)
    for i, line in enumerate(lines):
        parts = line.split()
        if len(parts) == 2:
            try:
                num_vehicles = int(parts[0])
                capacity = float(parts[1])
                break
            except ValueError:
                continue
    
    # Parse customers - find the data section
    data_started = False
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        parts = line.split()
        
        # Skip non-numeric lines
        if len(parts) >= 7:
            try:
                cust_id = int(parts[0])
                x = float(parts[1])
                y = float(parts[2])
                demand = float(parts[3])
                ready_time = float(parts[4])
                due_date = float(parts[5])
                service_time = float(parts[6])
                
                customer = Customer(
                    id=cust_id,
                    x=x,
                    y=y,
                    demand=demand,
                    ready_time=ready_time,
                    due_date=due_date,
                    service_time=service_time
                )
                
                if cust_id == 0:
                    depot = customer
                else:
                    customers.append(customer)
                    
            except ValueError:
                continue
    
    if depot is None:
        # Create default depot at origin
        depot = Customer(id=0, x=0, y=0, demand=0, ready_time=0, due_date=1000, service_time=0)
    
    # HHC standardized service time: 20 minutes per patient visit
    for customer in customers:
        customer.service_time = HHC_SERVICE_TIME
    
    return HHCInstance(
        name=name,
        num_vehicles=num_vehicles,
        vehicle_capacity=capacity,
        customers=customers,
        depot=depot
    )


def load_instance(instance_name: str, dataset_path: Optional[str] = None) -> HHCInstance:
    """
    Load a Solomon benchmark instance by name
    
    Args:
        instance_name: Name like "C101.25", "R201.50", "RC106.100"
        dataset_path: Path to datasets folder (default: ./datasets)
    
    Returns:
        HHCInstance object
    """
    if dataset_path is None:
        # Find datasets folder relative to this file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        dataset_path = os.path.join(os.path.dirname(current_dir), 'datasets')
    
    # Parse instance name (e.g., "C101.25" -> type="C101", size=25)
    parts = instance_name.split('.')
    if len(parts) == 2:
        instance_type = parts[0]  # e.g., "C101"
        size = parts[1]           # e.g., "25"
        filename = f"{instance_type.lower()}_{size}.txt"
    else:
        filename = f"{instance_name.lower()}.txt"
    
    # Search in subdirectories too
    filepath = os.path.join(dataset_path, filename)
    
    if not os.path.exists(filepath):
        # Try subdirectories (C_type, R_type, RC_type)
        for subdir in ['C_type', 'R_type', 'RC_type']:
            subpath = os.path.join(dataset_path, subdir, filename)
            if os.path.exists(subpath):
                filepath = subpath
                break
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Instance file not found: {filepath}")
    
    return parse_solomon_file(filepath)


def list_available_instances(dataset_path: Optional[str] = None) -> list:
    """List all available instances in the datasets folder"""
    if dataset_path is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        dataset_path = os.path.join(os.path.dirname(current_dir), 'datasets')
    
    instances = []
    if os.path.exists(dataset_path):
        # Check root directory
        for filename in os.listdir(dataset_path):
            if filename.endswith('.txt'):
                name = filename.replace('.txt', '').replace('_', '.').upper()
                instances.append(name)
        
        # Check subdirectories
        for subdir in ['C_type', 'R_type', 'RC_type']:
            subpath = os.path.join(dataset_path, subdir)
            if os.path.exists(subpath):
                for filename in os.listdir(subpath):
                    if filename.endswith('.txt'):
                        name = filename.replace('.txt', '').replace('_', '.').upper()
                        if name not in instances:
                            instances.append(name)
    
    return sorted(instances)
