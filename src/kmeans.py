"""
K-means Clustering Implementation
For the decomposition stage of K-NSGA-II
"""

import random
import math
from typing import List, Tuple, Optional
from .problem import Customer


class KMeans:
    """
    K-means clustering algorithm for customer grouping
    Used in Stage 1 (Decomposition) of K-NSGA-II
    """
    
    def __init__(self, k: int, max_iterations: int = 100, random_state: Optional[int] = None):
        """
        Initialize K-means clustering
        
        Args:
            k: Number of clusters (equals number of caregivers)
            max_iterations: Maximum iterations for convergence
            random_state: Seed for reproducibility
        """
        self.k = k
        self.max_iterations = max_iterations
        self.random_state = random_state
        
        if random_state is not None:
            random.seed(random_state)
        
        self.centroids: List[Tuple[float, float]] = []
        self.clusters: List[List[Customer]] = []
        self.labels: List[int] = []
    
    def _euclidean_distance(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between two points"""
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def _initialize_centroids(self, customers: List[Customer]) -> List[Tuple[float, float]]:
        """Initialize centroids using k-means++ method"""
        centroids = []
        
        # Choose first centroid randomly
        first = random.choice(customers)
        centroids.append((first.x, first.y))
        
        # Choose remaining centroids
        for _ in range(1, self.k):
            # Calculate distances to nearest centroid for each point
            distances = []
            for customer in customers:
                point = (customer.x, customer.y)
                min_dist = min(self._euclidean_distance(point, c) for c in centroids)
                distances.append(min_dist ** 2)  # Square for probability weighting
            
            # Choose next centroid with probability proportional to distance squared
            total = sum(distances)
            if total == 0:
                # All points are at centroid locations, choose randomly
                next_customer = random.choice(customers)
            else:
                probs = [d / total for d in distances]
                cumulative = 0
                r = random.random()
                next_customer = customers[-1]
                for i, p in enumerate(probs):
                    cumulative += p
                    if r <= cumulative:
                        next_customer = customers[i]
                        break
            
            centroids.append((next_customer.x, next_customer.y))
        
        return centroids
    
    def _assign_clusters(self, customers: List[Customer]) -> List[int]:
        """Assign each customer to the nearest centroid"""
        labels = []
        for customer in customers:
            point = (customer.x, customer.y)
            distances = [self._euclidean_distance(point, c) for c in self.centroids]
            labels.append(distances.index(min(distances)))
        return labels
    
    def _update_centroids(self, customers: List[Customer], labels: List[int]) -> List[Tuple[float, float]]:
        """Update centroids based on cluster assignments"""
        new_centroids = []
        
        for i in range(self.k):
            cluster_customers = [c for c, l in zip(customers, labels) if l == i]
            
            if cluster_customers:
                mean_x = sum(c.x for c in cluster_customers) / len(cluster_customers)
                mean_y = sum(c.y for c in cluster_customers) / len(cluster_customers)
                new_centroids.append((mean_x, mean_y))
            else:
                # Keep old centroid if cluster is empty
                new_centroids.append(self.centroids[i])
        
        return new_centroids
    
    def fit(self, customers: List[Customer]) -> 'KMeans':
        """
        Fit the K-means model to customer data
        
        Args:
            customers: List of Customer objects to cluster
        
        Returns:
            self
        """
        if len(customers) < self.k:
            raise ValueError(f"Not enough customers ({len(customers)}) for {self.k} clusters")
        
        # Initialize centroids
        self.centroids = self._initialize_centroids(customers)
        
        # Iterate until convergence or max iterations
        for iteration in range(self.max_iterations):
            # Assign customers to clusters
            new_labels = self._assign_clusters(customers)
            
            # Update centroids
            new_centroids = self._update_centroids(customers, new_labels)
            
            # Check for convergence
            if new_centroids == self.centroids:
                print(f"K-means converged at iteration {iteration + 1}")
                break
            
            self.centroids = new_centroids
            self.labels = new_labels
        
        # Build cluster lists
        self.clusters = [[] for _ in range(self.k)]
        for customer, label in zip(customers, self.labels):
            self.clusters[label].append(customer)
        
        return self
    
    def get_clusters(self) -> List[List[Customer]]:
        """Return the list of clusters"""
        return self.clusters
    
    def get_cluster_ids(self) -> List[List[int]]:
        """Return cluster customer IDs"""
        return [[c.id for c in cluster] for cluster in self.clusters]
    
    def predict(self, customer: Customer) -> int:
        """Predict cluster for a new customer"""
        point = (customer.x, customer.y)
        distances = [self._euclidean_distance(point, c) for c in self.centroids]
        return distances.index(min(distances))
