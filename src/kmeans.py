"""
K-means Clustering Implementation for HHC Problem
Divides patients into K clusters based on geographical and temporal features
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
from src.data_parser import ProblemInstance, Customer, get_customer_features


@dataclass
class Cluster:
    """Represents a cluster of customers"""
    id: int
    centroid: np.ndarray
    customer_indices: List[int]  # Indices into the customers list (0-indexed)
    
    @property
    def size(self) -> int:
        return len(self.customer_indices)
    
    def __repr__(self):
        return f"Cluster(id={self.id}, size={self.size}, centroid={self.centroid[:2]})"


class KMeansClustering:
    """
    K-means clustering for dividing patients among caregivers.
    
    The algorithm divides the set of patients into K similar clusters,
    where K corresponds to the number of caregivers. Patients in the same
    cluster have similar characteristics (geographical coordinates and
    visiting-time preferences).
    """
    
    def __init__(self, 
                 n_clusters: int,
                 max_iterations: int = 100,
                 tolerance: float = 1e-4,
                 random_state: int = None,
                 use_time_features: bool = True):
        """
        Initialize K-means clustering.
        
        Args:
            n_clusters: Number of clusters (K = number of caregivers)
            max_iterations: Maximum number of iterations
            tolerance: Convergence tolerance (WCSS change threshold)
            random_state: Random seed for reproducibility
            use_time_features: Include time window features in clustering
        """
        self.n_clusters = n_clusters
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.random_state = random_state
        self.use_time_features = use_time_features
        
        self.centroids = None
        self.clusters = None
        self.wcss_history = []
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def _initialize_centroids(self, features: np.ndarray) -> np.ndarray:
        """Initialize centroids using K-means++ method for better convergence"""
        n_samples = features.shape[0]
        centroids = []
        
        # Choose first centroid randomly
        idx = np.random.randint(0, n_samples)
        centroids.append(features[idx])
        
        # Choose remaining centroids
        for _ in range(1, self.n_clusters):
            # Calculate distances to nearest centroid
            distances = np.zeros(n_samples)
            for i in range(n_samples):
                min_dist = float('inf')
                for c in centroids:
                    dist = np.linalg.norm(features[i] - c)
                    min_dist = min(min_dist, dist)
                distances[i] = min_dist ** 2
            
            # Choose next centroid with probability proportional to distance
            probabilities = distances / distances.sum()
            idx = np.random.choice(n_samples, p=probabilities)
            centroids.append(features[idx])
        
        return np.array(centroids)
    
    def _assign_clusters(self, features: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """Assign each point to the nearest centroid"""
        n_samples = features.shape[0]
        assignments = np.zeros(n_samples, dtype=int)
        
        for i in range(n_samples):
            min_dist = float('inf')
            min_cluster = 0
            
            for j, centroid in enumerate(centroids):
                dist = np.linalg.norm(features[i] - centroid)
                if dist < min_dist:
                    min_dist = dist
                    min_cluster = j
            
            assignments[i] = min_cluster
        
        return assignments
    
    def _update_centroids(self, features: np.ndarray, assignments: np.ndarray) -> np.ndarray:
        """Update centroids as mean of assigned points"""
        new_centroids = np.zeros((self.n_clusters, features.shape[1]))
        
        for j in range(self.n_clusters):
            cluster_points = features[assignments == j]
            if len(cluster_points) > 0:
                new_centroids[j] = cluster_points.mean(axis=0)
            else:
                # If cluster is empty, reinitialize randomly
                new_centroids[j] = features[np.random.randint(0, len(features))]
        
        return new_centroids
    
    def _calculate_wcss(self, features: np.ndarray, assignments: np.ndarray, 
                        centroids: np.ndarray) -> float:
        """
        Calculate Within-Cluster Sum of Squares (WCSS).
        WCSS = sum of squared distances between each point and its centroid.
        """
        wcss = 0.0
        
        for i in range(len(features)):
            cluster_idx = assignments[i]
            wcss += np.linalg.norm(features[i] - centroids[cluster_idx]) ** 2
        
        return wcss
    
    def fit(self, instance: ProblemInstance) -> List[Cluster]:
        """
        Fit K-means clustering to the problem instance.
        
        Args:
            instance: Problem instance with customer data
        
        Returns:
            List of Cluster objects
        """
        # Extract features
        features = get_customer_features(instance, include_time=self.use_time_features)
        
        # Initialize centroids
        self.centroids = self._initialize_centroids(features)
        
        # Iterative optimization
        self.wcss_history = []
        prev_wcss = float('inf')
        
        for iteration in range(self.max_iterations):
            # Assign clusters
            assignments = self._assign_clusters(features, self.centroids)
            
            # Update centroids
            self.centroids = self._update_centroids(features, assignments)
            
            # Calculate WCSS
            wcss = self._calculate_wcss(features, assignments, self.centroids)
            self.wcss_history.append(wcss)
            
            # Check convergence
            if abs(prev_wcss - wcss) < self.tolerance:
                print(f"K-means converged at iteration {iteration + 1}")
                break
            
            prev_wcss = wcss
        
        # Create cluster objects
        self.clusters = []
        for j in range(self.n_clusters):
            customer_indices = np.where(assignments == j)[0].tolist()
            cluster = Cluster(
                id=j,
                centroid=self.centroids[j],
                customer_indices=customer_indices
            )
            self.clusters.append(cluster)
        
        return self.clusters
    
    def get_cluster_customers(self, cluster_id: int, instance: ProblemInstance) -> List[Customer]:
        """Get the Customer objects belonging to a cluster"""
        if self.clusters is None:
            raise ValueError("Must fit clustering first")
        
        cluster = self.clusters[cluster_id]
        return [instance.customers[i] for i in cluster.customer_indices]
    
    def get_cluster_customer_ids(self, cluster_id: int) -> List[int]:
        """Get customer IDs (1-indexed) belonging to a cluster"""
        if self.clusters is None:
            raise ValueError("Must fit clustering first")
        
        cluster = self.clusters[cluster_id]
        # Convert 0-indexed to 1-indexed customer IDs
        return [i + 1 for i in cluster.customer_indices]
    
    def get_balanced_clusters(self, instance: ProblemInstance, 
                              max_imbalance: float = 0.3) -> List[Cluster]:
        """
        Rebalance clusters to ensure fair distribution of patients.
        
        Args:
            instance: Problem instance
            max_imbalance: Maximum allowed imbalance ratio
        
        Returns:
            Rebalanced clusters
        """
        if self.clusters is None:
            self.fit(instance)
        
        total_customers = instance.num_customers
        target_size = total_customers // self.n_clusters
        max_size = int(target_size * (1 + max_imbalance))
        min_size = int(target_size * (1 - max_imbalance))
        
        features = get_customer_features(instance, include_time=self.use_time_features)
        
        # Check if rebalancing is needed
        needs_rebalancing = False
        for cluster in self.clusters:
            if cluster.size > max_size or cluster.size < min_size:
                needs_rebalancing = True
                break
        
        if not needs_rebalancing:
            return self.clusters
        
        # Rebalance by moving customers from large to small clusters
        for _ in range(100):  # Max rebalancing iterations
            # Find largest and smallest clusters
            sorted_clusters = sorted(self.clusters, key=lambda c: c.size, reverse=True)
            largest = sorted_clusters[0]
            smallest = sorted_clusters[-1]
            
            if largest.size - smallest.size <= 1:
                break
            
            # Move the customer closest to smallest cluster's centroid
            best_idx = None
            best_dist = float('inf')
            
            for idx in largest.customer_indices:
                dist = np.linalg.norm(features[idx] - smallest.centroid)
                if dist < best_dist:
                    best_dist = dist
                    best_idx = idx
            
            if best_idx is not None:
                largest.customer_indices.remove(best_idx)
                smallest.customer_indices.append(best_idx)
                
                # Update centroids
                if largest.customer_indices:
                    largest.centroid = features[largest.customer_indices].mean(axis=0)
                if smallest.customer_indices:
                    smallest.centroid = features[smallest.customer_indices].mean(axis=0)
        
        return self.clusters
    
    def print_cluster_summary(self, instance: ProblemInstance):
        """Print summary of clustering results"""
        if self.clusters is None:
            raise ValueError("Must fit clustering first")
        
        print("\n" + "=" * 50)
        print("K-means Clustering Summary")
        print("=" * 50)
        print(f"Number of clusters: {self.n_clusters}")
        print(f"Total customers: {instance.num_customers}")
        print(f"Final WCSS: {self.wcss_history[-1]:.4f}")
        print(f"Iterations: {len(self.wcss_history)}")
        
        print("\nCluster Details:")
        for cluster in self.clusters:
            customers = self.get_cluster_customers(cluster.id, instance)
            total_demand = sum(c.demand for c in customers)
            avg_ready = np.mean([c.ready_time for c in customers]) if customers else 0
            avg_due = np.mean([c.due_date for c in customers]) if customers else 0
            
            print(f"\n  Cluster {cluster.id}:")
            print(f"    Size: {cluster.size} customers")
            print(f"    Customer IDs: {self.get_cluster_customer_ids(cluster.id)}")
            print(f"    Total demand: {total_demand}")
            print(f"    Avg time window: [{avg_ready:.0f}, {avg_due:.0f}]")
            print(f"    Centroid (x,y): ({cluster.centroid[0]:.2f}, {cluster.centroid[1]:.2f})")


if __name__ == "__main__":
    # Test K-means clustering
    from src.data_parser import load_instance
    
    print("Testing K-means Clustering")
    print("=" * 50)
    
    # Load test instance
    instance = load_instance("C101.25")
    print(f"\nInstance: {instance.name}")
    print(f"Customers: {instance.num_customers}")
    print(f"Caregivers: {instance.num_vehicles}")
    
    # Perform clustering
    n_clusters = instance.num_vehicles  # One cluster per caregiver
    kmeans = KMeansClustering(n_clusters=n_clusters, random_state=42)
    
    clusters = kmeans.fit(instance)
    kmeans.print_cluster_summary(instance)
    
    # Test balanced clustering
    print("\n\nTesting balanced clustering...")
    balanced_clusters = kmeans.get_balanced_clusters(instance)
    
    print("\nBalanced cluster sizes:")
    for cluster in balanced_clusters:
        print(f"  Cluster {cluster.id}: {cluster.size} customers")
