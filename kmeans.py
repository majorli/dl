# K-means model

import numpy as np

class Kmeans:
    """K-means model"""

    def __init__(self, X):
        """constructor

            Create a new K-means model with given examples

        Arguments:
            X -- Data set
        """
        self.X = X
        
    def choose_centroids(self, K):
        """choose centroids
        
            Randomly choose K examples to be centroids

        Arguments:
            K -- Number of centroids, must be less than the number of examples

        Returns:
            centroids -- Randomly chosen centroids
        """
        m = self.X.shape[1]
        assert(K < m and K > 1)
        pm = np.random.permutation(m)

        return self.X[:, pm[range(K)]]
        
    def cluster_assignment(self, centroids):
        """cluster assignment

            Find the nearest controid and assign cluster of this centroid to each example
        
        Arguments:
            centroids -- Array of centroids

        Returns:
            clusters -- Array of cluster number of each example
        """
        m = self.X.shape[1]
        K = centroids.shape[1]
        assert(K < m and K > 1)
        norms = np.zeros((K, m))

        for k in range(K):
            norms[k, :] = np.linalg.norm(self.X - centroids[:, [k]], axis = 0, keepdims = True)

        return np.argmin(norms, axis=0)

    def move_centroids(self, clusters, centroids):
        """move centroids

            Move centroids to the mass centers of their clusters.
            Centroid with empty cluster will be ignored silently, return an array of new centroids without it.

        Arguments:
            clusters -- Current assigned clusters according to current positions of centroids
            centroids -- Current centroids

        Returns:
            centroids -- Array of new centroids located at the mass centers of their clusters
        """
        m = self.X.shape[1]
        K = centroids.shape[1]
        assert(K < m and K > 1)
        new_centroids = np.zeros(centroids.shape)
        masks = []

        for k in range(K):
            cluster = self.X[:, clusters == k]
            if cluster.shape[1] == 0:
                continue
            masks.append(k)
            new_centroids[:, k] = np.mean(cluster, axis = 1)

        return new_centroids[:, masks]

    def run(self, K = 2, num_tries = 1, max_iters = 100, distortions = False):

        # TODO: run k-means at most 'max_iters' iterations, until converge (np.sum(new_centroids - old_centroids) <= 1e-8)

        Distortions = []
        Clusters = []

        return Clusters, Distortions

