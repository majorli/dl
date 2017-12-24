# K-means model

import numpy as np

class Kmeans:
    """K-means model"""

    def __init__(self):
        self.X = None
        
    def feed_data(self, X):
        self.X = X

    def is_ready(self):
        return self.X is not None

    def choose_centroids(self, num_clusters):
        assert(self.is_ready())

        m = self.X.shape[1]
        permutation = np.random.permutation(m)

        return self.X[:, permutation[range(num_clusters)]]
        
    def cluster_assignment(self, centroids):
        assert(self.is_ready())

        m = self.X.shape[1]
        K = centroids.shape[1]
        norms = np.zeros((K, m))

        for k in range(K):
            norms[k, :] = np.linalg.norm(self.X - centroids[:, [k]], axis = 0, keepdims = True)

        return np.argmin(norms, axis=0)

    def move_centroids(self, clusters, centroids):
        assert(self.is_ready())

        new_centroids = np.zeros(centroids.shape)
        masks = []
        m = self.X.shape[1]
        K = centroids.shape[1]

        for k in range(K):
            cluster = self.X[:, clusters == k]
            if cluster.shape[1] == 0:
                continue
            masks.append(k)
            new_centroids[:, k] = np.mean(cluster, axis = 1)

        return new_centroids[:, masks]

    def run(self, num_clusters = 2, num_tries = 1, max_iters = 100, distortions = False):
        assert(self.is_ready())

        # TODO: run k-means at most 'max_iters' iterations, until converge (np.sum(new_centroids - old_centroids) <= 1e-8)

        Distortions = []
        Clusters = []

        return Clusters, Distortions

