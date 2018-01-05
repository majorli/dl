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

    def distortion(self, clusters, centroids):
        """distortion
            
            Compute the distortion on clusters and centroids.

        Arguments:
            clusters -- clusters
            centroids -- centroids

        Returns:
            d -- distortion
        """
        d = 0.0
        P = self.X.copy()
        K = centroids.shape[1]
        m = self.X.shape[1]
        for k in range(K):
            P[:, np.squeeze(clusters == k)] -= centroids[:, [k]]
        d = np.sum(np.linalg.norm(P, axis=0, keepdims=True) ** 2) / m
        return d

    def compare_clusters(self, c1, c2):
        no_change = False
        if c1.shape == c2.shape:
            no_change = np.all(c1 == c2)
        return no_change

    def moving_distance(self, c1, c2):
        dist = np.sum(np.linalg.norm(c1 - c2, axis=0, keepdims=True))
        return dist

    def run(self, K=2, num_tries=1, max_steps=0, keep_distortions=False):
        """run K-means

            Run K-means "num_tries" times, at most "max_steps" iterations each time.

        Keyword Arguments:
            K -- Number of clusters (default: {2})
            num_tries -- Number of running times (default: {1})
            max_steps -- Maximum number of steps each running, 0 to iterate until converge (default: {0})
            keep_distortions -- True to store distortions for all steps, False to store only the last step (defalut: {False})

        Returns:
            results -- Lists of K-means results for all running times, each result is a map {clusters, distortions:[(step, distortion)], steps, last_moving_distance}
                        
        """
        m = self.X.shape[1]
        assert(K > 1 and K < m)

        results = []
        for t in range(num_tries):
            result = {
                    "clusters" : None,
                    "distortions" : [],
                    "steps" : 0,
                    "last_moving_distance" : 0.0
                    }
            # initial centroids and clusters
            centroids = self.choose_centroids(K)
            clusters = self.cluster_assignment(centroids)
            step = 0
            while max_steps == 0 or step < max_steps:
                # keep old centroids and clusters
                old_centroids = centroids.copy()
                old_clusters = clusters.copy()
                # move centroids and re-assign clusters
                centroids = self.move_centroids(clusters, centroids)
                if centroids.shape[1] < 2:
                    # bad try
                    result = None
                    break
                clusters = self.cluster_assignment(centroids)
                # compare
                no_change = self.compare_clusters(clusters, old_clusters)
                distance = self.moving_distance(centroids, old_centroids)
                if (no_change and distance < 1e-8):
                    # converged
                    result["clusters"] = clusters
                    result["distortions"].append((step, self.distortion(clusters, centroids)))
                    result["steps"] = step
                    result["last_moving_distance"] = distance
                    break
                else:
                    if keep_distortions or step == max_steps - 1:
                        result["distortions"].append((step, self.distortion(clusters, centroids)))
                step = step + 1
            # one try finished, store result
            results.append(result)

        return results

