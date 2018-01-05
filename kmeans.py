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

    def run(self, K=2, num_tries=1, max_steps=0, print_distortions=False):
        """run K-means

            Run K-means "num_tries" times, at most "max_steps" iterations each time.

        Keyword Arguments:
            K -- Number of clusters (default: {2})
            num_tries -- Number of running times (default: {1})
            max_steps -- Maximum number of steps each running, 0 to iterate until converge (default: {0})
            print_distortions -- True to print distortions for all steps (defalut: {False})

        Returns:
            results -- Map of K-means results:
                        {
                          "clusters" : (num_tries * num_examples) matrix, each row is the clusters given out in each try,
                          "distortions" : (num_tries * 1) matrix, each element is the final distortion given out in each try,
                          "Ks" : (num_tries * 1) matrix, each element is the number of clusters given out in each try,
                          "last_moving_distances" : (num_tries * 1) matrix, each element is the total centroids moving distance of last moving in each try,
                          "smallest_distortion" : the smallest distortion among all tries,
                          "best_tries" : order-1 array of indice of best tries, i.e. the smallest final distortion
                        }
        """
        m = self.X.shape[1]
        assert(K > 1 and K < m)

        results = {}
        Clusters = np.zeros((0, m))     # append clusters of each try by np.vstack((Clusters, clusters))
        distortions = []                # append distortion and change to matix by np.array(distortions).reshape(-1, 1)
        Ks = []                         # append centroids.shape[1] and change to matix by np.array(Ks).reshape(-1, 1)
        last_moving_distances = []      # append last_moving_distance and change to matix by np.array(last_moving_distances).reshape(-1, 1)

        for t in range(num_tries):
            # initial centroids and clusters
            centroids = self.choose_centroids(K)
            clusters = self.cluster_assignment(centroids)
            step = 0
            bad_try = False
            distance = 0.0

            if print_distortions:
                print("Try " + str(t) + ":")
            while max_steps == 0 or step < max_steps:
                if print_distortions:
                    print("  Step " + str(step) + " distortion =" + str(self.distortion(clusters, centroids)))
                # keep old centroids and clusters
                old_centroids = centroids.copy()
                old_clusters = clusters.copy()
                # move centroids and re-assign clusters
                centroids = self.move_centroids(clusters, centroids)
                if centroids.shape[1] < 2:
                    # bad try, skip
                    bad_try = True
                    break
                clusters = self.cluster_assignment(centroids)
                # compare
                no_change = self.compare_clusters(clusters, old_clusters)
                distance = self.moving_distance(centroids, old_centroids)
                step = step + 1
                if no_change and distance < 1e-8:
                    # converged
                    break
            ## end of while

            # one try finished, store result
            if bad_try == False:
                Clusters = np.vstack((Clusters, clusters))
                distortion = self.distortion(clusters, centroids)
                distortions.append(distortion)
                Ks.append(centroids.shape[1])
                last_moving_distances.append(distance)
        ## end of for

        # store all results
        results["Clusters"] = Clusters
        results["distortions"] = np.array(distortions).reshape(-1, 1)
        results["Ks"] = np.array(Ks).reshape(-1, 1)
        results["last_moving_distances"] = np.array(last_moving_distances).reshape(-1, 1)
        min_distortion = np.min(results["distortions"])
        results["smallest_distortion"] = min_distortion
        results["best_tries"] = np.arange(Clusters.shape[0]).reshape(-1, 1)[(results["distortions"] == min_distortion).squeeze(), :].squeeze()

        return results

    def best_clusters(results):
        """retreive best clusters from results

            Static methods, invoke by "km.Kmeans.best_clusters(results)"

        Arguments:
            results -- Kmeans model running results

        Returns:
            best_clusters -- matrix of clusters from the best tries
        """
        return results["Clusters"][results["best_tries"], :]

