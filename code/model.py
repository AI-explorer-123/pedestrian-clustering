import numpy as np
from tqdm import trange


def lp_norm(x, p):
    abs_powers = np.abs(x) ** p
    sum_abs_powers = np.sum(abs_powers)
    norm = sum_abs_powers ** (1 / p)
    return norm


class KMeans:
    def __init__(self, k, seed):
        self.n_clusters = k
        self.centroids = None
        self.labels = None
        np.random.seed(seed)

    def _init_para(self, X):
        n_samples, _ = X.shape
        self._init_centroids(X)
        self.labels = np.random.randint(self.n_clusters, size=n_samples)

    def _init_centroids(self, X):
        n_samples, _ = X.shape
        index = np.random.randint(n_samples, size=self.n_clusters)
        self.centroids = X[index]

    def _generate_label(self, sample, p):
        dists = []
        for centroid in self.centroids:
            dist = lp_norm(sample-centroid, p)
            dists.append(dist)
        return np.argmax(np.array(dists))

    def _update_centroids(self, X, labels):
        for i in range(self.n_clusters):
            mask = np.where(labels == i, True, False)
            if X[mask].shape[0] != 0:
                self.centroids[i] = np.mean(X[mask], axis=0)

    def fit(self, X, p=2, max_iters=300):
        n_samples, _ = X.shape
        self._init_para(X)
        for _ in range(max_iters):
            new_labels = np.empty(shape=n_samples)
            for id, sample in enumerate(X):
                new_label = self._generate_label(sample, p)
                new_labels[id] = new_label

            if (new_labels != self.labels).any():
                self.labels = new_labels
                self._update_centroids(X, self.labels)
            else:
                return self.labels
        return self.labels


class DBSCAN:
    def __init__(self, eps=2.5, min_samples=5, p=2):
        self.eps = eps
        self.min_samples = min_samples
        self.p = p
        self.labels = None

    def _region_query(self, X, point):
        n_samples, _ = X.shape
        neighbors = []
        for i in range(n_samples):
            if lp_norm(X[i] - point, self.p) < self.eps:
                neighbors.append(i)
        return neighbors

    def _expand_cluster(self, X, labels, point_idx, cluster_id):
        neighbors = self._region_query(X, X[point_idx])
        if len(neighbors) < self.min_samples:
            labels[point_idx] = -1
            return False
        else:
            labels[point_idx] = cluster_id
            i = 0
            while i < len(neighbors):
                neighbor_idx = neighbors[i]
                if labels[neighbor_idx] == 0:
                    labels[neighbor_idx] = cluster_id
                    new_neighbors = self._region_query(X, X[neighbor_idx])
                    if len(new_neighbors) >= self.min_samples:
                        neighbors = neighbors + new_neighbors
                elif labels[neighbor_idx] == -1:
                    labels[neighbor_idx] = cluster_id
                i += 1
            return True

    def fit(self, X):
        n_samples, _ = X.shape
        self.labels = np.zeros(n_samples)
        cluster_id = 0

        for point_idx in range(n_samples):
            if self.labels[point_idx] == 0:
                if self._expand_cluster(X, self.labels, point_idx, cluster_id + 1):
                    cluster_id += 1
        return self.labels
