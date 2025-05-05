import numpy as np
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import cdist

def simple_cure(X, n_clusters=5, n_representatives=5, shrink_factor=0.2):
    sample_size = min(300, len(X))
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(len(X), sample_size, replace=False)
    X_sample = X[sample_idx]

    agglom = AgglomerativeClustering(n_clusters=n_clusters)
    sample_labels = agglom.fit_predict(X_sample)

    representatives = []
    for cluster_id in range(n_clusters):
        cluster_points = X_sample[sample_labels == cluster_id]
        if len(cluster_points) == 0:
            continue
        centroid = np.mean(cluster_points, axis=0)
        dists = cdist(cluster_points, [centroid])
        rep_idx = np.argsort(dists.ravel())[::-1][:n_representatives]
        reps = cluster_points[rep_idx]
        reps_shrunk = centroid + shrink_factor * (reps - centroid)
        representatives.extend(reps_shrunk)

    representatives = np.array(representatives)
    dist_to_reps = cdist(X, representatives)
    closest_rep_idx = np.argmin(dist_to_reps, axis=1)
    cluster_labels = closest_rep_idx // n_representatives
    return cluster_labels
