import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import matplotlib.pyplot as plt

def run_k_means(X, y, K = 14, norm = False, savename = "results/kmeans"):
    """
    Takes latent vectors and labels as input, returns accuracy of k means clustering
    """

    if norm:
        # Normalize and scale the data
        X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

    # Perform K-means clustering
    kmeans = KMeans(n_clusters=K, random_state=0).fit(X)

    # Get the cluster assignments for each data sample
    cluster_assignments = kmeans.labels_

    # Measure the Adjusted Rand Index (ARI)
    ari = adjusted_rand_score(y, cluster_assignments)

    print("ari: ", ari)

    fig, ax = plt.subplots()
    scatter = ax.scatter(latent_vectors[:,0], latent_vectors[:,1], c=cluster_assignments)
    legend = ax.legend(*scatter.legend_elements(), loc="upper right", title="Clusters")
    ax.add_artist(legend)
    plt.savefig(savename)
