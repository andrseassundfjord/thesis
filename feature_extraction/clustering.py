import numpy as np
from sklearn.cluster import KMeans, SpectralClustering
# Labeled cluster metrics
from sklearn.metrics import adjusted_rand_score, rand_score, adjusted_mutual_info_score, mutual_info_score, fowlkes_mallows_score
from sklearn.metrics import homogeneity_score, v_measure_score, completeness_score
# Cluster metrics
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import matplotlib.pyplot as plt
import torch
from load_dataset import get_dataloaders, VideoDataset, DataFrameTimeseriesDataset, LabelDataset
import torch.nn.functional as F
# Import models
from MVAE import MVAE
from MAE import MAE
from TimeseriesVAE import TimeseriesVAE
from VideoVAE import VideoVAE
from VideoAutoencoder import VideoAutoencoder
from VideoBert import VideoBERT
from VideoBERT_pretrained import VideoBERT_pretrained
import warnings
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

def run_cluster(X, y, K = 14, norm = True, savename = "model"):
    """
    Takes latent vectors and labels as input, returns accuracy of k means clustering
    """

    if norm:
        # Normalize and scale the data
        # X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
        X = (X - X.mean(dim=0)) / X.std(dim=0)

    with warnings.catch_warnings():
        #warnings.filterwarnings("ignore", message="Exited postprocessing with accuracies")
        #warnings.filterwarnings("ignore","Exited at iteration 383 with accuracies")
        #warnings.filterwarnings("ignore", "Graph is not fully connected, spectral embedding may not work as expected.")
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=K, random_state=42, n_init=10).fit(X)
        spectral = SpectralClustering(n_clusters=K, assign_labels='kmeans', n_init=10, random_state=42).fit(X)
        spectral_disc = SpectralClustering(n_clusters=K, assign_labels='discretize', random_state=42).fit(X)

        # Get the cluster assignments for each data sample
        kmeans_assignments = kmeans.labels_
        spectral_assignments = spectral.labels_
        spectral_disc_assignments = spectral_disc.labels_

        # Metrics
        # Adjusted rand index
        ari_kmeans = adjusted_rand_score(y, kmeans_assignments)
        ari_spectral = adjusted_rand_score(y, spectral_assignments)
        ari_disc = adjusted_rand_score(y, spectral_disc_assignments)
        # Rand index
        rand_kmeans = rand_score(y, kmeans_assignments)
        rand_spectral = rand_score(y, spectral_assignments)
        rand_disc = rand_score(y, spectral_disc_assignments)
        # Fowlkes Mallows
        fowlkes_mallows_kmeans = fowlkes_mallows_score(y, kmeans_assignments)
        fowlkes_mallows_spectral = fowlkes_mallows_score(y, spectral_assignments)
        fowlkes_mallows_disc = fowlkes_mallows_score(y, spectral_disc_assignments)
        # Completeness
        completeness_kmeans = completeness_score(y, kmeans_assignments)
        completeness_spectral = completeness_score(y, spectral_assignments)
        completeness_disc = completeness_score(y, spectral_disc_assignments)
        # Homogeneity
        homogeneity_kmeans = homogeneity_score(y, kmeans_assignments)
        homogeneity_spectral = homogeneity_score(y, spectral_assignments)
        homogeneity_disc = homogeneity_score(y, spectral_disc_assignments)
        # V measure
        v_measure_kmeans = v_measure_score(y, kmeans_assignments)
        v_measure_spectral = v_measure_score(y, spectral_assignments)
        v_measure_disc = v_measure_score(y, spectral_disc_assignments)
        # Silhouette score
        silhouette_kmeans = silhouette_score(X, kmeans_assignments)
        silhouette_spectral = silhouette_score(X, spectral_assignments)
        silhouette_disc = silhouette_score(X, spectral_disc_assignments)
        # CHS
        ch_kmeans = calinski_harabasz_score(X, kmeans_assignments)
        ch_spectral = calinski_harabasz_score(X, spectral_assignments)
        ch_disc = calinski_harabasz_score(X, spectral_disc_assignments)

    print("              K-means results      Spectral clustering results       Spectral disc results")
    print(f"ARI:          {ari_kmeans:.5f}                    {ari_spectral:.5f}                  {ari_disc:.5f}")
    print(f"Rand:         {rand_kmeans:.5f}                    {rand_spectral:.5f}                    {rand_disc:.5f}")
    print(f"FM:           {fowlkes_mallows_kmeans:.5f}                    {fowlkes_mallows_spectral:.5f}                    {fowlkes_mallows_disc:.5f}")
    print(f"Completeness: {completeness_kmeans:.5f}                    {completeness_spectral:.5f}                    {completeness_disc:.5f}")
    print(f"Homogeneity:  {homogeneity_kmeans:.5f}                    {homogeneity_spectral:.5f}                    {homogeneity_disc:.5f}")
    print(f"V measure:    {v_measure_kmeans:.5f}                    {v_measure_spectral:.5f}                    {v_measure_disc:.5f}")
    print(f"Silhouette:   {silhouette_kmeans:.5f}                    {silhouette_spectral:.5f}                    {silhouette_disc:.5f}")
    print(f"CH Score:     {ch_kmeans:.5f}                    {ch_spectral:.5f}                    {ch_disc:.5f}")


    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X)

    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(X_pca[:,0], X_pca[:,1], X_pca[:,2], c=kmeans_assignments)
    legend = ax.legend(*scatter.legend_elements(), loc="center left", bbox_to_anchor=(1.1, 0.5), title="Class labels")
    ax.add_artist(legend)
    if norm:
        plt.savefig(f"results/clustering/plots_{savename}_norm/{savename}_kmeans_k_{K}", bbox_inches='tight')
    else:
        plt.savefig(f"results/clustering/plots_{savename}/{savename}_kmeans_k_{K}", bbox_inches='tight')
    plt.close(fig)
    plt.clf()
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(X_pca[:,0], X_pca[:,1], X_pca[:,2], c=spectral_assignments)
    legend = ax.legend(*scatter.legend_elements(), loc="center left", bbox_to_anchor=(1.1, 0.5), title="Class labels")
    ax.add_artist(legend)
    if norm:
        plt.savefig(f"results/clustering/plots_{savename}_norm/{savename}_spectral_k_{K}", bbox_inches='tight')
    else:
        plt.savefig(f"results/clustering/plots_{savename}/{savename}_spectral_k_{K}", bbox_inches='tight')
    plt.clf()
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(X_pca[:,0], X_pca[:,1], X_pca[:,2], c=spectral_disc_assignments)
    legend = ax.legend(*scatter.legend_elements(), loc="center left", bbox_to_anchor=(1.1, 0.5), title="Class labels")
    ax.add_artist(legend)
    if norm:
        plt.savefig(f"results/clustering/plots_{savename}_norm/{savename}_spectral_disc_k_{K}", bbox_inches='tight')
    else: 
        plt.savefig(f"results/clustering/plots_{savename}/{savename}_spectral_disc_k_{K}", bbox_inches='tight')
    plt.close(fig)

def prep_timeseries(timeseries):
    masks = []
    for idx, t in enumerate(timeseries):
        nan_mask = torch.isnan(t)
        # Replace NaN values with 0 using boolean masking
        t[nan_mask] = 0.0
        missing_mask = t.eq(-999)
        # Replace -999 with -1
        t[missing_mask] = 0.0
        mask = nan_mask | missing_mask
        masks.append(mask)
        # If features are continous
        if idx in [0, 3, 5]:
            timeseries[idx] = F.normalize(t, p=1, dim=1)
    return timeseries

<<<<<<< HEAD
def get_latent(model, latent_dim, hidden_layers, split_size = 1):
=======
def get_latent(model, latent_dim, hidden_layers):
>>>>>>> 3d4166e88f8c6bfbb231d645829b909bac5bfa79
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Define the model architecture
<<<<<<< HEAD
    model = model(input_dims= [(64 // split_size, 128, 128, 3), (200 // split_size, 352)], latent_dim=latent_dim, 
                    hidden_layers = hidden_layers, dropout = 0.2).to(device)
=======
    model = model(input_dims= [(64, 128, 128, 3), (200, 352)], latent_dim=latent_dim, 
                    hidden_layers = hidden_layers, dropout= 0.2).to(device)

>>>>>>> 3d4166e88f8c6bfbb231d645829b909bac5bfa79
    model_name = model.__class__.__name__

    # Load the model state
    if split_size > 1:
        model.load_state_dict(torch.load(f'augmented_models/{model_name}_state.pth'))
    else:
        model.load_state_dict(torch.load(f'models/{model_name}_state.pth'))

    # Set the model to evaluation mode

    video_train_loader, video_test_loader, timeseries_train_loader, timeseries_test_loader, label_train, label_test, risk_train, risk_test = get_dataloaders(
                                                '/work5/share/NEDO/nedo-2019/data/processed_rosbags_topickles/fixed_pickles', 
                                                "/work5/share/NEDO/nedo-2019/data/01_driving_data/movie", 
                                                train_ratio = 0.7,
                                                batch_size = 32, 
                                                save = False,
                                                load = True
                                            )
    latents = []
    labels = []
    
    model.eval()
    for video, timeseries, label in zip(video_test_loader, timeseries_test_loader, label_test):
        video_slices = torch.split(video, video.size(2) // split_size, dim=2)
        timeseries_slices = [[] for _ in range(split_size)]
        for t in timeseries:
            split_t = torch.split(t, t.size(1) // split_size, dim = 1)
            for idx, split in enumerate(split_t):
                timeseries_slices[idx].append(split)
        with torch.no_grad():
            for i in range(split_size):
                video = video_slices[i]
                timeseries = timeseries_slices[i]
                if "Video" in model_name:
                    video = video.to(device)
                    if "VAE" in model_name:
                        recon_video, kl_divergence, latent_representation, mus = model(video)
                        latent = mus.to("cpu")
                    else: 
                        recon_video, latent_representation = model(video)
                        latent = latent_representation.to("cpu")
                elif "M" in model_name:
                    video = video.to(device)
                    timeseries = [t.to(device) for t in timeseries]
                    timeseries = prep_timeseries(timeseries)
                    recon_video, recon_timeseries, kl_divergence, latent_representation, mus = model([video, timeseries])
                    latent = mus.to("cpu")
                else:
                    timeseries = [t.to(device) for t in timeseries]
                    timeseries = prep_timeseries(timeseries)
                    recon_timeseries, kl_divergence, latent_representation, mus = model(timeseries)
                    latent = mus.to("cpu")
                
                latents.append(latent)
                labels.append(label)

    latents = torch.cat(latents, dim = 0)
    flattened_labels = []
    for batch in labels:
        for label in batch:
            flattened_labels.append(label)
    return latents, flattened_labels, model_name

<<<<<<< HEAD
def run_clustering(model, latent_dim, hidden_layers, split_size = 1):
    print("Started")
    latents, labels, model_name = get_latent(model, latent_dim, hidden_layers, split_size)
=======
def run_cluster(model, latent_dim, hidden_layers):
    print("Started")
    latents, labels, model_name = get_latent(model, latent_dim, hidden_layers)
>>>>>>> 3d4166e88f8c6bfbb231d645829b909bac5bfa79
    print("Latent representations ready")
    for i in range(2, 15):
        print("\nCluster ", i, flush = True)
        run_cluster(latents, labels, K = i, norm = False, savename = model_name)
    print("Finished cluster for ", model_name)

if __name__ == "__main__":
    print("Run Cluster")
    torch.manual_seed(42)
    np.random.seed(42)
<<<<<<< HEAD
    latent_dim = 512
    video_hidden_shape = [128, 256, 512, 512]
    timeseries_hidden_dim = 1024
    timeseries_num_layers = 3
    hidden_layers = [video_hidden_shape, timeseries_hidden_dim, timeseries_num_layers]
    run_clustering(VideoAutoencoder, latent_dim, hidden_layers, split_size = 4)
=======
    run_cluster(VideoAutoencoder)
>>>>>>> 3d4166e88f8c6bfbb231d645829b909bac5bfa79
