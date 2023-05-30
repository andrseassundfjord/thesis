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
from TimeBERT import TimeBERT
from MidMVAE import MidMVAE
from MAE import MAE
from TimeseriesVAE import TimeseriesVAE
from VideoVAE import VideoVAE
from VideoAutoencoder import VideoAutoencoder
from VideoBert import VideoBERT
from VideoBERT_pretrained import VideoBERT_pretrained
import warnings
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE

def run_cluster(X, y, K = 14, norm = True, savename = "model"):
    """
    Takes latent vectors and labels as input, returns accuracy of k means clustering
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Exited postprocessing with accuracies")
        #warnings.filterwarnings("ignore", "Graph is not fully connected, spectral embedding may not work as expected.")
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=K, random_state=42, n_init=10).fit(X)
        spectral = SpectralClustering(n_clusters=K, assign_labels='kmeans', n_init=10, random_state=42).fit(X)
        gmm = GaussianMixture(n_components=K)  # num_clusters is the desired number of clusters
        gmm.fit(X)  # X is your input data of shape (batch_size, latent_dim)

        # Obtain cluster assignments for the input data

        # Get the cluster assignments for each data sample
        kmeans_assignments = kmeans.labels_
        spectral_assignments = spectral.labels_
        gmm_labels = gmm.predict(X)

        # Metrics
        # Adjusted rand index
            
        ari_kmeans = adjusted_rand_score(y, kmeans_assignments)
        ari_spectral = 0
        if np.max(spectral_assignments) > 1:
            ari_spectral = adjusted_rand_score(y, spectral_assignments)
        ari_gmm = adjusted_rand_score(y, gmm_labels)
        # Rand index
        rand_kmeans = rand_score(y, kmeans_assignments)
        rand_spectral = 0
        if np.max(spectral_assignments) > 1:
            rand_spectral = rand_score(y, spectral_assignments)
        rand_gmm = rand_score(y, gmm_labels)
        # Fowlkes Mallows
        fowlkes_mallows_kmeans = fowlkes_mallows_score(y, kmeans_assignments)
        fowlkes_mallows_spectral = 0
        if np.max(spectral_assignments) > 1:
            fowlkes_mallows_spectral = fowlkes_mallows_score(y, spectral_assignments)
        fowlkes_mallows_gmm = fowlkes_mallows_score(y, gmm_labels)
        # Completeness
        completeness_kmeans = completeness_score(y, kmeans_assignments)
        completeness_spectral = 0
        if np.max(spectral_assignments) > 1:
            completeness_spectral = completeness_score(y, spectral_assignments)
        completeness_gmm = completeness_score(y, gmm_labels)
        # Homogeneity
        homogeneity_kmeans = homogeneity_score(y, kmeans_assignments)
        homogeneity_spectral = 0
        if np.max(spectral_assignments) > 1:
            homogeneity_spectral = homogeneity_score(y, spectral_assignments)
        homogeneity_gmm = homogeneity_score(y, gmm_labels)
        # V measure
        v_measure_kmeans = v_measure_score(y, kmeans_assignments)
        v_measure_spectral = 0
        if np.max(spectral_assignments) > 1:
            v_measure_spectral = v_measure_score(y, spectral_assignments)
        v_measure_gmm = v_measure_score(y, gmm_labels)
        # Silhouette score
        silhouette_kmeans = silhouette_score(X, kmeans_assignments)
        silhouette_spectral = 0
        if np.max(spectral_assignments) > 1:
            silhouette_spectral = silhouette_score(X, spectral_assignments)
        silhouette_gmm = silhouette_score(X, gmm_labels)
        # CHS
        ch_kmeans = calinski_harabasz_score(X, kmeans_assignments)
        ch_spectral = 0
        if np.max(spectral_assignments) > 1:
            ch_spectral = calinski_harabasz_score(X, spectral_assignments)
        ch_gmm = calinski_harabasz_score(X, gmm_labels)

    print("              K-means results      Spectral clustering results       GMM results")
    print(f"ARI:          {ari_kmeans:.5f}                    {ari_spectral:.5f}                  {ari_gmm:.5f}")
    print(f"Rand:         {rand_kmeans:.5f}                    {rand_spectral:.5f}                    {rand_gmm:.5f}")
    print(f"FM:           {fowlkes_mallows_kmeans:.5f}                    {fowlkes_mallows_spectral:.5f}                    {fowlkes_mallows_gmm:.5f}")
    print(f"Completeness: {completeness_kmeans:.5f}                    {completeness_spectral:.5f}                    {completeness_gmm:.5f}")
    print(f"Homogeneity:  {homogeneity_kmeans:.5f}                    {homogeneity_spectral:.5f}                    {homogeneity_gmm:.5f}")
    print(f"V measure:    {v_measure_kmeans:.5f}                    {v_measure_spectral:.5f}                    {v_measure_gmm:.5f}")
    print(f"Silhouette:   {silhouette_kmeans:.5f}                    {silhouette_spectral:.5f}                    {silhouette_gmm:.5f}")
    print(f"CH Score:     {ch_kmeans:.5f}                 {ch_spectral:.5f}                    {ch_gmm:.5f}")

    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X)
    X_tsne = TSNE(n_components=3, learning_rate='auto',
                   init='random', perplexity=3).fit_transform(X)

    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(X_pca[:,0], X_pca[:,1], X_pca[:,2], c=kmeans_assignments)
    legend = ax.legend(*scatter.legend_elements(), loc="center left", bbox_to_anchor=(1.1, 0.5), title="Class labels")
    ax.add_artist(legend)
    plt.savefig(f"results/clustering/plots_{savename}/{savename}_kmeans_k_{K}", bbox_inches='tight')
    plt.close(fig)
    plt.clf()
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(X_pca[:,0], X_pca[:,1], X_pca[:,2], c=spectral_assignments)
    legend = ax.legend(*scatter.legend_elements(), loc="center left", bbox_to_anchor=(1.1, 0.5), title="Class labels")
    ax.add_artist(legend)
    plt.savefig(f"results/clustering/plots_{savename}/{savename}_spectral_k_{K}", bbox_inches='tight')
    plt.clf()
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(X_pca[:,0], X_pca[:,1], X_pca[:,2], c=gmm_labels)
    legend = ax.legend(*scatter.legend_elements(), loc="center left", bbox_to_anchor=(1.1, 0.5), title="Class labels")
    ax.add_artist(legend)
    plt.savefig(f"results/clustering/plots_{savename}/{savename}_gmm_k_{K}", bbox_inches='tight')
    plt.close(fig)
    if K == 14:
        # t-SNE
        plt.clf()
        fig = plt.figure(figsize=(10, 4))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(X_tsne[:,0], X_tsne[:,1], X_tsne[:,2], c=kmeans_assignments)
        legend = ax.legend(*scatter.legend_elements(), loc="center left", bbox_to_anchor=(1.1, 0.5), title="Class labels")
        ax.add_artist(legend)
        plt.savefig(f"results/clustering/plots_{savename}/tsne_{savename}_kmeans_k_{K}", bbox_inches='tight')
        plt.close(fig)
        plt.clf()
        fig = plt.figure(figsize=(10, 4))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(X_tsne[:,0], X_tsne[:,1], X_tsne[:,2], c=spectral_assignments)
        legend = ax.legend(*scatter.legend_elements(), loc="center left", bbox_to_anchor=(1.1, 0.5), title="Class labels")
        ax.add_artist(legend)
        plt.savefig(f"results/clustering/plots_{savename}/tsne_{savename}_spectral_k_{K}", bbox_inches='tight')
        plt.clf()
        fig = plt.figure(figsize=(10, 4))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(X_tsne[:,0], X_tsne[:,1], X_tsne[:,2], c=gmm_labels)
        legend = ax.legend(*scatter.legend_elements(), loc="center left", bbox_to_anchor=(1.1, 0.5), title="Class labels")
        ax.add_artist(legend)
        plt.savefig(f"results/clustering/plots_{savename}/tsne_{savename}_gmm_k_{K}", bbox_inches='tight')
        plt.close(fig)

    return silhouette_kmeans, ch_kmeans, silhouette_spectral, ch_spectral, silhouette_gmm, ch_gmm

def prep_timeseries(timeseries):
    masks = []
    for idx, t in enumerate(timeseries):
        nan_mask = torch.isnan(t)
        # Replace NaN values with 0 using boolean masking
        t[nan_mask] = -999
        missing_mask = t.eq(-999)
        # Replace -999 with -1
        t[missing_mask] = -999
        mask = nan_mask | missing_mask
        masks.append(mask)
        # If features are continous
        if idx in [0, 3, 5]:
            t[mask] = 0.000000001
            timeseries[idx] -= timeseries[idx].min(-1, keepdim=True)[0]
            timeseries[idx] /= torch.add(timeseries[idx].max(-1, keepdim=True)[0], 0.000000001)
            nans = torch.isnan(timeseries[idx])
            timeseries[idx][nans] = 0.5
            t[mask] -999

    return timeseries

def get_latent(model, latent_dim, hidden_layers, split_size = 1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Define the model architecture
    model = model(input_dims= [(64 // split_size, 128, 128, 3), (256 // split_size, 352)], latent_dim=latent_dim, 
                    hidden_layers = hidden_layers, dropout = 0.2).to(device)
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
        label = torch.tensor([int(l)-1 for l in label]).to(device)
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
                    if model_name != "VideoAutoencoder":
                        recon_video, kl_divergence, latent_representation, mus = model(video)
                        latent = mus
                    else: 
                        recon_video, latent_representation = model(video)
                        latent = latent_representation
                elif "Time" in model_name:
                    timeseries = [t.to(device) for t in timeseries]
                    timeseries = prep_timeseries(timeseries)
                    recon_timeseries, kl_divergence, latent_representation, mus = model(timeseries)
                    latent = mus
                else:
                    video = video.to(device)
                    timeseries = [t.to(device) for t in timeseries]
                    timeseries = prep_timeseries(timeseries)
                    recon_video, recon_timeseries, kl_divergence, latent_representation, mus = model([video, timeseries])
                    latent = mus
                
                latents.append(latent_representation.to("cpu"))
                labels.append(label.to("cpu"))

    latents = torch.cat(latents, dim = 0)
    labels = torch.cat(labels, dim = 0)
    return latents, labels, model_name

def run_clustering(model, latent_dim, hidden_layers, split_size = 1):
    print("Started Cluster task")
    latents, labels, model_name = get_latent(model, latent_dim, hidden_layers, split_size)
    kmeans_silhouette = []
    kmeans_ch = []
    spectral_silhouette = []
    spectral_ch = []
    gmm_silhouette = []
    gmm_ch = []
    print("Latent representations ready")
    for i in range(2, 15, 2):
        print("\nCluster ", i, flush = True)
        km_s, km_ch, s_s, s_ch, g_s, g_ch = run_cluster(latents, labels, K = i, norm = False, savename = model_name)
        kmeans_silhouette.append(km_s)
        kmeans_ch.append(km_ch)
        spectral_silhouette.append(s_s)
        spectral_ch.append(s_ch)
        gmm_silhouette.append(g_s)
        gmm_ch.append(g_ch)
    
    print("FOR PLOT")
    print(kmeans_silhouette)
    print(kmeans_ch)
    print(spectral_silhouette)
    print(spectral_ch)
    print(gmm_silhouette)
    print(gmm_ch)
    print(" ")
    print("Finished cluster for ", model_name)

if __name__ == "__main__":
    print("Run Cluster")
    torch.manual_seed(42)
    np.random.seed(42)
    latent_dim = 32
    video_hidden_shape = [128, 256, 512, 512]
    timeseries_hidden_dim = 1024
    timeseries_num_layers = 3
    hidden_layers = [video_hidden_shape, timeseries_hidden_dim, timeseries_num_layers]
    run_clustering(TimeBERT, latent_dim, hidden_layers, split_size = 4)
