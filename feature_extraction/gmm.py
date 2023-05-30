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
from TimeBERT import TimeBERT
from MidMVAE import MidMVAE
from TimeseriesVAE import TimeseriesVAE
from VideoVAE import VideoVAE
from VideoAutoencoder import VideoAutoencoder
from VideoBert import VideoBERT
from VideoBERT_pretrained import VideoBERT_pretrained
import warnings
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
import seaborn as sns

def run_gmm(X_train, X_test, y_train, y_test, num_classes = 14, norm = False, savename = "model", classes_list = None):
    masks = []
    for val in classes_list:
        masks.append(y_train.eq(val))
    mask = masks[0]
    for m in masks[1:]:
        mask = mask | m
    X_train = X_train[mask]
    y_train = y_train[mask]
    masks = []
    for val in classes_list:
        masks.append(y_test.eq(val))
    mask = masks[0]
    for m in masks[1:]:
        mask = mask | m
    X_test = X_test[mask]
    y_test = y_test[mask]
    # Initialize and fit the GMM
    gmm = GaussianMixture(n_components=num_classes)  # num_classes is the number of classes in your classification task
    gmm.fit(X_train)
    train_features = gmm.predict_proba(X_train)

    # Train a logistic regression classifier on the GMM features
    classifier = LogisticRegression(solver="newton-cg", multi_class="multinomial", max_iter=250, penalty="l2")
    classifier.fit(train_features, y_train)

    # Extract GMM features for test data
    test_features = gmm.predict_proba(X_test)

    # Predict class labels using the trained classifier
    y_pred = classifier.predict(test_features)

    cm = confusion_matrix(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    label_ticks = [str(i) for i in classes_list]

    print(f"\nEvaluation of GMM classification\n")

    print("Confusion matrix")
    print(label_ticks)
    for idx, line in enumerate(cm):
        print(f"{idx+1} {line}")

    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.clf()
    sns.set(font_scale=1.2) # adjust the font size
    sns.heatmap(cm_norm, annot=False, fmt='.2f', xticklabels= label_ticks, yticklabels=label_ticks, cmap='Reds')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    #plt.title('Confusion Matrix')
    plt.savefig(f"results/gmm/{savename}_confusion_matrix_gmm")
    
    print(f"F1 Score: {f1}")

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
    X_train = []
    y_train = []
    X_test = []
    y_test = []

    model.eval()
    for video, timeseries, label in zip(video_train_loader, timeseries_train_loader, label_train):
        video_slices = torch.split(video, video.size(2) // split_size, dim=2)
        timeseries_slices = [[] for _ in range(split_size)]
        label = torch.tensor([int(l)-1 for l in label])
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
                        latent = mus.to("cpu")
                    else: 
                        recon_video, latent_representation = model(video)
                        latent = latent_representation.to("cpu")
                elif "Time" in model_name:
                    timeseries = [t.to(device) for t in timeseries]
                    timeseries = prep_timeseries(timeseries)
                    recon_timeseries, kl_divergence, latent_representation, mus = model(timeseries)
                    latent = mus.to("cpu")
                else:
                    video = video.to(device)
                    timeseries = [t.to(device) for t in timeseries]
                    timeseries = prep_timeseries(timeseries)
                    recon_video, recon_timeseries, kl_divergence, latent_representation, mus = model([video, timeseries])
                    latent = mus.to("cpu")
                
                X_train.append(latent)
                y_train.append(label)

    X_train = torch.cat(X_train, dim = 0)
    y_train = torch.cat(y_train, dim = 0)
    
    model.eval()
    for video, timeseries, label in zip(video_test_loader, timeseries_test_loader, label_test):
        video_slices = torch.split(video, video.size(2) // split_size, dim=2)
        timeseries_slices = [[] for _ in range(split_size)]
        label = torch.tensor([int(l)-1 for l in label])
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
                        latent = mus.to("cpu")
                    else: 
                        recon_video, latent_representation = model(video)
                        latent = latent_representation.to("cpu")
                elif "Time" in model_name:
                    timeseries = [t.to(device) for t in timeseries]
                    timeseries = prep_timeseries(timeseries)
                    recon_timeseries, kl_divergence, latent_representation, mus = model(timeseries)
                    latent = mus.to("cpu")
                else:
                    video = video.to(device)
                    timeseries = [t.to(device) for t in timeseries]
                    timeseries = prep_timeseries(timeseries)
                    recon_video, recon_timeseries, kl_divergence, latent_representation, mus = model([video, timeseries])
                    latent = mus.to("cpu")
                
                X_test.append(latent)
                y_test.append(label)

    X_test = torch.cat(X_test, dim = 0)
    y_test = torch.cat(y_test, dim = 0)


    return X_train, y_train, X_test, y_test, model_name

def run_gmm_classification(model, latent_dim, hidden_layers, split_size = 1, classes_list = None):
    if classes_list == None:
        classes_list = range(14)
    print("Started GMM classification")
    X_train, y_train, X_test, y_test, model_name = get_latent(model, latent_dim, hidden_layers, split_size)
    print("Latent representations ready")
    run_gmm(X_train = X_train, X_test = X_test, y_train = y_train, y_test = y_test, num_classes=len(classes_list), norm = False, savename = model_name, classes_list = classes_list)
    print("Finished GMM Classification for ", model_name)

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    latent_dim = 2048
    video_hidden_shape = [128, 256, 512, 512]
    timeseries_hidden_dim = 1051224
    timeseries_num_layers = 3
    hidden_layers = [video_hidden_shape, timeseries_hidden_dim, timeseries_num_layers]
    run_gmm_classification(MVAE, latent_dim, hidden_layers, split_size = 4)
