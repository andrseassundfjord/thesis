import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from load_dataset import get_dataloaders, VideoDataset, DataFrameTimeseriesDataset
from MVAE import MVAE
from TimeseriesVAE import TimeseriesVAE
from VideoVAE import VideoVAE
from VideoAutoencoder import VideoAutoencoder
from VideoBERT_pretrained import VideoBERT_pretrained
from TimeBERT import TimeBERT
from MAE import MAE
from HMAE import HMAE
import time
import csv
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR
import torch.nn.functional as F
import random
import math
import copy
from torch.cuda.amp import GradScaler, autocast
from sklearn.cluster import KMeans
from clustering import run_clustering
from classification import train_test_classification
from risk_prediction import train_test_risk

def mape_calc(y_true, y_pred):
    if torch.max(y_true) < 249:
        y_pred = 100 * y_pred
        y_true = 100 * y_true
    abs_error = torch.abs(y_true - y_pred)
    perc_error = abs_error / torch.abs(y_true)
    perc_error[torch.isinf(perc_error)] = 0  # Handle divide by zero errors
    mean_perc_error = 100.0 * torch.mean(perc_error)
    return mean_perc_error

def generate_random(x, p=0.5):
    scaled_x = (x - 1) / 29 # 29 = 30 - 1, max number of features
    probability_of_one = p * (1 / (1 + math.exp(-10 * scaled_x)))
    choices = [0, 1]
    weights = [1 - probability_of_one, probability_of_one]
    return random.choices(choices, weights)[0]

def mask_features(tensor, batch_size, num_features):
    masked = copy.deepcopy(tensor)
    for p in [0.9, 0.7, 0.5, 0.3, 0.1]:
        if generate_random(num_features, p = p) == 1:
            batch_idx = random.randint(0, masked.size(0) - 1)
            feature_idx = random.randint(0, masked.size(2) - 1)
            masked[batch_idx, :, feature_idx] = -99
    return masked

def reg_loss(model):
    # Regularization term
    reg_loss = 0
    for param in model.parameters():
        reg_loss += torch.sum(torch.square(param))
    # Total loss
    return 0.1 * reg_loss

def kmeans_loss(z):
    z = z.detach().to("cpu").numpy()
    k =  8
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(z)
    centers = kmeans.cluster_centers_
    loss = 0
    for i in range(len(z)):
        loss += np.linalg.norm(z[i] - centers[labels[i]])
    return loss

def run(
        savename,
        load = True,
        train_ratio = 0.7,
        batch_size = 32,
        lr = 0.00001,
        num_epochs = 50,
        latent_dim = 16,
        optimizer_arg = optim.Adam,
        model_arg = MVAE,
        video_hidden_shape = [16, 32, 64, 256],
        timeseries_hidden_dim = 32,
        timeseries_num_layers = 1,
        dropout = 0.1,
        save = False,
        pretrain = False
    ):

    print(f"Started: {savename}")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush = True)

    # Set random seed
    torch.manual_seed(42)
    np.random.seed(42)

    video_train_loader, video_test_loader, timeseries_train_loader, timeseries_test_loader, label_train, label_test, risk_train, risk_test = get_dataloaders(
                                                    '/work5/share/NEDO/nedo-2019/data/processed_rosbags_topickles/fixed_pickles', 
                                                    "/work5/share/NEDO/nedo-2019/data/01_driving_data/movie", 
                                                    train_ratio = train_ratio,
                                                    batch_size = batch_size, 
                                                    save = True,
                                                    load = load
                                                )

    # Get number of modalities and input shapes
    frame_len = video_train_loader.dataset.frame_len 
    size = video_train_loader.dataset.size 

    input_dims = [(frame_len, size, size, 3), (200, 352)] # (200, 352) is not actually used, but manually set in TimeseriesVAE and MVAE
    num_features_list = [29, 6, 12, 4, 1, 9, 1, 5]

    # Initialize model
    model = model_arg(input_dims=input_dims, 
                        latent_dim=latent_dim, 
                        hidden_layers = [video_hidden_shape,
                                        timeseries_hidden_dim,
                                        timeseries_num_layers],
                        dropout = dropout,
                    ).to(device)
    
    print(f"Model: {model.__class__.__name__}")

    # Define loss function
    reconstruction_loss = nn.MSELoss(reduction='sum')

    # Define optimizer
    optimizer = optimizer_arg(model.parameters(), lr=lr)

    # LR scheduler
    scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=10) # Recude lr by factor after patience epochs
    #scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    #scheduler = StepLR(optimizer, step_size=int(num_epochs/4), gamma=0.1)
    print("Number of parameters in model: ", sum(p.numel() for p in model.parameters()))

    # Train model
    train_losses = []
    test_losses = []
    test_losses_video = []
    test_losses_time = []
    mapes_video = []
    mapes_time = []
    best_val_loss = float(math.inf)
    best_val_loss_epoch = 0
    start_time = time.time()
    print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
    print("torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(0)/1024/1024/1024))
    print("torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024))
    print("Start training")
    for epoch in range(num_epochs):
        train_loss = 0
        model.train()
        for i, (video, timeseries) in enumerate(zip(video_train_loader, timeseries_train_loader)):
            optimizer.zero_grad()
            if model.__class__.__name__ == "VideoVAE":
                # Move data to device
                video = video.to(device)
                # Forward pass
                recon_video, kl_divergence, latent, mus = model(video)
                loss = reconstruction_loss(recon_video, video)
                loss += kl_divergence
                loss += reg_loss(model)
                loss += kmeans_loss(mus)
            elif model.__class__.__name__ == "TimeseriesVAE":
                # Move data to device
                timeseries = [t.to(device) for t in timeseries]
                masks = []
                for idx, t in enumerate(timeseries):
                    nan_mask = torch.isnan(t)
                    # Replace NaN values with 0 using boolean masking
                    t[nan_mask] = 0.0
                    missing_mask = t.eq(-99)
                    # Replace -99 with -1
                    t[missing_mask] = 0.0
                    mask = nan_mask | missing_mask
                    masks.append(mask)
                    # If features are continous
                    if idx in [0, 3, 5]:
                        timeseries[idx] = F.normalize(t, p=1, dim=-1)
                timeseries_input = [mask_features(t, batch_size, num_features_list[idx]).to(device) for idx, t in enumerate(timeseries)]
                # Forward pass
                recon_timeseries, kl_divergence, latent_representation, mus = model(timeseries_input)
                loss = kl_divergence
                recon_split = []
                recon_split.extend(torch.split(recon_timeseries[0], [t.size(2) for t in timeseries[:3]], dim=-1))
                recon_split.extend(torch.split(recon_timeseries[1], [t.size(2) for t in timeseries[3:5]], dim=-1))
                recon_split.extend(torch.split(recon_timeseries[2], [t.size(2) for t in timeseries[5:]], dim=-1))
                for recon, nan_mask, t in zip(recon_split, masks, timeseries):
                    loss += reconstruction_loss(recon[~nan_mask], t[~nan_mask])
                loss += reg_loss(model)
                loss += kmeans_loss(mus)
            elif model.__class__.__name__ == "TimeseriesVAE2":
                # Move data to device
                timeseries = [t.to(device) for t in timeseries]
                masks = []
                for idx, t in enumerate(timeseries):
                    nan_mask = torch.isnan(t)
                    # Replace NaN values with 0 using boolean masking
                    t[nan_mask] = 0.0
                    missing_mask = t.eq(-99)
                    # Replace -99 with -1
                    t[missing_mask] = 0.0
                    mask = nan_mask | missing_mask
                    masks.append(mask)
                    # If features are continous
                    timeseries[idx] = F.normalize(t, p=1, dim=-1)
                timeseries_input = [mask_features(t, batch_size, num_features_list[idx]).to(device) for idx, t in enumerate(timeseries)]
                # Forward pass
                recon_timeseries, kl_divergence = model(timeseries_input)
                loss = kl_divergence
                recon_split = []
                recon_split.extend(torch.split(recon_timeseries, [t.size(2) for t in timeseries], dim=-1))
                for recon, nan_mask, t in zip(recon_split, masks, timeseries):
                    loss += reconstruction_loss(recon[~nan_mask], t[~nan_mask])
            elif model.__class__.__name__ == "VideoBERT" or model.__class__.__name__ == "VideoBERT_pretrained" or model.__class__.__name__ == "VideoAutoencoder":
                # Move data to device
                video = video.to(device)
                # Forward pass
                reconstructed, latent_representation = model(video)
                loss = reconstruction_loss(reconstructed, video)
                loss += reg_loss(model)
                loss += kmeans_loss(latent_representation)
            else:
                # Move data to device
                video = video.to(device)
                timeseries = [t.to(device) for t in timeseries]
                # Mask timeseries data
                masks = []
                for idx, t in enumerate(timeseries):
                    nan_mask = torch.isnan(t)
                    # Replace NaN values with 0 using boolean masking
                    t[nan_mask] = 0.0
                    missing_mask = t.eq(-99)
                    # Replace -99 with -1
                    t[missing_mask] = 0.0
                    mask = nan_mask | missing_mask
                    masks.append(mask)
                    # If features are continous
                    if idx in [0, 3, 5]:
                        timeseries[idx] = F.normalize(t, p=1, dim=-1)
                timeseries_input = [mask_features(t, batch_size, num_features_list[idx]).to(device) for idx, t in enumerate(timeseries)]
                # Forward pass
                recon_video, recon_timeseries, kl_divergence, latent_representation, mus = model((video, timeseries_input))
                loss = kl_divergence + reconstruction_loss(recon_video, video)
                recon_split = []
                print(recon_timeseries[0].size(), recon_timeseries[1].size(), recon_timeseries[2].size())
                recon_split.extend(torch.split(recon_timeseries[0], [t.size(2) for t in timeseries[:3]], dim=-1))
                recon_split.extend(torch.split(recon_timeseries[1], [t.size(2) for t in timeseries[3:5]], dim=-1))
                recon_split.extend(torch.split(recon_timeseries[2], [t.size(2) for t in timeseries[5:]], dim=-1))
                for recon, nan_mask, t in zip(recon_split, masks, timeseries):
                    loss += reconstruction_loss(recon[~nan_mask], t[~nan_mask])
                loss += kmeans_loss(mus)

            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            if model.__class__.__name__ == "HMAE":
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2)
        train_loss /= len(video_train_loader.dataset)
        train_losses.append(train_loss)

        # Evaluate model on test data
        test_loss = 0
        test_time_loss = 0
        test_video_loss = 0
        mape_time = 0
        mape_video = 0
        model.eval()
        with torch.no_grad():
            for video, timeseries in zip(video_test_loader, timeseries_test_loader):
                if model.__class__.__name__ == "VideoVAE":
                    # Move data to device
                    video = video.to(device)
                    # Forward pass
                    recon_video, kl_divergence, latent, mus = model(video)
                    loss = reconstruction_loss(recon_video, video)
                    test_video_loss += loss.item()
                    loss += kl_divergence
                    loss += reg_loss(model)
                    loss += kmeans_loss(mus)
                    mape_video += mape_calc(y_true=video, y_pred=recon_video)
                elif model.__class__.__name__ == "TimeseriesVAE":
                    # Move data to device
                    timeseries = [t.to(device) for t in timeseries]
                    masks = []
                    for idx, t in enumerate(timeseries):
                        nan_mask = torch.isnan(t)
                        # Replace NaN values with 0 using boolean masking
                        t[nan_mask] = 0.0
                        missing_mask = t.eq(-99)
                        t[missing_mask] = 0.0
                        mask = nan_mask | missing_mask
                        masks.append(mask)
                        # If features are continous
                        if idx in [0, 3, 5]:
                            timeseries[idx] = F.normalize(t, p=1, dim=-1)

                    timeseries_input = [mask_features(t, batch_size, num_features_list[idx]).to(device) for idx, t in enumerate(timeseries)]
                    # Forward pass
                    recon_timeseries, kl_divergence, latent_representation, mus = model(timeseries)
                    loss = 0
                    recon_split = []
                    recon_split.extend(torch.split(recon_timeseries[0], [t.size(2) for t in timeseries[:3]], dim=-1))
                    recon_split.extend(torch.split(recon_timeseries[1], [t.size(2) for t in timeseries[3:5]], dim=-1))
                    recon_split.extend(torch.split(recon_timeseries[2], [t.size(2) for t in timeseries[5:]], dim=-1))
                    for recon, nan_mask, t in zip(recon_split, masks, timeseries):
                        loss += reconstruction_loss(recon[~nan_mask], t[~nan_mask])
                        mape_time += mape_calc(y_true=t[~nan_mask], y_pred=recon[~nan_mask]) / 8
                    test_time_loss += loss.item()
                    loss += kl_divergence
                    loss += reg_loss(model)
                    loss += kmeans_loss(mus)
                elif model.__class__.__name__ == "TimeseriesVAE2":
                    # Move data to device
                    timeseries = [t.to(device) for t in timeseries]
                    masks = []
                    for idx, t in enumerate(timeseries):
                        nan_mask = torch.isnan(t)
                        # Replace NaN values with 0 using boolean masking
                        t[nan_mask] = 0.0
                        missing_mask = t.eq(-99)
                        t[missing_mask] = 0.0
                        mask = nan_mask | missing_mask
                        masks.append(mask)
                        # If features are continous
                        timeseries[idx] = F.normalize(t, p=1, dim=-1)
                    timeseries_input = [mask_features(t, batch_size, num_features_list[idx]).to(device) for idx, t in enumerate(timeseries)]
                    # Forward pass
                    recon_timeseries, kl_divergence = model(timeseries)
                    loss = 0
                    recon_split = []
                    recon_split.extend(torch.split(recon_timeseries, [t.size(2) for t in timeseries], dim=-1))
                    for recon, nan_mask, t in zip(recon_split, masks, timeseries):
                        loss += reconstruction_loss(recon[~nan_mask], t[~nan_mask])
                    test_time_loss = loss.item()
                    loss += kl_divergence
                elif model.__class__.__name__ == "VideoBERT" or model.__class__.__name__ == "VideoBERT_pretrained" or model.__class__.__name__ == "VideoAutoencoder":
                    # Move data to device
                    video = video.to(device)
                    # Forward pass
                    reconstructed, latent_representation = model(video)
                    loss = reconstruction_loss(reconstructed, video)
                    test_video_loss += loss.item()
                    loss += reg_loss(model)
                    loss += kmeans_loss(latent_representation)
                    mape_video += mape_calc(y_true=video, y_pred=reconstructed)
                else:
                    # Move data to device
                    video = video.to(device)
                    timeseries = [t.to(device) for t in timeseries]
                    # Mask timeseries data
                    masks = []
                    for idx, t in enumerate(timeseries):
                        nan_mask = torch.isnan(t)
                        # Replace NaN values with 0 using boolean masking
                        t[nan_mask] = 0.0
                        missing_mask = t.eq(-99)
                        # Replace -99 with -1
                        t[missing_mask] = 0.0
                        mask = nan_mask | missing_mask
                        masks.append(mask)
                        # If features are continous
                        if idx in [0, 3, 5]:
                            timeseries[idx] = F.normalize(t, p=1, dim=-1)
                    timeseries_input = [mask_features(t, batch_size, num_features_list[idx]).to(device) for idx, t in enumerate(timeseries)]
                    # Forward pass
                    recon_video, recon_timeseries, kl_divergence, latent_representation, mus = model((video, timeseries_input))
                    #Calculate loss for video
                    video_loss = reconstruction_loss(recon_video, video)
                    # Calculate loss for timeseries
                    recon_split = []
                    recon_split.extend(torch.split(recon_timeseries[0], [t.size(2) for t in timeseries[:3]], dim=-1))
                    recon_split.extend(torch.split(recon_timeseries[1], [t.size(2) for t in timeseries[3:5]], dim=-1))
                    recon_split.extend(torch.split(recon_timeseries[2], [t.size(2) for t in timeseries[5:]], dim=-1))
                    time_loss = 0
                    for recon, nan_mask, t in zip(recon_split, masks, timeseries):
                        time_loss += reconstruction_loss(recon[~nan_mask], t[~nan_mask])
                        mape_time += mape_calc(y_true=t[~nan_mask], y_pred=recon[~nan_mask]) / 8
                    mape_video += mape_calc(y_true=video, y_pred=recon_video)
                    loss = kl_divergence + video_loss + time_loss
                    loss += kmeans_loss(mus)
                    test_time_loss += time_loss.item()
                    test_video_loss += video_loss.item()
                test_loss += loss.item()
        # lr schedule step
        scheduler.step(test_loss) # For plateau
        #scheduler.step() # for other
        test_loss /= len(video_test_loader.dataset)
        mapes_time.append(mape_time / len(video_test_loader.dataset))
        mapes_video.append(mape_video / len(video_test_loader.dataset))
        if test_video_loss > 0:
            test_losses_video.append(test_video_loss / len(video_test_loader.dataset))
        if test_time_loss > 0:
            test_losses_time.append(test_time_loss/ len(video_test_loader.dataset))
        test_losses.append(test_loss)
        if test_loss < best_val_loss:
            best_val_loss = test_loss
            best_val_loss_epoch = epoch
            if save:
                torch.save(model.state_dict(), f'models/{type(model).__name__}_state.pth')
        # Print loss
        if ( epoch + 1 ) % 10 == 0:
            end_time = time.time()
            print(f"Time taken: {end_time - start_time:.2f} seconds")
            print('Epoch: {} \t Train Loss: {:.6f}\t Test Loss: {:.6f}'.format(epoch, train_loss, test_loss), flush = True)

    print("Finished training")
    end_time = time.time()
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    print(f"Best test loss: {best_val_loss:.6f} at epoch: {best_val_loss_epoch}")
    if len(test_losses_time) > 0:
        print(f"Timeseries loss: {test_losses_time[best_val_loss_epoch]:.6f} at epoch: {best_val_loss_epoch}")
    if len(test_losses_video) > 0:
        print(f"Video loss: {test_losses_video[best_val_loss_epoch]:.6f} at epoch: {best_val_loss_epoch}")

    print(f"Timeseries MAPE: {mapes_time[best_val_loss_epoch]:.6f}")
    print(f"Video MAPE: {mapes_video[best_val_loss_epoch]:.6f}")
    plot_loss(train_losses, test_losses, "results/loss_plots/{}".format(savename), num_epochs)
    plot_loss_individual(test_losses_time, test_losses_video, "results/loss_plots/{}_individual".format(savename), num_epochs)
    num_params = sum(p.numel() for p in model.parameters())

    hyperparameters = [savename, type(model).__name__, end_time - start_time, train_losses[-1], test_losses[-1], num_params, lr, num_epochs, batch_size, 
                        latent_dim, input_dims[0], input_dims[1], video_hidden_shape, timeseries_num_layers, timeseries_hidden_dim,
                        "LeakyReLU", dropout, "dropout, kl div", type(optimizer).__name__, type(scheduler).__name__, "kaiminghe_uniform, xavier for lstm", "None", "Yes", ""]

    write_hyperparameters_to_file(hyperparameters, "results/hyperparameters.csv")

def write_hyperparameters_to_file(hyperparameters, file_path):
    """
    Write a list of hyperparameters to a file as a new column.
    
    Parameters:
    hyperparameters (list): A list of hyperparameter values.
    file_path (str): The path to the file to write to.
    """
    hyperparameters = [format(value, '.6f') if isinstance(value, float) else str(value) for value in hyperparameters]
    
    # Write the updated data to the file
    with open(file_path, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(hyperparameters)

def plot_loss(train_losses, test_losses, savename, num_epochs):
    # Plot the training and testing losses and accuracies
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.plot(np.linspace(1, num_epochs, num_epochs), train_losses, label='Training')
    ax.plot(np.linspace(1, num_epochs, num_epochs), test_losses, label='Testing')
    ax.set_title('Loss over Epochs')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    plt.savefig(savename)

def plot_loss_individual(time_loss, video_loss, savename, num_epochs):
    # Plot the training and testing losses and accuracies
    if len(time_loss) > 0 and len(video_loss) > 0:
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.plot(np.linspace(1, num_epochs, num_epochs), time_loss, label='Timeseries')
        ax.plot(np.linspace(1, num_epochs, num_epochs), video_loss, label='Video')
        ax.set_title('Loss over Epochs')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        plt.savefig(savename)

def save_num_frames(video_train_loader, video_test_loader):
    # Save and plot num_frames
    num_frames = video_train_loader.dataset.num_frames
    for key, value in video_test_loader.dataset.num_frames.items():
        if key in num_frames:
            num_frames[key] += value
        else:
            num_frames[key] = value
    # Save the result as a CSV file
    with open('results/frames_stats.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        for key, value in num_frames.items():
            writer.writerow([key, value])

    plot_num_frames(num_frames)

def plot_num_frames(num_frames):
    # Plot the histogram
    plt.clf()
    # Get the keys and values as lists
    x_values = list(num_frames.keys())
    y_values = list(num_frames.values())

    # Set the y-axis to a logarithmic scale
    plt.yscale('log')
    # Create the bar plot
    plt.bar(x_values, y_values)
    plt.xlabel('Frames')
    plt.ylabel('Log Number of Samples')
    plt.title('Histogram of Frames')
    plt.savefig("results/frame_stats_log")

if __name__ == "__main__":
    model_arg = MVAE
    latent_dim = 512
    video_hidden_shape = [128, 256, 512, 512]
    timeseries_hidden_dim = 1024
    timeseries_num_layers = 3
    hidden_layers = [video_hidden_shape, timeseries_hidden_dim, timeseries_num_layers]

    """
    CHANGE LOSS
    CHANGE MASKING
    ADD MASKED_MAPE
    """

    run(
        "MVAE_full",
        load = True,
        train_ratio = 0.7,
        batch_size = 32,
        lr = 0.00001,
        num_epochs = 150,
        latent_dim = latent_dim,
        optimizer_arg = optim.Adam,
        model_arg = model_arg,
        video_hidden_shape = video_hidden_shape,
        timeseries_hidden_dim = timeseries_hidden_dim,
        timeseries_num_layers = timeseries_num_layers,
        dropout = 0.2,
        save = True,
        pretrain = True
    )

    run_clustering(model_arg, latent_dim, hidden_layers)
    train_test_classification(model_arg, epochs=40, lr=0.1, latent_dim=latent_dim, hidden_dim=512, hidden_layers=hidden_layers)
    train_test_risk(model_arg, epochs=40, lr=0.1, latent_dim=latent_dim, hidden_dim=512, hidden_layers=hidden_layers)
