import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from load_dataset import get_dataloaders, VideoDataset, DataFrameTimeseriesDataset
from MVAE import MVAE
from TimeseriesVAE import TimeseriesVAE
from TimeseriesVAE2 import TimeseriesVAE2
from VideoVAE import VideoVAE
import time
import csv
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR
import torch.nn.functional as F
import random
import math
import copy

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
            masked[batch_idx, :, feature_idx] = 0.0
    return masked

def run(
        savename,
        load = True,
        train_ratio = 0.7,
        batch_size = 32,
        lr = 0.001,
        num_epochs = 50,
        latent_dim = 16,
        optimizer_arg = optim.Adam,
        model_arg = MVAE,
        video_hidden_shape = [16, 32, 64, 256],
        timeseries_hidden_dim = 32,
        timeseries_num_layers = 1,
        dropout = 0.1,
        save = False
    ):

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush = True)

    #torch.cuda.empty_cache()

    # Set random seed
    torch.manual_seed(42)
    np.random.seed(42)

    video_train_loader, video_test_loader, timeseries_train_loader, timeseries_test_loader = get_dataloaders(
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

    # Initialize MVAE model
    model = model_arg(input_dims=input_dims, 
                        latent_dim=latent_dim, 
                        hidden_layers = [video_hidden_shape,
                                        timeseries_hidden_dim,
                                        timeseries_num_layers],
                        dropout = dropout
                    ).to(device)

    params = list(model.parameters())
    # Define loss function
    reconstruction_loss = nn.MSELoss(reduction='sum')
    bert_loss = nn.CrossEntropyLoss()

    # Define optimizer
    optimizer = optimizer_arg(model.parameters(), lr=lr)

    # LR scheduler
    scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=5) # Recude lr by factor after patience epochs
    #scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    #scheduler = StepLR(optimizer, step_size=int(num_epochs/4), gamma=0.1)

    # Train model
    train_losses = []
    test_losses = []

    start_time = time.time()
    print("Start training")
    for epoch in range(num_epochs):
        train_loss = 0
        model.train()
        for i, (video, timeseries) in enumerate(zip(video_train_loader, timeseries_train_loader)):
            if model.__class__.__name__ == "VideoVAE":
                # Move data to device
                video = video.to(device)
                # Forward pass
                recon_video, kl_divergence, _ = model(video)
                loss = reconstruction_loss(recon_video, video)
                loss += kl_divergence
            elif model.__class__.__name__ == "TimeseriesVAE":
                # Move data to device
                timeseries = [t.to(device) for t in timeseries]
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
                timeseries_input = [mask_features(t, batch_size, num_features_list[idx]).to(device) for idx, t in enumerate(timeseries)]
                # Forward pass
                recon_timeseries, kl_divergence = model(timeseries_input)
                loss = kl_divergence
                recon_split = []
                recon_split.extend(torch.split(recon_timeseries[0], [t.size(2) for t in timeseries[:3]], dim=-1))
                recon_split.extend(torch.split(recon_timeseries[1], [t.size(2) for t in timeseries[3:5]], dim=-1))
                recon_split.extend(torch.split(recon_timeseries[2], [t.size(2) for t in timeseries[5:]], dim=-1))
                for recon, nan_mask, t in zip(recon_split, masks, timeseries):
                    loss += reconstruction_loss(recon[~nan_mask], t[~nan_mask])
            elif model.__class__.__name__ == "TimeseriesVAE2":
                # Move data to device
                timeseries = [t.to(device) for t in timeseries]
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
                timeseries_input = [mask_features(t, batch_size, num_features_list[idx]).to(device) for idx, t in enumerate(timeseries)]
                # Forward pass
                recon_timeseries, kl_divergence = model(timeseries_input)
                loss = kl_divergence
                recon_split = []
                recon_split.extend(torch.split(recon_timeseries, [t.size(2) for t in timeseries], dim=-1))
                for recon, nan_mask, t in zip(recon_split, masks, timeseries):
                    nan_mask = torch.isnan(recon)
                    if torch.any(nan_mask):
                        print("nan in recon")
                    loss += reconstruction_loss(recon[~nan_mask], t[~nan_mask])
            elif model.__class__.__name__ == "VideoBert":
                # Move data to device
                video = video.to(device)
                # Forward pass
                output = model(video)
                loss = bert_loss(output) # add labels
            else:
                # Move data to device
                video = video.to(device)
                timeseries = timeseries.to(device)
                nan_mask = torch.isnan(timeseries)
                # Replace NaN values with 0 using boolean masking
                timeseries[nan_mask] = 0.0
                # Forward pass
                recon_video, recon_timeseries, kl_divergence = model([video, timeseries])
                loss = reconstruction_loss(recon_timeseries[~nan_mask], timeseries[~nan_mask]) + reconstruction_loss(recon_video, video)
                loss += kl_divergence

            train_loss += loss.item()
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_loss /= len(video_train_loader.dataset)
        train_losses.append(train_loss)

        # Evaluate model on test data
        test_loss = 0
        model.eval()
        with torch.no_grad():
            for video, timeseries in zip(video_test_loader, timeseries_test_loader):
                if model.__class__.__name__ == "VideoVAE":
                    # Move data to device
                    video = video.to(device)
                    # Forward pass
                    recon_video, kl_divergence, _ = model(video)
                    loss = reconstruction_loss(recon_video, video)
                    loss += kl_divergence
                elif model.__class__.__name__ == "TimeseriesVAE":
                    # Move data to device
                    timeseries = [t.to(device) for t in timeseries]
                    masks = []
                    for idx, t in enumerate(timeseries):
                        nan_mask = torch.isnan(t)
                        # Replace NaN values with 0 using boolean masking
                        t[nan_mask] = 0.0
                        missing_mask = t.eq(-999)
                        t[missing_mask] = 0.0
                        mask = nan_mask | missing_mask
                        masks.append(mask)
                        # If features are continous
                        if idx in [0, 3, 5]:
                            timeseries[idx] = F.normalize(t, p=1, dim=1)

                    timeseries_input = [mask_features(t, batch_size, num_features_list[idx]).to(device) for idx, t in enumerate(timeseries)]
                    # Forward pass
                    recon_timeseries, kl_divergence = model(timeseries)
                    loss = kl_divergence
                    recon_split = []
                    recon_split.extend(torch.split(recon_timeseries[0], [t.size(2) for t in timeseries[:3]], dim=-1))
                    recon_split.extend(torch.split(recon_timeseries[1], [t.size(2) for t in timeseries[3:5]], dim=-1))
                    recon_split.extend(torch.split(recon_timeseries[2], [t.size(2) for t in timeseries[5:]], dim=-1))
                    for recon, nan_mask, t in zip(recon_split, masks, timeseries):
                        loss += reconstruction_loss(recon[~nan_mask], t[~nan_mask])
                elif model.__class__.__name__ == "TimeseriesVAE2":
                    # Move data to device
                    timeseries = [t.to(device) for t in timeseries]
                    masks = []
                    for idx, t in enumerate(timeseries):
                        nan_mask = torch.isnan(t)
                        # Replace NaN values with 0 using boolean masking
                        t[nan_mask] = 0.0
                        missing_mask = t.eq(-999)
                        t[missing_mask] = 0.0
                        mask = nan_mask | missing_mask
                        masks.append(mask)
                        # If features are continous
                        if idx in [0, 3, 5]:
                            timeseries[idx] = F.normalize(t, p=1, dim=1)
                    timeseries_input = [mask_features(t, batch_size, num_features_list[idx]).to(device) for idx, t in enumerate(timeseries)]
                    # Forward pass
                    recon_timeseries, kl_divergence = model(timeseries)
                    loss = kl_divergence
                    recon_split = []
                    recon_split.extend(torch.split(recon_timeseries, [t.size(2) for t in timeseries], dim=-1))
                    for recon, nan_mask, t in zip(recon_split, masks, timeseries):
                        loss += reconstruction_loss(recon[~nan_mask], t[~nan_mask])
                elif model.__class__.__name__ == "VideoBert":
                    # Move data to device
                    video = video.to(device)
                    # Forward pass
                    output = model(video)
                    loss = bert_loss(output) # add labels
                else:
                    # Move data to device
                    video = video.to(device)
                    timeseries = timeseries.to(device)
                    nan_mask = torch.isnan(timeseries)
                    # Replace NaN values with 0 using boolean masking
                    timeseries[nan_mask] = 0.0
                    # Forward pass
                    recon_video, recon_timeseries, kl_divergence = model([video, timeseries])
                    loss = reconstruction_loss(recon_timeseries[~nan_mask], timeseries[~nan_mask]) + reconstruction_loss(recon_video, video)
                    loss += kl_divergence
                test_loss += loss.item()
        # lr schedule step
        scheduler.step(test_loss) # For plateau
        #scheduler.step() # for other
        test_loss /= len(video_test_loader.dataset)
        test_losses.append(test_loss)
        # Print loss
        if ( epoch + 1 ) % 10 == 0:
            end_time = time.time()
            print(f"Time taken: {end_time - start_time:.2f} seconds")
            print('Epoch: {} \t Train Loss: {:.6f}\t Test Loss: {:.6f}'.format(epoch, train_loss, test_loss), flush = True)

    print("Finished training")
    end_time = time.time()
    print(f"Time taken: {end_time - start_time:.2f} seconds")

    plot_loss(train_losses, test_losses, "results/loss_plots/{}".format(savename), num_epochs)
    num_params = sum(p.numel() for p in model.parameters())

    hyperparameters = [savename, type(model).__name__, end_time - start_time, train_losses[-1], test_losses[-1], num_params, lr, num_epochs, batch_size, 
                        latent_dim, input_dims[0], input_dims[1], video_hidden_shape, timeseries_num_layers, timeseries_hidden_dim,
                        "LeakyReLU", dropout, "dropout, kl div", type(optimizer).__name__, type(scheduler).__name__, "kaiminghe_uniform, xavier for lstm", "None", "Yes", ""]

    write_hyperparameters_to_file(hyperparameters, "results/hyperparameters.csv")
    if save:
        torch.save(model.state_dict(), f'models/{type(model).__name__}_state.pth')

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
    run(
        "test",
        load = True,
        train_ratio = 0.7,
        batch_size = 32,
        lr = 0.0001,
        num_epochs = 10,
        latent_dim = 256,
        optimizer_arg = optim.Adam,
        model_arg = TimeseriesVAE2,
        video_hidden_shape = [32, 64, 128, 256],
        timeseries_hidden_dim = 64,
        timeseries_num_layers = 3,
        dropout = 0.1,
        save = False
    )