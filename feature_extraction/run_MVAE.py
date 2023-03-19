import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from load_dataset import get_dataloaders
from MVAE import MVAE
import time
import sys
import csv
import pandas as pd

# Set experiment specifics
savename = "results/video_VAE"
load = True

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

#torch.cuda.empty_cache()

# Set random seed
torch.manual_seed(42)
np.random.seed(42)

# Set train test split ratio
train_ratio = 0.7

# Set batch size
batch_size = 2

# Set learning rate
lr = 0.001

# Set number of epochs
num_epochs = 10

# Set latent space dimensions
latent_dim = 4

video_train_loader, video_test_loader, timeseries_train_loader, timeseries_test_loader = get_dataloaders(
                                                '../experiment1/resampled_pickles', 
                                                "/work5/share/NEDO/nedo-2019/data/01_driving_data/movie", 
                                                train_ratio = train_ratio,
                                                batch_size = batch_size, 
                                                save = True,
                                                load = load
                                            )

# Get number of modalities and input shapes
frame_len = video_train_loader.dataset.frame_len 
size = video_train_loader.dataset.size 

input_dims = [(frame_len, size, size, 3), (200, 352)]

# Initialize MVAE model
model = MVAE(input_dims=input_dims, latent_dim=latent_dim).to(device)

# Define loss function
reconstruction_loss = nn.MSELoss(reduction='sum')

# Define optimizer
optimizer = optim.Adam(model.parameters(), lr=lr)

# Train model
train_losses = []
test_losses = []

start_time = time.time()
print("Start training")
for epoch in range(num_epochs):
    train_loss = 0
    model.train()
    for i, (video, timeseries) in enumerate(zip(video_train_loader, timeseries_train_loader)):
    #for i, timeseries in enumerate(timeseries_train_loader):
        # Move data to device
        video = video.to(device)
        timeseries = timeseries.to(device)
        nan_mask = torch.isnan(timeseries)
        # Replace NaN values with 0 using boolean masking
        timeseries[nan_mask] = 0.0
        # Forward pass
        #recon_timeseries, kl_divergence = model([video, timeseries])
        recon_video, kl_divergence = model([video, timeseries])
        #loss = reconstruction_loss(recon_timeseries[~nan_mask], timeseries[~nan_mask])
        loss = reconstruction_loss(recon_video, video)
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
        #for i, timeseries in enumerate(timeseries_test_loader):
            # Move data to device
            video = video.to(device)
            timeseries = timeseries.to(device)        
            nan_mask = torch.isnan(timeseries)
            # Replace NaN values with -1 using boolean masking
            timeseries[nan_mask] = 0.0
            # Forward pass
            #recon_timeseries, kl_divergence = model([video, timeseries])
            recon_video, kl_divergence = model([video, timeseries])
            #recon_video, recon_timeseries, kl_divergence = model([video, timeseries])
            #loss = reconstruction_loss(recon_timeseries, timeseries) + reconstruction_loss(recon_timeseries, timeseries)
            #loss = reconstruction_loss(recon_timeseries[~nan_mask], timeseries[~nan_mask])
            loss = reconstruction_loss(recon_video, video)
            loss += kl_divergence
            test_loss += loss.item()

    test_loss /= len(video_test_loader.dataset)
    test_losses.append(test_loss)
    # Print loss
    #if ( epoch + 1 ) % 10 == 0:
    end_time = time.time()
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    print('Epoch: {} \t Train Loss: {:.6f}\t Test Loss: {:.6f}'.format(epoch, train_loss, test_loss))

print("Finished training")
end_time = time.time()
print(f"Time taken: {end_time - start_time:.2f} seconds")

# Plot the training and testing losses and accuracies
fig, ax = plt.subplots(figsize=(12, 12))
ax.plot(np.linspace(1, num_epochs, num_epochs), train_losses, label='Training')
ax.plot(np.linspace(1, num_epochs, num_epochs), test_losses, label='Testing')
ax.set_title('Loss over Epochs')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.legend()
plt.savefig(savename)

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

# Plot the histogram
plt.hist(num_frames.keys(), bins=len(num_frames))
plt.xlabel('Frames')
plt.ylabel('Number of Samples')
plt.title('Histogram of Frames')
plt.savefig("results/frame_stats")

