import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np
from load_dataset import get_dataloaders
from MVAE import MVAE

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set random seed
torch.manual_seed(42)
np.random.seed(42)

# Set train test split ratio
train_ratio = 0.7

# Set batch size
batch_size = 32

# Set learning rate
lr = 0.001

# Set number of epochs
num_epochs = 50

# Set latent space dimensions
latent_dim = 32

train_loader, test_loadet = get_dataloaders('../experiment1/resampled_pickles', 
                                                "/work5/share/NEDO/nedo-2019/data/01_driving_data/movie", 
                                                train_ratio = train_ratio,
                                                batch_size = batch_size, 
                                                save = True,
                                                load = True
                                            )

# Get number of modalities and input shapes
input_dims = [(182, 128, 128, 3), (200, 352)]

# Initialize MVAE model
model = MVAE(input_dims=input_dims, latent_dim=latent_dim).to(device)

# Define loss function
reconstruction_loss = nn.MSELoss(reduction='sum')

# Define optimizer
optimizer = optim.Adam(model.parameters(), lr=lr)

# Train model
train_losses = []
test_losses = []

print("Start training")
for epoch in range(num_epochs):
    train_loss = 0
    model.train()
    for i, (video, timeseries) in enumerate(train_loader):
        # Move data to device
        video = video.to(device)
        timeseries = timeseries.to(device)

        # Forward pass
        recon_video, recon_timeseries, mu, logvar = model(video, timeseries)
        loss = reconstruction_loss(recon_video, video) + reconstruction_loss(recon_timeseries, timeseries)
        kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss += kl_divergence
        train_loss += loss.item()

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss /= len(train_loader.dataset)
    train_losses.append(train_loss)

    # Evaluate model on test data
    test_loss = 0
    model.eval()
    with torch.no_grad():
        for video, timeseries in test_loader:
            # Move data to device
            video = video.to(device)
            timeseries = timeseries.to(device)

            # Forward pass
            recon_video, recon_timeseries, mu, logvar = model(video, timeseries)
            loss = reconstruction_loss(recon_video, video) + reconstruction_loss(recon_timeseries, timeseries)
            kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss += kl_divergence
            test_loss += loss.item()

    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    # Print loss
    if epoch % 10 == 0:
        print('Epoch: {} \t Train Loss: {:.6f}\t Test Loss: {:.6f}'.format(epoch, train_loss, test_loss))

print("Finished training")
# Plot the training and testing losses and accuracies
fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(train_losses, label='Training')
ax.plot(test_losses, label='Testing')
ax.set_title('Loss over Epochs')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.legend()
plt.savefig("mvae_loss")