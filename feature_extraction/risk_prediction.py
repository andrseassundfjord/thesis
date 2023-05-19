import numpy as np
# Labeled cluster metrics
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from load_dataset import get_dataloaders, VideoDataset, DataFrameTimeseriesDataset, LabelDataset
import torch.nn.functional as F
# Import models
from MVAE import MVAE
from TimeBERT import TimeBERT
from MidMVAE import MidMVAE
from TimeseriesVAE import TimeseriesVAE
from VideoVAE import VideoVAE
from VideoAutoencoder import VideoAutoencoder
from VideoBert import VideoBERT
from MAE import MAE
from VideoBERT_pretrained import VideoBERT_pretrained
import math
from sklearn.metrics import mean_absolute_percentage_error
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR

def reg_loss(model):
    # Regularization term
    reg_loss = 0
    for param in model.parameters():
        reg_loss += torch.sum(torch.square(param))
    # Total loss
    return 0.1 * reg_loss 

class SimpleModel(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)  # num_classes is the number of classes in your classification task
        
    def forward(self, x):
        x = self.fc1(x)
        x = nn.functional.leaky_relu(x)
        x = self.fc2(x)
        x = 10 * torch.sigmoid(x) - 5
        x = x.view(-1)
        return x

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
            timeseries[idx] = F.normalize(t, p=1, dim=-1)
    return timeseries

def plot_difference(y_pred, y_true, model_name):
    plt.clf()
    diff = torch.sub(y_true, y_pred).detach().to("cpu").numpy()
    plt.hist(diff, bins=20)
    plt.xlabel('Difference')
    plt.ylabel('Frequency')
    plt.savefig(f"results/riskScore_results/{model_name}") 

def train_test_risk(model, epochs = 100, lr = 0.01, latent_dim = 32, hidden_dim = 512, hidden_layers = [[128, 256, 512, 512], 256, 3], split_size = 1):
    print("\nStart risk score fine-tuning")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Define the model architecture
    pretrained_model = model(input_dims= [(64 // split_size, 128, 128, 3), (200 // split_size, 352)], latent_dim=latent_dim, 
                    hidden_layers = hidden_layers, dropout= 0.2).to(device)

    model_name = pretrained_model.__class__.__name__

    # Load the model state
    if split_size > 1:
        pretrained_model.load_state_dict(torch.load(f'augmented_models/{model_name}_state.pth'))
    else:
        pretrained_model.load_state_dict(torch.load(f'models/{model_name}_state.pth'))
    
    for param in pretrained_model.parameters():
        param.requires_grad = False

    if pretrained_model.__class__.__name__ == "MAE" or pretrained_model.__class__.__name__ == "HMAE":
        simple_model = SimpleModel(latent_dim * 2, hidden_dim).to(device)
    else:
        simple_model = SimpleModel(latent_dim, hidden_dim).to(device)

    # Define loss function
    criterion = nn.MSELoss()

    # Define optimizer
    optimizer = optim.Adam(simple_model.parameters(), lr=lr)

    # LR scheduler
    scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=5) # Recude lr by factor after patience epochs

    video_train_loader, video_test_loader, timeseries_train_loader, timeseries_test_loader, label_train, label_test, risk_train, risk_test = get_dataloaders(
                                                '/work5/share/NEDO/nedo-2019/data/processed_rosbags_topickles/fixed_pickles', 
                                                "/work5/share/NEDO/nedo-2019/data/01_driving_data/movie", 
                                                train_ratio = 0.7,
                                                batch_size = 32, 
                                                save = False,
                                                load = True
                                            )
    train_losses = []
    test_losses = []
    best_val_loss = float(math.inf)
    best_val_loss_epoch = 0
    for epoch in range(epochs):
        # Train
        # Set the pretrained model to evaluation mode
        pretrained_model.eval()
        simple_model.train()
        train_loss = 0
        for video, timeseries, riskScore in zip(video_train_loader, timeseries_train_loader, risk_train):
            loss = 0
            optimizer.zero_grad()
            riskScore = torch.tensor([float(r) for r in riskScore]).to(device)
            video_slices = torch.split(video, video.size(2) // split_size, dim=2)
            timeseries_slices = [[] for _ in range(split_size)]
            for t in timeseries:
                split_t = torch.split(t, t.size(1) // split_size, dim = 1)
                for idx, split in enumerate(split_t):
                    timeseries_slices[idx].append(split)
            for i in range(split_size):
                video = video_slices[i]
                timeseries = timeseries_slices[i]
                if "Video" in model_name:
                    video = video.to(device)
                    if model_name != "VideoAutoencoder":
                        recon_video, kl_divergence, latent_representation, mus = pretrained_model(video)
                        latent = mus
                    else: 
                        recon_video, latent_representation = pretrained_model(video)
                        latent = latent_representation
                elif "Time" in model_name:
                    timeseries = [t.to(device) for t in timeseries]
                    timeseries = prep_timeseries(timeseries)
                    recon_timeseries, kl_divergence, latent_representation, mus = pretrained_model(timeseries)
                    latent = mus
                else:
                    video = video.to(device)
                    timeseries = [t.to(device) for t in timeseries]
                    timeseries = prep_timeseries(timeseries)
                    recon_video, recon_timeseries, kl_divergence, latent_representation, mus = pretrained_model([video, timeseries])
                    latent = mus
            
                output = simple_model(latent)
                loss += criterion(output, riskScore)
            loss += reg_loss(simple_model)
            
            train_loss += loss.item()
            loss.backward()
            optimizer.step()

        train_loss /= (len(video_train_loader.dataset) * split_size)
        train_losses.append(train_loss)
        # Test
        pretrained_model.eval()
        simple_model.eval()
        test_loss = 0
        with torch.no_grad():
            for video, timeseries, riskScore in zip(video_test_loader, timeseries_test_loader, risk_test):
                riskScore = torch.tensor([float(r) for r in riskScore]).to(device)
                loss = 0
                video_slices = torch.split(video, video.size(2) // split_size, dim=2)
                timeseries_slices = [[] for _ in range(split_size)]
                for t in timeseries:
                    split_t = torch.split(t, t.size(1) // split_size, dim = 1)
                    for idx, split in enumerate(split_t):
                        timeseries_slices[idx].append(split)
                for i in range(split_size):
                    video = video_slices[i]
                    timeseries = timeseries_slices[i]
                    if "Video" in model_name:
                        video = video.to(device)
                        if model_name != "VideoAutoencoder":
                            recon_video, kl_divergence, latent_representation, mus = pretrained_model(video)
                            latent = mus
                        else: 
                            recon_video, latent_representation = pretrained_model(video)
                            latent = latent_representation
                    elif "Time" in model_name:
                        timeseries = [t.to(device) for t in timeseries]
                        timeseries = prep_timeseries(timeseries)
                        recon_timeseries, kl_divergence, latent_representation, mus = pretrained_model(timeseries)
                        latent = mus
                    else:
                        video = video.to(device)
                        timeseries = [t.to(device) for t in timeseries]
                        timeseries = prep_timeseries(timeseries)
                        recon_video, recon_timeseries, kl_divergence, latent_representation, mus = pretrained_model([video, timeseries])
                        latent = mus
                    
                    output = simple_model(latent)
                    loss += criterion(output, riskScore)
                loss += reg_loss(simple_model)
                test_loss += loss.item()
        # lr schedule step
        scheduler.step(test_loss) # For plateau

        test_loss /= (len(video_test_loader.dataset) * split_size)
        test_losses.append(test_loss)
        if test_loss < best_val_loss:
            best_val_loss = test_loss
            best_val_loss_epoch = epoch
            if split_size > 1:
                torch.save(simple_model.state_dict(), f'augmented_models/{model_name}_risk_state.pth')
            else:
                torch.save(simple_model.state_dict(), f'models/{model_name}_risk_state.pth')

        # Print loss
        if ( epoch + 1 ) % 5 == 0:
            print('Epoch: {} \t Train Loss: {:.6f}\t Test Loss: {:.6f}'.format(epoch, train_loss, test_loss), flush = True)

    print(f"Finished training {model_name} for risk score prediction")
    print(f"Best test loss: {best_val_loss:.6f} at epoch: {best_val_loss_epoch}")

    evaluate(pretrained_model, model_name, latent_dim, hidden_dim, split_size)

def evaluate(pretrained_model, model_name, latent_dim, hidden_dim, split_size = 1):
    print("Start evaluation", flush = True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # load simple model
    if model_name == "MAE" or model_name == "HMAE":
        simple_model = SimpleModel(latent_dim * 2, hidden_dim).to(device)
    else:
        simple_model = SimpleModel(latent_dim, hidden_dim).to(device)
    if split_size > 1:
        simple_model.load_state_dict(torch.load(f'augmented_models/{model_name}_risk_state.pth'))
    else:
        simple_model.load_state_dict(torch.load(f'models/{model_name}_risk_state.pth'))
    # Load dataloaders
    video_train_loader, video_test_loader, timeseries_train_loader, timeseries_test_loader, label_train, label_test, risk_train, risk_test = get_dataloaders(
                                                '/work5/share/NEDO/nedo-2019/data/processed_rosbags_topickles/fixed_pickles', 
                                                "/work5/share/NEDO/nedo-2019/data/01_driving_data/movie", 
                                                train_ratio = 0.7,
                                                batch_size = 32, 
                                                save = False,
                                                load = True
                                            )

    y_preds = []
    riskScores = []

    pretrained_model.eval()
    simple_model.eval()
    with torch.no_grad():
        for video, timeseries, riskScore in zip(video_test_loader, timeseries_test_loader, risk_test):
            video_slices = torch.split(video, video.size(2) // split_size, dim=2)
            timeseries_slices = [[] for _ in range(split_size)]
            for t in timeseries:
                split_t = torch.split(t, t.size(1) // split_size, dim = 1)
                for idx, split in enumerate(split_t):
                    timeseries_slices[idx].append(split)
            for i in range(split_size):
                video = video_slices[i]
                timeseries = timeseries_slices[i]
                if "Video" in model_name:
                    video = video.to(device)
                    if model_name != "VideoAutoencoder":
                        recon_video, kl_divergence, latent_representation, mus = pretrained_model(video)
                        latent = mus
                    else: 
                        recon_video, latent_representation = pretrained_model(video)
                        latent = latent_representation
                elif "Time" in model_name:
                    timeseries = [t.to(device) for t in timeseries]
                    timeseries = prep_timeseries(timeseries)
                    recon_timeseries, kl_divergence, latent_representation, mus = pretrained_model(timeseries)
                    latent = mus
                else:
                    video = video.to(device)
                    timeseries = [t.to(device) for t in timeseries]
                    timeseries = prep_timeseries(timeseries)
                    recon_video, recon_timeseries, kl_divergence, latent_representation, mus = pretrained_model([video, timeseries])
                    latent = mus
                
                output = simple_model(latent)

                y_preds.append(output.to("cpu"))
                riskScore = torch.tensor([float(r) for r in riskScore])
                riskScores.append(riskScore)

    y_preds = torch.cat(y_preds, dim = 0)
    riskScores = torch.cat(riskScores, dim = 0)
    
    mape = mean_absolute_percentage_error(y_true=riskScores, y_pred=y_preds)

    print(f"\nEvaluation of fine-tuned {model_name} model\n")
    
    print(f"MAPE Score: {mape}")

    plot_difference(y_pred=y_preds, y_true=riskScores, model_name=model_name)

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    train_test_risk(MVAE, epochs=80, lr=0.1, latent_dim=64, hidden_dim = 1024, hidden_layers=[[128, 256, 512, 512], 512, 3], split_size=4)