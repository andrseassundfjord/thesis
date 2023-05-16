import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
from load_dataset import get_dataloaders, VideoDataset, DataFrameTimeseriesDataset, LabelDataset
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR
# Import models
from MVAE import MVAE
from TimeseriesVAE import TimeseriesVAE
from VideoVAE import VideoVAE
from VideoAutoencoder import VideoAutoencoder
from VideoBert import VideoBERT
from VideoBERT_pretrained import VideoBERT_pretrained
from MAE import MAE
from TimeBERT import TimeBERT
from MidMVAE import MidMVAE
import math
from sklearn.metrics import confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

def reg_loss(model):
    # Regularization term
    reg_loss = 0
    for param in model.parameters():
        reg_loss += torch.sum(torch.square(param))
    # Total loss
    return 0.1 * reg_loss 

class SimpleModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        x = self.fc1(x)
        x = nn.functional.leaky_relu(x)
        x = self.fc2(x)
        #x = nn.functional.softmax(x, dim=1)
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

def train_test_classification(model, epochs = 100, lr = 0.1, latent_dim = 32, hidden_dim = 512, hidden_layers = [[128, 256, 512, 512], 256, 3], split_size = 1):
    print("\nStart classification fine-tuning")
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
        simple_model = SimpleModel(latent_dim * 2, hidden_dim, 14).to(device)
    else: 
        simple_model = SimpleModel(latent_dim, hidden_dim, 14).to(device)

    # Define loss function
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([249.0, 250.0, 250.0, 250.0, 133.0, 250.0, 57.0, 250.0, 48.0, 249.0, 248.0, 175.0, 88.0, 188.0]).to(device))

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
        for video, timeseries, label in zip(video_train_loader, timeseries_train_loader, label_train):
            loss = 0
            optimizer.zero_grad()
            label = torch.tensor([int(l)-1 for l in label]).to(device)
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
                loss += criterion(output, label)
            
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
            for video, timeseries, label in zip(video_test_loader, timeseries_test_loader, label_test):
                loss = 0
                label = torch.tensor([int(l)-1 for l in label]).to(device)
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
                        if "VAE" in model_name:
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
                    loss += criterion(output, label)
                loss += reg_loss(simple_model)
                test_loss += loss.item()
        # lr schedule step
        scheduler.step(test_loss) # For plateau
        #scheduler.step() # for other
        test_loss /= (len(video_test_loader.dataset) * split_size)
        test_losses.append(test_loss)
        if test_loss < best_val_loss:
            best_val_loss = test_loss
            best_val_loss_epoch = epoch
            if split_size > 1:
                torch.save(simple_model.state_dict(), f'augmented_models/{model_name}_simple_state.pth')
            else:
                torch.save(simple_model.state_dict(), f'models/{model_name}_simple_state.pth')

        # Print loss
        if ( epoch + 1 ) % 5 == 0:
            print('Epoch: {} \t Train Loss: {:.6f}\t Test Loss: {:.6f}'.format(epoch, train_loss, test_loss), flush = True)

    print("Finished training")
    print(f"Best test loss: {best_val_loss:.6f} at epoch: {best_val_loss_epoch}")

    evaluate(pretrained_model, model_name, latent_dim = latent_dim, hidden_dim = hidden_dim, split_size = split_size)

def evaluate(pretrained_model, model_name, latent_dim = 32, hidden_dim = 256, split_size = 1):
    print("Start evaluation", flush = True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # load simple model
    if model_name == "MAE" or model_name == "HMAE":
        simple_model = SimpleModel(latent_dim * 2, hidden_dim, 14).to(device)
    else: 
        simple_model = SimpleModel(latent_dim, hidden_dim, 14).to(device)
        
    if split_size > 1:
        simple_model.load_state_dict(torch.load(f'augmented_models/{model_name}_simple_state.pth'))
    else:
        simple_model.load_state_dict(torch.load(f'models/{model_name}_simple_state.pth'))
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
    labels = []

    pretrained_model.eval()
    simple_model.eval()
    with torch.no_grad():
        for video, timeseries, label in zip(video_test_loader, timeseries_test_loader, label_test):
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
                y_pred = torch.argmax(output, dim = 1)
                y_pred = torch.add(y_pred, 1)

                y_preds.append(y_pred.to("cpu"))
                label = torch.tensor([int(l) for l in label])
                labels.append(label)

    y_preds = torch.cat(y_preds, dim = 0)
    labels = torch.cat(labels, dim = 0)
    
    cm = confusion_matrix(labels, y_preds)
    f1 = f1_score(labels, y_preds, average="weighted")
    label_ticks = [str(i) for i in range(1, 15)]

    print(f"\nEvaluation of fine-tuned {model_name} model\n")

    print("Confusion matrix")
    print(label_ticks)
    for idx, line in enumerate(cm):
        print(f"{idx+1} {line}")

    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.set(font_scale=1.2) # adjust the font size
    sns.heatmap(cm_norm, annot=False, fmt='.2f', xticklabels= label_ticks, yticklabels=label_ticks, cmap='Reds')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    #plt.title('Confusion Matrix')
    plt.savefig(f"results/classification_results/{model_name}_confusion_matrix")
    
    print(f"F1 Score: {f1}")

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    train_test_classification(TimeBERT, epochs=65, lr=0.1, latent_dim=64, hidden_dim = 1024, hidden_layers=[[128, 256, 512, 512], 1024, 3], split_size=4)