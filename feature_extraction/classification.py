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
from HMAE import HMAE
import math
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
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

        init.uniform_(self.fc1.weight)
        init.uniform_(self.fc2.weight)
    
    def forward(self, x):
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        x = nn.functional.relu(x)

        return x

def prep_timeseries(timeseries):
    masks = []
    for idx, t in enumerate(timeseries):
        nan_mask = torch.isnan(t)
        # Replace NaN values with 0 using boolean masking
        t[nan_mask] = -99
        missing_mask = t.eq(-99)
        # Replace -99 with -1
        t[missing_mask] = -99
        mask = nan_mask | missing_mask
        masks.append(mask)
        # If features are continous
        if idx in [0, 3, 5]:
            t[mask] = 0.000000001
            timeseries[idx] -= timeseries[idx].min(-1, keepdim=True)[0]
            timeseries[idx] /= torch.add(timeseries[idx].max(-1, keepdim=True)[0], 0.000000001)
            nans = torch.isnan(timeseries[idx])
            timeseries[idx][nans] = 0.5
            t[mask] -99

    return timeseries

def train_test_classification(model, epochs = 100, lr = 0.1, latent_dim = 32, hidden_dim = 512, hidden_layers = [[128, 256, 512, 512], 256, 3], split_size = 1, classes_list = None):
    print(f"lr: {lr}, hidden_dim: {hidden_dim}", flush = True)
    if classes_list == None:
        classes_list = range(14)
    print("\nStart classification fine-tuning")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Define the model architecture
    pretrained_model = model(input_dims= [(64 // split_size, 256, 256, 3), (256 // split_size, 352)], latent_dim=latent_dim, 
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
        simple_model = SimpleModel(latent_dim * 2, hidden_dim, len(classes_list)).to(device)
    else: 
        simple_model = SimpleModel(latent_dim, hidden_dim, len(classes_list)).to(device)

    # Define loss function
    weights_vals = [249.0, 250.0, 250.0, 250.0, 133.0, 250.0, 57.0, 250.0, 48.0, 249.0, 248.0, 175.0, 88.0, 188.0]
    weights = [weights_vals[i] for i in classes_list]
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights).to(device))

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
            masks = []
            for val in classes_list:
                masks.append(label.eq(val))
            mask = masks[0]
            for m in masks[1:]:
                mask = mask | m

            label = label[mask]
            if label.size(0) == 0:
                continue
            for c_idx, c in enumerate(classes_list):
                label[label.eq(c)] = c_idx
            video_slices = torch.split(video, video.size(2) // split_size, dim=2)
            timeseries_slices = [[] for _ in range(split_size)]
            for t in timeseries:
                split_t = torch.split(t, t.size(1) // split_size, dim = 1)
                for idx, split in enumerate(split_t):
                    timeseries_slices[idx].append(split.to(device)[mask])
            for i in range(split_size):
                video = video_slices[i].to(device)
                video = video[mask]
                timeseries = timeseries_slices[i]
                if "Video" in model_name:
                    if model_name != "VideoAutoencoder":
                        recon_video, kl_divergence, latent_representation, mus = pretrained_model(video)
                        latent = mus
                    else: 
                        recon_video, latent_representation = pretrained_model(video)
                        latent = latent_representation
                elif "Time" in model_name:
                    timeseries = prep_timeseries(timeseries)
                    recon_timeseries, kl_divergence, latent_representation, mus = pretrained_model(timeseries)
                    latent = mus                   
                else:
                    timeseries = prep_timeseries(timeseries)
                    recon_video, recon_timeseries, kl_divergence, latent_representation, mus = pretrained_model([video, timeseries])
                    latent = mus

                output = simple_model(latent_representation)
                loss += criterion(output, label)
            
            #loss += reg_loss(simple_model)
            
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
                masks = []
                for val in classes_list:
                    masks.append(label.eq(val))
                mask = masks[0]
                for m in masks[1:]:
                    mask = mask | m

                label = label[mask]
                if label.size(0) == 0:
                    continue
                for c_idx, c in enumerate(classes_list):
                    label[label.eq(c)] = c_idx
                video_slices = torch.split(video, video.size(2) // split_size, dim=2)
                timeseries_slices = [[] for _ in range(split_size)]
                for t in timeseries:
                    split_t = torch.split(t, t.size(1) // split_size, dim = 1)
                    for idx, split in enumerate(split_t):
                        timeseries_slices[idx].append(split.to(device)[mask])
                for i in range(split_size):
                    video = video_slices[i].to(device)
                    video = video[mask]
                    timeseries = timeseries_slices[i]
                    if "Video" in model_name:
                        if model_name != "VideoAutoencoder":
                            recon_video, kl_divergence, latent_representation, mus = pretrained_model(video)
                            latent = mus
                        else: 
                            recon_video, latent_representation = pretrained_model(video)
                            latent = latent_representation
                    elif "Time" in model_name:
                        timeseries = prep_timeseries(timeseries)
                        recon_timeseries, kl_divergence, latent_representation, mus = pretrained_model(timeseries)
                        latent = mus
                    else:
                        timeseries = prep_timeseries(timeseries)
                        recon_video, recon_timeseries, kl_divergence, latent_representation, mus = pretrained_model([video, timeseries])
                        latent = mus

                    output = simple_model(latent_representation)
                    loss += criterion(output, label)
                #loss += reg_loss(simple_model)
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
        if epoch > best_val_loss_epoch + 15:
            break

    plot_loss(train_losses=train_losses, test_losses=test_losses, savename=f"results/loss_plots/simple_class_{model_name}", num_epochs=epochs)

    print("Finished training")
    print(f"Best test loss: {best_val_loss:.6f} at epoch: {best_val_loss_epoch}")

    evaluate(pretrained_model, model_name, latent_dim = latent_dim, hidden_dim = hidden_dim, split_size = split_size, classes_list = classes_list)

def evaluate(pretrained_model, model_name, latent_dim = 32, hidden_dim = 256, split_size = 1, classes_list = []):
    print("Start evaluation, classes for ", classes_list, flush = True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # load simple model
    if model_name == "MAE" or model_name == "HMAE":
        simple_model = SimpleModel(latent_dim * 2, hidden_dim, len(classes_list)).to(device)
    else: 
        simple_model = SimpleModel(latent_dim, hidden_dim, len(classes_list)).to(device)
        
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
            label = torch.tensor([int(l)-1 for l in label])
            masks = []
            for val in classes_list:
                masks.append(label.eq(val))
            mask = masks[0]
            for m in masks[1:]:
                mask = mask | m

            label = label[mask]
            if label.size(0) == 0:
                continue
            for c_idx, c in enumerate(classes_list):
                label[label.eq(c)] = c_idx
            video_slices = torch.split(video, video.size(2) // split_size, dim=2)
            timeseries_slices = [[] for _ in range(split_size)]
            for t in timeseries:
                split_t = torch.split(t, t.size(1) // split_size, dim = 1)
                for idx, split in enumerate(split_t):
                    timeseries_slices[idx].append(split.to(device)[mask])
            for i in range(split_size):
                video = video_slices[i].to(device)
                video = video[mask]
                timeseries = timeseries_slices[i]
                if "Video" in model_name:
                    #video = video.to(device)
                    if model_name != "VideoAutoencoder":
                        recon_video, kl_divergence, latent_representation, mus = pretrained_model(video)
                        latent = mus
                    else: 
                        recon_video, latent_representation = pretrained_model(video)
                        latent = latent_representation
                elif "Time" in model_name:
                    #timeseries = [t.to(device) for t in timeseries]
                    timeseries = prep_timeseries(timeseries)
                    recon_timeseries, kl_divergence, latent_representation, mus = pretrained_model(timeseries)
                    latent = mus
                else:
                    #video = video.to(device)
                    #timeseries = [t.to(device) for t in timeseries]
                    timeseries = prep_timeseries(timeseries)
                    recon_video, recon_timeseries, kl_divergence, latent_representation, mus = pretrained_model([video, timeseries])
                    latent = mus
                
                output = simple_model(latent_representation)
                y_pred = torch.argmax(output, dim = 1)

                y_preds.append(y_pred.to("cpu"))
                labels.append(label)

    y_preds = torch.cat(y_preds, dim = 0)
    labels = torch.cat(labels, dim = 0)
    
    cm = confusion_matrix(labels, y_preds)
    f1 = f1_score(labels, y_preds, average="weighted")
    acc = accuracy_score(labels, y_preds)
    label_ticks = [str(c) for c in classes_list]

    print(f"\nEvaluation of fine-tuned {model_name} model\n")

    #print("Confusion matrix")
    #print(label_ticks)
    #for idx, line in enumerate(cm):
    #    print(f"{idx+1} {line}")

    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.clf()
    sns.set(font_scale=1.5, rc = {'figure.figsize':(15, 12)}) # adjust the font size
    sns.heatmap(cm_norm, annot=False, fmt='.2f', xticklabels= label_ticks, yticklabels=label_ticks, cmap='Reds')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(f"results/classification_results/{model_name}_confusion_matrix")
    
    print(f"F1 Score: {f1}")
    print(f"Accuracy: {acc}", flush = True)


def plot_loss(train_losses, test_losses, savename, num_epochs):
    # Plot the training and testing losses and accuracies
    plt.clf()
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.plot(np.linspace(1, len(train_losses), len(train_losses)), train_losses, label='Training')
    ax.plot(np.linspace(1, len(train_losses), len(train_losses)), test_losses, label='Testing')
    ax.set_title('Loss over Epochs')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    plt.savefig(savename)

def calc_std():
    f1_scores = np.array([0.2932960611074359, 0.2553308175229437, 0.3043153150134501, 0.3605155252845558, 0.3134652628817942, 0.33146839614905904, 0.3098342406252461, 0.3136696453886199, 0.30997928956331405, 0.2624863840731026])
    acc_scores = np.array([0.35777777777777775, 0.31444444444444447, 0.3388888888888889, 0.3933333333333333, 0.35777777777777775, 0.37444444444444447, 0.34444444444444444, 0.35444444444444445, 0.3466666666666667, 0.31333333333333335])
    print(f"F1: \n Mean: {np.mean(f1_scores)}, std: {np.std(f1_scores)}")
    print(f"Accuracy: \n Mean: {np.mean(acc_scores)}, std: {np.std(acc_scores)}")

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    hidden = 512
    latent_dim = 32
    calc_std()
    #train_test_classification(VideoAutoencoder, epochs=100, lr=0.001, latent_dim=latent_dim, hidden_dim = 512, hidden_layers=[[32, 64, 128, 256], hidden, 2], split_size=4, classes_list = None)
    #or i in range(10):
    #    train_test_classification(VideoAutoencoder, epochs=100, lr=0.001, latent_dim=latent_dim, hidden_dim = 512, hidden_layers=[[32, 64, 128, 256], hidden, 2], split_size=4, classes_list = [1, 5, 9])