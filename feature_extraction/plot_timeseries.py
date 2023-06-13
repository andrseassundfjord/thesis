import torch
from VideoVAE import VideoVAE
import cv2
import random
import numpy as np
import os
import math
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
from load_dataset import get_dataloaders, VideoDataset, DataFrameTimeseriesDataset
from TimeseriesVAE import TimeseriesVAE
from MVAE import MVAE
from MAE import MAE
from MidMVAE import MidMVAE
import torch.nn.functional as F
from PIL import Image

def make_plots(tensors, model_name):
    x_np = np.linspace(1, 64, 64)
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))
    # Plot the first subplot
    axes[0, 0].plot(x_np, tensors[0], label='Reconstructed')
    axes[0, 0].plot(x_np, tensors[1], label='Original')
    axes[0, 0].set_xlabel('Timestep')
    axes[0, 0].set_ylabel('Heart Rate (BPM)')
    axes[0, 0].set_title('Plot of heart rate')
    axes[0, 0].legend()

    # Plot the second subplot
    axes[0, 1].plot(x_np, tensors[2], label='Reconstructed')
    axes[0, 1].plot(x_np, tensors[3], label='Original')
    axes[0, 1].set_xlabel('Timestep')
    axes[0, 1].set_ylabel('State')
    axes[0, 1].set_title('Plot of turn signal state')
    axes[0, 1].legend()

    # Plot the third subplot
    axes[1, 0].plot(x_np, tensors[4], label='Reconstructed')
    axes[1, 0].plot(x_np, tensors[5], label='Original')
    axes[1, 0].set_xlabel('Timestep')
    axes[1, 0].set_ylabel('Distance (m)')
    axes[1, 0].set_title('Plot of distance to pedestrian')
    axes[1, 0].legend()

    # Plot the fourth subplot
    axes[1, 1].plot(x_np, tensors[6], label='Reconstructed')
    axes[1, 1].plot(x_np, tensors[7], label='Original')
    axes[1, 1].set_xlabel('Timestep')
    axes[1, 1].set_ylabel('Relative Speed (m/s)')
    axes[1, 1].set_title('Plot of relative speed to other vehicle')
    axes[1, 1].legend()

    # Adjust the spacing between subplots
    plt.tight_layout()

    # Show the plot
    plt.savefig(f"results/reconstructed_timeseries/{model_name}")

def save_video(video, decoded, split_size, model_name):
    video = video[0]
    decoded = decoded[0]
    video_shape = [1, 3, 64 // split_size, 128, 128]

    # Save video as mp4 file using OpenCV
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('results/videos/{}_original.mp4'.format(model_name), fourcc, 30, (video_shape[4], video_shape[3]), isColor=True)
    video = video.squeeze(0)
    video = video * 255.0
    video = video.permute(1, 2, 3, 0).detach().cpu().numpy()
    first_frame = video[0, :, :, :]
    first_frame = np.uint8(first_frame)
    img = TF.to_pil_image(first_frame)
    #img = Image.fromarray(first_frame, "RGB")
    img.save(f"results/videos/{model_name}_original.png")
    for i in range(video_shape[2]):
        frame = video[i, :, :, :]
        frame = np.uint8(frame)
        frame = TF.to_pil_image(frame)
        out.write(np.array(frame))
    out.release()
    # Save decoded as mp4 file
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('results/videos/{}_reconstructed.mp4'.format(model_name), fourcc, 30, (video_shape[4], video_shape[3]), isColor=True)
    decoded = decoded.squeeze(0)
    decoded = decoded * 255.0
    decoded = decoded.permute(1, 2, 3, 0).detach().cpu().numpy()
    first_frame = decoded[0, :, :, :]
    first_frame = np.uint8(first_frame)
    img = TF.to_pil_image(first_frame)
    #img = Image.fromarray(first_frame, "RGB")
    img.save(f"results/videos/{model_name}_recon.png")
    for i in range(video_shape[2]):
        frame = decoded[i, :, :, :]
        frame = np.uint8(frame)
        frame = TF.to_pil_image(frame)
        out.write(np.array(frame))
    out.release()

def get_timeseries():
    video_train_loader, video_test_loader, timeseries_train_loader, timeseries_test_loader, label_train, label_test, risk_train, risk_test = get_dataloaders(
                                                    '/work5/share/NEDO/nedo-2019/data/processed_rosbags_topickles/fixed_pickles', 
                                                    "/work5/share/NEDO/nedo-2019/data/01_driving_data/movie", 
                                                    train_ratio = 0.7,
                                                    batch_size = 32, 
                                                    save = True,
                                                    load = True
                                                )
    for i, (video, timeseries) in enumerate(zip(video_train_loader, timeseries_train_loader)):
        return video, timeseries

def prep_timeseries(timeseries):
    masks = []
    for idx, t in enumerate(timeseries):
        nan_mask = torch.isnan(t)
        # Replace NaN values with 0 using boolean masking
        t[nan_mask] = -99
        missing_mask = t.eq(-99)
        # Replace -99 with -1
        #t[missing_mask] = 0.0
        mask = nan_mask | missing_mask
        masks.append(mask)
        # If features are continous
        if idx in [0, 3, 5]:
            timeseries[idx] = F.normalize(t, p=1, dim=-1)
    return timeseries

def save_plots(model_arg, latent_dim, hidden_dim, split_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Get arguments from file
    # Define the model architecture, make sure it matches with model 
    model = model_arg(input_dims= [(64 // split_size, 256, 256, 3), (256 // split_size, 352)], latent_dim=latent_dim, hidden_layers = [[32, 64, 128, 256], hidden_dim, 2], dropout= 0.2).to(device)
    model_name = model.__class__.__name__
    # Load the model state
    if split_size > 1:
        model.load_state_dict(torch.load(f'augmented_models/{model_name}_state.pth'))
    else: 
        model.load_state_dict(torch.load(f'models/{model_name}_state.pth'))

    # Set the model to evaluation mode

    video, timeseries = get_timeseries()
    timeseries_slices = [[] for _ in range(split_size)]
    for t in timeseries:
        split_t = torch.split(t, t.size(1) // split_size, dim = 1)
        for idx, split in enumerate(split_t):
            timeseries_slices[idx].append(split)
    timeseries = timeseries_slices[0]
    timeseries = [t.to(device) for t in timeseries]
    timeseries = prep_timeseries(timeseries)

    model.eval()
    if "Time" in model_name:
        recon_timeseries, kl_divergence, latent_representation, mus = model(timeseries)
    elif "Video" in model_name:
        video_slices = torch.split(video, video.size(2) // split_size, dim=2)
        video = video_slices[0].to(device)
        if model_name == "VideoAutoencoder":
            recon_video, latent_representation = model(video)   
        else:
            recon_video, kl_divergence, latent_representation, mus = model(video)         
    else:
        video_slices = torch.split(video, video.size(2) // split_size, dim=2)
        video = video_slices[0].to(device)
        recon_video, recon_timeseries, kl_divergence, latent_representation, mus = model((video, timeseries))
    
    if "Video" not in model_name:
        y11 = recon_timeseries[0][1, :, 5].detach().to("cpu").numpy()
        y12 = timeseries[0][1, :, 5].detach().to("cpu").numpy()
        y21 = recon_timeseries[2][0, :, 44].detach().to("cpu").numpy()
        y22 = timeseries[2][2, :, 9].detach().to("cpu").numpy()
        y31 = recon_timeseries[1][2, :, 0].detach().to("cpu").numpy()
        y32 = timeseries[3][2, :, 0].detach().to("cpu").numpy()
        y41 = recon_timeseries[2][2, :, 2].detach().to("cpu").numpy()
        y42 = timeseries[5][2, :, 2].detach().to("cpu").numpy()

        make_plots([y11, y12, y21, y22, y31, y32, y41, y42], model_name)

    if "Time" not in model_name:
        save_video(video, recon_video, split_size=split_size, model_name=model_name)

if __name__ == "__main__":
    latent_dim = 32
    hidden_dim = 512
    split_size = 4
    save_plots(TimeseriesVAE, latent_dim, hidden_dim, split_size)
    print("Finished")