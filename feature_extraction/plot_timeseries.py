import torch
from VideoVAE import VideoVAE
import cv2
import random
import numpy as np
import os
import math
import torchvision.transforms.functional as TF

def get_timeseries():
    video_train_loader, video_test_loader, timeseries_train_loader, timeseries_test_loader = get_dataloaders(
                                                    '/work5/share/NEDO/nedo-2019/data/processed_rosbags_topickles/fixed_pickles', 
                                                    "/work5/share/NEDO/nedo-2019/data/01_driving_data/movie", 
                                                    train_ratio = train_ratio,
                                                    batch_size = batch_size, 
                                                    save = True,
                                                    load = True
                                                )



def save_plots():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Get arguments from file
    # Define the model architecture, make sure it matches with model 
    model = TimeseriesVAE(input_dims= [(64, 128, 128, 3), (200, 352)], latent_dim=64, hidden_layers = [[32, 64, 128, 256], [32], [3]], dropout= 0.1).to(device)

    # Load the model state
    model.load_state_dict(torch.load('models/TimeseriesVAE_state.pth'))

    # Set the model to evaluation mode
    model.eval()

    timeseries = get_video()
    timeseries = timeseries.to(device)
    decoded, kl, encoded = model(timeseries)

    print("Finished")

if __name__ == "__main__":
    save_plots()