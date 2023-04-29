import torch
from VideoVAE import VideoVAE
from VideoAutoencoder import VideoAutoencoder
import cv2
import random
import numpy as np
import os
import math
import torchvision.transforms.functional as TF

def get_video():
    path = "/work5/share/NEDO/nedo-2019/data/01_driving_data/movie"
    files = os.listdir(path)
    filename = random.choice(files)
    cap = cv2.VideoCapture(path + "/" + filename)
    frames = []
    read_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        read_frames.append(frame)
    n_frames = len(read_frames)
    counter = 0
    for frame in read_frames:
        # Oversample, often giving us too many samples
        if counter % math.floor(n_frames / 64) == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (128, 128))
            frames.append(frame)
        counter += 1
    frames = [frames[i] for i in sorted(random.sample(range(len(frames)), 64))]
    cap.release()
    video = np.stack(frames)
    video = np.transpose(video, (3, 0, 1, 2)) # new order of shape: (num_channels, num_frames, height, width)
    video = torch.from_numpy(video).float() / 255.0 # Normalize pixel values to [0, 1]
    video = video.unsqueeze(0)
    return video, filename

def save_video(model_type):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Get arguments from file
    # Define the model architecture
    model = model_type(input_dims= [(64, 128, 128, 3), (200, 352)], latent_dim=256, 
                    hidden_layers = [[128, 256, 512, 512], 256, 3], dropout= 0.2).to(device)

    model_name = model.__class__.__name__    
    # Load the model state
    model.load_state_dict(torch.load(f'models/{model_name}_state.pth'))

    # Set the model to evaluation mode
    model.eval()

    video, filename = get_video()
    video = video.to(device)
    if "VAE" in model_name:
        decoded, kl, encoded = model(video)
    else: 
        decoded, encoded = model(video)

    filename = filename.split(".")[0]

    video_shape = [1, 3, 64, 128, 128]

    # Save video as mp4 file using OpenCV
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('results/videos/{}_{}_original.mp4'.format(model_name, filename), fourcc, 30, (video_shape[4], video_shape[3]), isColor=True)
    video = video.squeeze(0)
    video = video * 255.0
    video = video.permute(1, 2, 3, 0).detach().cpu().numpy()
    for i in range(video_shape[2]):
        frame = video[i, :, :, :]
        frame = np.uint8(frame)
        frame = TF.to_pil_image(frame)
        out.write(np.array(frame))
    out.release()
    # Save decoded as mp4 file
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('results/videos/{}_{}_reconstructed.mp4'.format(model_name, filename), fourcc, 30, (video_shape[4], video_shape[3]), isColor=True)
    decoded = decoded.squeeze(0)
    decoded = decoded * 255.0
    decoded = decoded.permute(1, 2, 3, 0).detach().cpu().numpy()
    for i in range(video_shape[2]):
        frame = decoded[i, :, :, :]
        frame = np.uint8(frame)
        frame = TF.to_pil_image(frame)
        out.write(np.array(frame))
    out.release()

    print("Finished")

if __name__ == "__main__":
    save_video(VideoAutoencoder)