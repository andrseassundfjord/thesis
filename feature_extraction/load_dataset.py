import os
import cv2
import torch
import glob
import pickle
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import ConcatDataset
from torchvision import transforms
import random
import pandas as pd
import math

import ffmpeg

class VideoDataset_ffmpeg(Dataset):
    def __init__(self, file_paths):
        self.file_paths = file_paths

        # Set output video resolution to 480p
        self.resolution = '854x480'

        # Set output video bit rate to 500 kbps
        self.bitrate = '500k'

    def __getitem__(self, idx):
        # Get the file path of the video
        path = self.file_paths[idx]

        # Create a video stream and set the resolution and bit rate
        stream = ffmpeg.input(path).video.filter('scale', self.resolution).filter('bitrate', self.bitrate)

        # Create an output stream with H.264 video codec and AAC audio codec
        output = ffmpeg.output(stream, 'pipe:', format='rawvideo', pix_fmt='rgb24')

        # Run the ffmpeg command and read the output as a numpy array
        out, _ = ffmpeg.run(output, capture_stdout=True)
        video = np.frombuffer(out, np.uint8).reshape([-1, 480, 854, 3])

        # Normalize pixel values to [0, 1]
        video = video.astype(np.float32) / 255.0

        # Return the processed video
        return video

    def __len__(self):
        return len(self.file_paths)

class VideoDataset(Dataset):
    def __init__(self, file_paths, frame_len = 64, size = 128):
        self.file_paths = file_paths
        self.num_frames = {}
        self.frame_len = frame_len
        self.size = size # Height and width
        self.videos = self.get_videos()

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        return self.videos[idx]

    def get_videos(self):
        videos = []
        for idx, _ in enumerate(self.file_paths): 
            cap = cv2.VideoCapture(self.file_paths[idx])
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
                if counter % math.floor(n_frames / self.frame_len) == 0:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = cv2.resize(frame, (self.size, self.size))
                    frames.append(frame)
                counter += 1
            frames = [frames[i] for i in sorted(random.sample(range(len(frames)), self.frame_len))]
            cap.release()
            video = np.stack(frames)
            if n_frames in self.num_frames:
                self.num_frames[n_frames] += 1
            else: 
                self.num_frames[n_frames] = 1
            video = np.transpose(video, (3, 0, 1, 2)) # new order of shape: (num_channels, num_frames, height, width)
            video = torch.from_numpy(video).float() / 255.0 # Normalize pixel values to [0, 1]
            videos.append(video)
        return videos

class RosbagTimeseriesDataset(Dataset):
    def __init__(self, bag_file_paths, topic_name):
        self.bag_file_paths = bag_file_paths
        self.topic_name = topic_name
        
        # Read all the messages from the specified topic in all the bag files and concatenate them into a single numpy array
        self.data = []
        for bag_file_path in self.bag_file_paths:
            bag = rosbag.Bag(bag_file_path)
            for _, msg, _ in bag.read_messages(topics=[self.topic_name]):
                self.data.append(msg.data)
            bag.close()
        self.data = np.concatenate(self.data)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

class DataFrameTimeseriesDataset(Dataset):
    def __init__(self, dataframes):
        self.dataframes = dataframes

    def __len__(self):
        return len(self.dataframes)

    def __getitem__(self, idx):
        data = self.dataframes[idx]
        # Extract the timestamp column and convert to a numpy array
        timestamps = data.index.values.astype(np.int64) // 10 ** 9
        # Use the timestamps for anything?
        # Extract the data columns and convert to a numpy array
        data = data.values.astype(np.float32)
        return data

def load_data(folder_path, video_path):
    class_dict = {}
    all_features = set(pd.read_csv("stats/feature_statistics.csv").columns[1:])
    for file_path in glob.glob(os.path.join(folder_path, '*.pkl')):
        filename = file_path.split("/")[-1].split(".")[0]
        video_name = glob.glob(os.path.join(video_path, filename + '.mp4'))
        if len(video_name) == 1:
            video_name = video_name[0]
            with open(file_path, 'rb') as f:
                sample = pickle.load(f)
            label = video_name.split(".")[0].split("_")[-1]
            if label not in class_dict:
                class_dict[label] = [[], []]
            missing_features = all_features - set(sample.columns)
            for feature in missing_features:
                sample[feature] = pd.Series(dtype='float32')
                sample = sample.copy()
            sample_sorted = sample.sort_index(axis = 1)
            class_dict[label][0].append(sample_sorted)
            class_dict[label][1].append(video_name)
        else:
            print(video_name, file_path)
    return class_dict

def split_data(class_dict, train_ratio=0.7, batch_size = 32, save = False):
    video_list_train = []
    video_list_test = []
    timeseries_list_train = []
    timeseries_list_test = []
    train_labels = []
    test_labels = []
    
    for label in class_dict:
        samples = class_dict[label][0]
        videos = class_dict[label][1]
        n_train = int(len(samples) * train_ratio)
        combined = list(zip(samples, videos))  # Combine samples and videos into tuples
        random.shuffle(combined)  # Shuffle the combined list
        samples, videos = zip(*combined)  # Unpack the shuffled tuples back into separate lists
        # Add video 
        video_list_train += videos[:n_train]
        video_list_test += videos[n_train:]
        # Add timeseries data
        timeseries_list_train += samples[:n_train]
        timeseries_list_test += samples[n_train:]
        # Add labels
        train_labels += [label] * n_train
        test_labels += [label] * (len(samples) - n_train)
        
    # Define the data loaders
    video_dataset_train = VideoDataset(video_list_train)
    video_dataset_test = VideoDataset(video_list_test)
    timeseries_dataset_train = DataFrameTimeseriesDataset(timeseries_list_train)
    timeseries_dataset_test = DataFrameTimeseriesDataset(timeseries_list_train)
    # Concat datasets
    #train_dataset = ConcatDataset([video_dataset_train, timeseries_dataset_train])
    #test_dataset = ConcatDataset([video_dataset_test, timeseries_dataset_test])
    # Make dataloaders
    #train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    #test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    video_train_dataloader = DataLoader(video_dataset_train, batch_size=batch_size, shuffle=True)
    video_test_dataloader = DataLoader(video_dataset_test, batch_size=batch_size, shuffle=False)
    timeseries_train_dataloader = DataLoader(timeseries_dataset_train, batch_size=batch_size, shuffle=True)
    timeseries_test_dataloader = DataLoader(timeseries_dataset_test, batch_size=batch_size, shuffle=False)
    if save:
        torch.save(video_train_dataloader, 'dataloaders/video_train_dataloader.pth')
        torch.save(video_test_dataloader, 'dataloaders/video_test_dataloader.pth')
        torch.save(timeseries_train_dataloader, 'dataloaders/timeseries_train_dataloader.pth')
        torch.save(timeseries_test_dataloader, 'dataloaders/timeseries_test_dataloader.pth')
    return video_train_dataloader, video_test_dataloader, timeseries_train_dataloader, timeseries_test_dataloader

def get_dataloaders(pickle_path, video_path, train_ratio = 0.7, batch_size = 32, save = False, load = False):
    if load:
        video_train_dataloader = torch.load('dataloaders/video_train_dataloader.pth')
        video_test_dataloader = torch.load('dataloaders/video_test_dataloader.pth')
        timeseries_train_dataloader = torch.load('dataloaders/timeseries_train_dataloader.pth')
        timeseries_test_dataloader = torch.load('dataloaders/timeseries_test_dataloader.pth')        
        return video_train_dataloader, video_test_dataloader, timeseries_train_dataloader, timeseries_test_dataloader
    class_dict = load_data(pickle_path, video_path)
    return split_data(class_dict, train_ratio = train_ratio, batch_size = batch_size, save = save)

if __name__ == "__main__":
    print("Start")
    class_dict = load_data('../experiment1/resampled_pickles', "/work5/share/NEDO/nedo-2019/data/01_driving_data/movie")
    #train_loader, test_loader = split_data(class_dict)