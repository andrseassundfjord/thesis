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

class VideoDataset(Dataset):
    def __init__(self, file_paths):
        self.file_paths = file_paths
        self.frame_counter = {}

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        cap = cv2.VideoCapture(self.file_paths[idx])
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (128, 128))
            frames.append(frame)
        cap.release()
        video = np.stack(frames)
        while len(video) < 182:
            empty_frame = np.zeros_like(video[0])  # Create an empty frame with the same shape as the first frame
            video = np.concatenate((video, empty_frame[np.newaxis, ...]), axis=0)  # Append the empty frame to the end of the video array
        #video = np.transpose(video, (1, 3, 128, 128)) # Shape: (num_frames, num_channels, height, width)
        video = torch.from_numpy(video).float() / 255.0 # Normalize pixel values to [0, 1]
        return video

class TimeseriesDataset(Dataset):
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

        # Extract the data columns and convert to a numpy array
        data = data.values.astype(np.float32)
        return timestamps, data

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
                sample[feature] = pd.Series(dtype='float64')
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
    train_dataset = ConcatDataset([video_dataset_train, timeseries_dataset_train])
    test_dataset = ConcatDataset([video_dataset_test, timeseries_dataset_test])
    # Make dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    if save:
        torch.save(train_dataloader, 'dataloaders/train_dataloader.pth')
        torch.save(test_dataloader, 'dataloaders/test_dataloader.pth')
    return train_dataloader, test_dataloader

def get_dataloaders(pickle_path, video_path, train_ratio = 0.7, batch_size = 32, save = False, load = False):
    if load:
        train_dataloader = torch.load('dataloaders/train_dataloader.pth')
        test_dataloader = torch.load('dataloaders/test_dataloader.pth')
        return train_dataloader, test_dataloader
    class_dict = load_data(pickle_path, video_path)
    return split_data(class_dict, train_ratio = train_ratio, batch_size = batch_size, save = save)

if __name__ == "__main__":
    class_dict = load_data('../experiment1/resampled_pickles', "/work5/share/NEDO/nedo-2019/data/01_driving_data/movie")
    #train_loader, test_loader = split_data(class_dict)