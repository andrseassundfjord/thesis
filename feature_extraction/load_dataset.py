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
import time

class LabelDataset(Dataset):
    def __init__(self, labels):
        self.labels = labels 

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.labels[idx]

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

class DataFrameTimeseriesDataset(Dataset):
    def __init__(self, dataframes, n_samples = 200):
        """
        dataframes: list of dataframes, with missing all columns
        """
        self.dataframes = dataframes
        self.n_samples = n_samples
        self.split_list = self.split_dataframes()

    def split_dataframes(self):
        """
        Function to split df into multiple dataframes. Seperate by type standard, mobileye_pedestrian, and mobileye_cars

        """
        # read in the text files and store the column names in a list
        file_list = ["stats/standard_cont.txt","stats/standard_cat1.txt", "stats/standard_cat2.txt",
                     "stats/mobileye_pedestrian_cont.txt", "stats/mobileye_pedestrian_cat2.txt",
                     "stats/mobileye_cars_cont.txt", "stats/mobileye_cars_cat1.txt", "stats/mobileye_cars_cat2.txt"]
        col_lists = []
        for file_name in file_list:
            with open(file_name, 'r') as f:
                col_lists.append([line.strip() for line in f])
        # Expand Mobileye lists so that it contains all possible mobileye features
        n = 64  # number of times to multiply the list
        for i, c_list in enumerate(col_lists[3:]):
            num = 16 if (i+3) > 4 else 7
            new_list = []
            for j in range(n):
                for col in c_list:
                    number = int(col.split('/')[-1]) + (num * j)
                    new_list.append(f"/mobileye/{number}/data")
            col_lists[i+3] = new_list
        split_list = []
        
        for df in self.dataframes:
            df_dict = {}
            # iterate over the list of column name lists and create a dataframe for each
            for i, cols in enumerate(col_lists):
                df_name = file_list[i].split("/")[-1].split(".")[0]
                valid_cols = [col for col in cols if col in df.columns]
                if len(valid_cols) == 0 and "standard" not in df_name:
                    df_dict[df_name] = pd.DataFrame(data=[-999]*self.n_samples, columns=['empty']).values.astype(np.float32)
                else:
                    split_df = df[valid_cols].copy()
                    # Resample df, and interpolate or ffill based on continous or categorical values
                    interpolate = "cont" in df_name
                    split_df.set_index(df.index, inplace=True)
                    resampled_df = self.resample_dataframe(split_df, interpolate=interpolate)
                    # Add all mandatory features
                    if "standard" in df_name:
                        # if any columns are missing, add them as empty columns
                        missing_cols = set(cols) - set(df.columns)
                        missing_df = pd.DataFrame(columns=list(missing_cols))
                        resampled_df = pd.concat([resampled_df, missing_df], axis=1) 
                    df_dict[df_name] = resampled_df.values.astype(np.float32)
            # Add dict to list
            split_list.append(df_dict)

        return split_list
    
    def resample_dataframe(self, df, interpolate = False):
        # Replace True/False with 1/0
        df = df.replace(True, 1)
        df = df.replace(False, 0)
        if df.isna().all().all():
            empty_df = pd.DataFrame(np.empty((self.n_samples, len(df.columns))))
            empty_df[:] = np.nan
            empty_df.columns = df.columns
            return empty_df
        # resample to n_samples samples
        min, max = df.index[0], df.index[-1]
        interval = (max - min) / (self.n_samples - 1)
        if interpolate:
            resampled_df = df.resample(interval).mean(numeric_only = True).interpolate().bfill()
        else:
            #resampled_df = df.resample(interval).mean(numeric_only = True).ffill().bfill()
            resampled_df = df.resample(interval).median(numeric_only = True).ffill().bfill()
        # Some have n_samples + 1 for some reason, remove last if thats the case 
        resampled_df = resampled_df.iloc[:self.n_samples]
        return resampled_df
    
    def __len__(self):
        return len(self.dataframes)

    def __getitem__(self, idx):
        data_dict = self.split_list[idx]
        data_list = [torch.tensor(data) for data in data_dict.values()]
        # Extract the timestamp column and convert to a numpy array
        #timestamps = data.index.values.astype(np.int64) // 10 ** 9
        # Use the timestamps for anything?
        # Extract the data columns and convert to a numpy array
        #data = data.values.astype(np.float32)
        return data_list

def custom_collate(batch, padding_value=-999):
    # Get the longest length in the batch
    max_lens = [max([data[i].shape[1] for data in batch]) for i in range(len(batch[0]))]

    # Initialize list to hold the padded tensors
    padded_batch = [[] for _ in range(len(batch[0]))]

    # Pad each tensor with the custom padding value along its second dimension
    for sample in batch:
        for i, data in enumerate(sample):
            padding_needed = max_lens[i] - data.shape[1]
            padded_data = torch.cat([data, torch.full((data.shape[0], padding_needed), padding_value)], dim=1)
            padded_batch[i].append(padded_data)

    # Stack the tensors along the first dimension
    stacked_tensors = [torch.stack(padded_data_list, dim=0) for padded_data_list in padded_batch]

    return stacked_tensors

def load_data(folder_path, video_path):
    class_dict = {}
    for file_path in glob.glob(os.path.join(folder_path, '*.pkl4')):
        filename = file_path.split("/")[-1].split(".")[0]
        video_name = glob.glob(os.path.join(video_path, filename + '.mp4'))
        if len(video_name) == 1:
            video_name = video_name[0]
            with open(file_path, 'rb') as f:
                sample = pickle.load(f)
            sample_df = series_to_dataframe(sample)
            label = video_name.split(".")[0].split("_")[-1]
            if label not in class_dict:
                class_dict[label] = [[], []]
            class_dict[label][0].append(sample_df)
            class_dict[label][1].append(video_name)
        else:
            print(video_name, file_path)
    return class_dict

def series_to_dataframe(series):
    # Set index to timestamp
    series = series.apply(lambda df: df.set_index(pd.to_datetime(df.index, unit='ns')))
    df = pd.DataFrame()
    # Add columns to dataframe
    for df_name in series.index:
        new_cols = series[df_name]
        if "header" in new_cols.columns:
            new_cols = new_cols.drop(columns = ["header"])
        # Unpack dict into columns
        while new_cols.dtypes.iloc[0] == "object":
            for col in new_cols.columns:
                new_cols = pd.concat([new_cols.drop([col], axis=1), new_cols[col].apply(pd.Series).add_prefix('{}/'.format(col))], axis=1)
        df = df.join(new_cols.add_prefix("{}/".format(df_name)), how = "outer")
    return df

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
    timeseries_dataset_test = DataFrameTimeseriesDataset(timeseries_list_test)
    label_dataset_train = LabelDataset(train_labels)
    label_dataset_test = LabelDataset(test_labels)
    # Concat datasets
    #train_dataset = ConcatDataset([video_dataset_train, timeseries_dataset_train])
    #test_dataset = ConcatDataset([video_dataset_test, timeseries_dataset_test])
    # Make dataloaders
    #train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    #test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    video_train_dataloader = DataLoader(video_dataset_train, batch_size=batch_size, shuffle=True)
    video_test_dataloader = DataLoader(video_dataset_test, batch_size=batch_size, shuffle=False)
    timeseries_train_dataloader = DataLoader(timeseries_dataset_train, batch_size=batch_size, shuffle=True, collate_fn = custom_collate)
    timeseries_test_dataloader = DataLoader(timeseries_dataset_test, batch_size=batch_size, shuffle=False, collate_fn = custom_collate)
    label_dataloader_train = DataLoader(label_dataset_train, batch_size = batch_size, shuffle=True)
    label_dataloader_test = DataLoader(label_dataset_test, batch_size=batch_size, shuffle=True)
    if save:
        torch.save(video_train_dataloader, 'dataloaders/video_train_dataloader.pth')
        torch.save(video_test_dataloader, 'dataloaders/video_test_dataloader.pth')
        torch.save(timeseries_train_dataloader, 'dataloaders/timeseries_train_dataloader.pth')
        torch.save(timeseries_test_dataloader, 'dataloaders/timeseries_test_dataloader.pth')
        torch.save(label_dataloader_train, 'dataloaders/label_train_dataloader.pth')
        torch.save(label_dataloader_test, 'dataloaders/label_test_dataloader.pth')
    return video_train_dataloader, video_test_dataloader, timeseries_train_dataloader, timeseries_test_dataloader, label_dataloader_train, label_dataloader_test

def get_dataloaders(pickle_path, video_path, train_ratio = 0.7, batch_size = 32, save = False, load = False):
    if load:
        video_train_dataloader = torch.load('dataloaders/video_train_dataloader.pth')
        video_test_dataloader = torch.load('dataloaders/video_test_dataloader.pth')
        timeseries_train_dataloader = torch.load('dataloaders/timeseries_train_dataloader.pth')
        timeseries_test_dataloader = torch.load('dataloaders/timeseries_test_dataloader.pth')
        label_train = torch.load('dataloaders/label_train_dataloader.pth')
        label_test = torch.load("dataloaders/label_test_dataloader.pth")
        return video_train_dataloader, video_test_dataloader, timeseries_train_dataloader, timeseries_test_dataloader, label_train, label_test
    class_dict = load_data(pickle_path, video_path)
    return split_data(class_dict, train_ratio = train_ratio, batch_size = batch_size, save = save)

if __name__ == "__main__":
    start_time = time.time()
    print("Started", flush = True)
    video_train_loader, video_test_loader, timeseries_train_loader, timeseries_test_loader, l_train, l_test = get_dataloaders(
                                                        '/work5/share/NEDO/nedo-2019/data/processed_rosbags_topickles/fixed_pickles', 
                                                        "/work5/share/NEDO/nedo-2019/data/01_driving_data/movie", 
                                                        save = False, batch_size=32
                                                    )
    end_time = time.time() - start_time
    print("Finished, took {} seconds".format(end_time))
    