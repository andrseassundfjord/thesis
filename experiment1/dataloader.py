import sys
import numpy as np
import pandas as pd
import torch
import os
import pickle

class Dataloader:

    def __init__(self, path):
        self.path = path
        self.filenames = os.listdir(path)
        self.n_files = len(self.filenames)

def get_feature_from_pickle(path, feature_name, save_path="balanced_pickles", n_samples = 200):
        """

        """
        files = os.listdir(path)
        features, labels = [], []
        for filename in files:
            unpickled_df = pd.read_pickle(path + "/" + filename)
            index = unpickled_df.index
            feature = np.zeros(n_samples)
            if feature_name in index:
                feature = unpickled_df[feature_name]
                print(feature.shape)
                feature = feature.resample(n_samples).mean()
                print(feature.shape)
                feature = feature.to_numpy().reshape(feature.shape[0])
                return

            features.append(feature)
            label = filename.split("_")[-1].split(".")[0]
            labels.append(label)
        features = np.array(features)
        labels = np.array(labels)
        print(labels)
        print(features)
        return features, labels

def example():
    # create a sample series with Linux timestamps
    data = [10, 20, 30, 40, 50]
    timestamps = [1623750000, 1623750100, 1623750200, 1623750300, 1623750400]
    series = pd.Series(data=data, index=timestamps)
    # convert Linux timestamps to DatetimeIndex
    series.index = pd.to_datetime(series.index, unit='s')
    # resample the series to have 100 samples
    resampled_series = series.resample('L').mean()
    # interpolate to fill any missing values
    resampled_series = resampled_series.interpolate(method='linear')
    # resample to final number of samples
    final_resampled_series = resampled_series.iloc[::len(resampled_series)//100]

if __name__ == "__main__":
    df = pd.DataFrame({"A": [1, 2], "B": [3.0, 4.5]})
    print(df.resample(10).pad())
    
    #path = "/work5/share/NEDO/nedo-2019/data/processed_rosbags_topickles/fixed_pickles"
    #get_feature_from_pickle(path, '/driver/blink')
