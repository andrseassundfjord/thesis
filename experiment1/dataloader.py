import sys
import numpy as np
import pandas as pd
import torch
import os
import pickle
import math
import matplotlib.pyplot as plt

def get_min_max_timestamps(series):
    min, max = series[0].index[0], series[0].index[-1]
    for i in series.index:
        if series[i].index[0] < min: min = series[i].index[0]
        if series[i].index[-1] > max : max = series[i].index[-1]
    return min, max

def compare_timestamps():
    baseline = pd.read_csv("results/test_baseline.csv")
    test = pd.read_csv("results/test2.csv")
    baseline = baseline.set_index(baseline["Unnamed: 0"])
    test = test.set_index(test["Unnamed: 0"])
    plt.plot(baseline.index, baseline["data/driver/eye_opening_rate"], color="red", marker="o")
    plt.plot(test.index, test["data/driver/eye_opening_rate"], ls="--", color="blue")
    plt.savefig("results/compare_timestamps")

def save_single_df(path, n_samples = 200):
        """
        Reformat pickled series to single dataframe, with specified number of samples.
        """
        files = os.listdir(path)
        for filename in files:
            series = pd.read_pickle(path + "/" + filename)
            series = series.apply(lambda df: df.set_index(pd.to_datetime(df.index, unit='ns')))
            min, max = get_min_max_timestamps(series)
            time_range = pd.date_range(start=min, end=max, periods=n_samples)
            interval = (max - min) / (n_samples - 1)
            df = pd.DataFrame()
            for df_name in series.index:
                df = df.join(series[df_name].add_suffix(df_name), how = "outer")
            df.to_csv("results/test_baseline.csv")
            resampled_df = df.resample(interval).mean(numeric_only = True).interpolate()
            resampled_df.to_csv("results/test2.csv")
            print(resampled_df.index)
            return
            #resampled_df.to_pickle("single_df_pickles/{}.pkl".format(filename))

if __name__ == "__main__":
    path = "/work5/share/NEDO/nedo-2019/data/processed_rosbags_topickles/fixed_pickles"
    #save_single_df(path)
    compare_timestamps()
