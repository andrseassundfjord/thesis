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

def plot_compare_timestamps():
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

def get_dataset_statistics(path, savepath):
    files = os.listdir(path)
    n_files = len(files)
    feature_statistics = pd.DataFrame(index = [
        "n_samples", "min_value", "max_value", 
        "type", "isnumber", "n_columns", "n_present",
        "n_missing", "mean", "median", "std"
        ])
    class_statistics = pd.DataFrame(index = range(1, 15))
    class_statistics["class_counter"] = np.zeros(14)
    for filename in files:
        series = pd.read_pickle(path + "/" + filename)
        label = filename.split(".")[0].split("_")[-1]
        class_statistics["class_counter"][int(label)] += 1
        for df_name in series.index:
            if df_name not in feature_statistics.columns:
                val_type = str(type(series[df_name].to_numpy().flatten()[0]))
                isnumber = "float" in val_type or "int" in val_type
                feature_statistics[df_name] = [[], math.inf, 0, val_type, isnumber, len(series[df_name].columns), 0, 0, 0, 0, 0]
            feature_statistics[df_name]["n_samples"].append(len(series[df_name]))
            feature_statistics = feature_statistics.copy()
            if feature_statistics[df_name]["isnumber"] and feature_statistics[df_name]["n_columns"] == 1:
                agg = series[df_name].agg(["min", "max"]).to_numpy().flatten()
                min_val, max_val = agg[0], agg[1]
                if min_val < feature_statistics[df_name]["min_value"]: feature_statistics[df_name]["min_value"] = min_val 
                if max_val > feature_statistics[df_name]["max_value"]: feature_statistics[df_name]["max_value"] = max_val
    for col in feature_statistics.columns:
        n_samples = feature_statistics[col]["n_samples"]
        feature_statistics[col]["n_present"] += len(n_samples)
        feature_statistics[col]["n_missing"] += (n_files - len(n_samples))
        feature_statistics[col]["mean"] += np.mean(n_samples)
        feature_statistics[col]["median"] += np.median(n_samples)
        feature_statistics[col]["std"] += np.std(n_samples)
    feature_statistics.to_csv(savepath + "feature_statistics.csv")
    class_statistics.to_csv(savepath + "class_statistics.csv")

def plot_statistics(path="results/"):
    feature_stats = pd.read_csv(path + "feature_statistics.csv")
    class_stats = pd.read_csv(path + "class_statistics.csv")
    # Plot class stats
    plt.bar(range(1, 15), class_stats["class_counter"])
    plt.savefig(path + "class_stats")
    plt.clf()
    # Plot mean
    print(feature_stats)
    plt.bar(range(300), feature_stats.loc["mean"])
    plt.savefig(path + "feature_stats_mean_samples")
    plt.clf()
    # Plot std
    plt.bar(range(300), feature_stats["std"])
    plt.savefig(path + "feature_stats_std_samples")
    plt.clf()
    # Plot number of times feature is present
    plt.bar(range(300), feature_stats["n_present"])
    plt.savefig(path + "feature_stats_present_samples")
    plt.clf()

if __name__ == "__main__":
    path = "/work5/share/NEDO/nedo-2019/data/processed_rosbags_topickles/fixed_pickles"
    #save_single_df(path)
    #compare_timestamps()
    #get_dataset_statistics(path, "results/")
    plot_statistics()