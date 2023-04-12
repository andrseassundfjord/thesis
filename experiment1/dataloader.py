import sys
import numpy as np
import pandas as pd
import torch
import os
import pickle
import math
import matplotlib.pyplot as plt
import seaborn as sns
import random

def plot_compare_timestamps(n_samples = 200, original = "results/original_rate.csv", sampled = "results/resampled_rate.csv"):
    baseline = pd.read_csv(original)
    test = pd.read_csv(sampled)
    baseline = baseline.set_index(baseline["Unnamed: 0"])
    test = test.set_index(test["Unnamed: 0"])
    df = baseline["/driver/eye_opening_rate/data"].to_frame()
    df = df.join(test["/driver/eye_opening_rate/data"].to_frame().add_suffix("_sampled"), how = "outer")
    df = df.reset_index()
    plt.plot(df.index, df["/driver/eye_opening_rate/data"], color="red", marker="o")
    plt.plot(df.index, df["/driver/eye_opening_rate/data_sampled"], ls="--", color="blue", marker="x")
    plt.savefig("results/compare_timestamps_{}_samples_interpolate_bfill".format(str(n_samples)))

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

def resample_dataframe(df, n_samples = 100, interpolate = False, normalize = False):
    # Replace True/False with 1/0
    df = df.replace(True, 1)
    df = df.replace(False, 0)
    # resample to n_samples samples
    min, max = df.index[0], df.index[-1]
    interval = (max - min) / (n_samples - 1)
    if interpolate:
        resampled_df = df.resample(interval).mean(numeric_only = True).interpolate().bfill()
    else:
        resampled_df = df.resample(interval).mean(numeric_only = True).ffill().bfill()
    # Some have n_samples + 1 for some reason, remove last if thats the case 
    resampled_df = resampled_df.iloc[:n_samples]

    #if normalize:

    return resampled_df

def save_single_df(path, n_samples = 200, test=False, interpolate = False, normalize = False):
        """
        Reformat pickled series to single dataframe, with specified number of samples.
        """
        files = os.listdir(path)
        for filename in files:
            # Read pickle
            series = pd.read_pickle(path + "/" + filename)
            df = series_to_dataframe(series)
            # Weird empty file
            if len(df.index) == 0:
                print(filename)
            else:
                resampled_df = resample_dataframe(df, n_samples = n_samples, interpolate = interpolate, normalize = normalize)
                # Test with only one file
                if test:
                    df.to_csv("results/original_rate.csv")
                    resampled_df.to_csv("results/resampled_rate.csv")
                    plot_compare_timestamps(n_samples = n_samples, original = "results/original_rate.csv", sampled = "results/resampled_rate.csv")
                    return
                resampled_df.to_pickle("interpolated_pickles/{}.pkl".format(filename))

def get_n_samples():
    """
    Function for counting number of samples for each feature in each video. 
    Also for calculating mean, median and std of this.
    """
    path = "/work5/share/NEDO/nedo-2019/data/processed_rosbags_topickles/fixed_pickles"
    files = os.listdir(path)
    n_files = len(files)
    feature_statistics = pd.DataFrame(index = ["n_present", "mean", "median", "std"])
    n_samples = pd.DataFrame(index = np.arange(n_files))
    for filename in files:
        series = pd.read_pickle(path + "/" + filename)
        for df_name in series.index:
            if df_name not in feature_statistics.columns:
                feature_statistics[df_name] = [0, 0, 0, 0]
                feature_statistics = feature_statistics.copy()
            if df_name not in n_samples.columns:
                n_samples[df_name] = np.zeros(n_files)
                n_samples = n_samples.copy()
            feature_statistics[df_name]["n_present"] += 1
            pos = n_samples[df_name].idxmin()
            n_samples[df_name][pos] = len(series[df_name])
    for col in feature_statistics.columns:
        n = n_samples[col]
        n = n[n != 0]
        feature_statistics[col]["mean"] += np.mean(n)
        feature_statistics[col]["median"] += np.median(n)
        feature_statistics[col]["std"] += np.std(n)
    feature_statistics.transpose().to_csv("results/feature_sample_size_statistics.csv")
    n_samples.to_csv("results/n_samples_stats.csv")
    plot_statistics()

def get_dataset_statistics():
    path = "resampled_pickles"
    files = os.listdir(path)
    n_files = len(files)
    feature_statistics = pd.DataFrame(index = [
        "min_value", "max_value", "type", "isnumber"])
    class_statistics = pd.DataFrame(index = range(1, 15))
    class_statistics["class_counter"] = np.zeros(14)
    for filename in files:
        series = pd.read_pickle(path + "/" + filename)
        label = filename.split(".")[0].split("_")[-1]
        class_statistics["class_counter"][int(label)] += 1
        for df_name in series.columns:
            if df_name not in feature_statistics.columns:
                val_type = str(type(series[df_name].to_numpy().flatten()[0]))
                isnumber = "float" in val_type or "int" in val_type
                # Check shape of series[df_name]
                # If bigger than one
                # add each seperately
                # how to add numbers further down?
                # one solution is:
                # index_list = list(series.index)
                # for df_name in index_list
                # index_list.append(name) when multiple cols in series[df_name]
                # continue so it doesnt fuck everything else
                feature_statistics[df_name] = [math.inf, 0, val_type, isnumber]
                feature_statistics = feature_statistics.copy()
            if feature_statistics[df_name]["isnumber"]:
                agg = series[df_name].agg(["min", "max"]).to_numpy().flatten()
                min_val, max_val = agg[0], agg[1]
                if min_val < feature_statistics[df_name]["min_value"]: feature_statistics[df_name]["min_value"] = min_val 
                if max_val > feature_statistics[df_name]["max_value"]: feature_statistics[df_name]["max_value"] = max_val
            else: 
                print(df_name)
    feature_statistics.to_csv("results/feature_statistics.csv")
    class_statistics.to_csv("results/class_statistics.csv")

def plot_statistics(path="results/"):
    feature_stats = pd.read_csv(path + "feature_sample_size_statistics.csv").set_index("Unnamed: 0")
    #feature_stats = feature_stats.drop(columns = ["Unnamed: 0"])
    class_stats = pd.read_csv(path + "class_statistics.csv")
    # Plot class stats
    plt.bar(range(1, 15), class_stats["class_counter"])
    plt.xticks(np.arange(1, 15, 1))
    plt.xlabel('Class')
    plt.ylabel('Occurences in NEDO-2019')
    plt.savefig(path + "class_stats")
    plt.clf()
    # Plot mean
    values = np.asarray(feature_stats["mean"], dtype="float")
    plt.hist(values, bins = int(len(values)/2))
    plt.xlabel('Mean number of Samples')
    plt.ylabel('Number of Features')
    plt.savefig(path + "feature_stats_mean_samples")
    plt.clf()
    # Plot median
    values = np.asarray(feature_stats["median"], dtype="float")
    plt.hist(values, bins = int(len(values)/2))
    plt.xlabel('Median number of Samples')
    plt.ylabel('Number of Features')
    plt.savefig(path + "feature_stats_median_samples")
    plt.clf()
    # Plot std
    values = np.asarray(feature_stats["std"], dtype="float")
    plt.hist(values, bins = int(len(values)/2))
    plt.xlabel('Standard deviation of number of Samples')
    plt.ylabel('Number of Features')
    plt.savefig(path + "feature_stats_std_samples")
    plt.clf()
    # Plot number of times feature is present
    values = np.asarray(feature_stats["n_present"], dtype="float")
    plt.hist(values, bins = int(len(values)/2))
    plt.xlabel('Number of Samples present in')
    plt.ylabel('Number of Features')
    plt.savefig(path + "feature_stats_present_samples")
    plt.clf()
    # Plot distribution of sample sizes
    n_samples = pd.read_csv("results/n_samples_stats.csv")
    n_samples = n_samples.set_index(n_samples["Unnamed: 0"])
    n_samples = n_samples.drop(columns = ["Unnamed: 0"])
    fig = sns.heatmap(n_samples).figure
    fig.savefig("results/distribution")

def compute_sample_correlation(n_files = 1, interpolate = False):
    path = "/work5/share/NEDO/nedo-2019/data/processed_rosbags_topickles/fixed_pickles"
    files = os.listdir(path)
    series_list = [pd.read_pickle(path + "/" + filename) for filename in random.sample(files, n_files)]
    # Load the original timeseries dataframe
    df_list = [series_to_dataframe(series) for series in series_list]
    # Define a list of n values to resample the dataframe
    n_values = [50, 75, 100, 125, 150, 175, 200]
    # Get list of all features 
    all_features = set().union(*[set(df.columns) for df in df_list])
    # Create an empty DataFrame to store the correlation coefficients
    corr_df = pd.DataFrame(columns=['n', "mean_size_diff", "max_size_diff", "std_size_diff", "MSE", 'mean_corr', "std_corr", "min", "n_under_0.9", "n_under_0.8", "n_under_0.7", "n_under_0.5"], index = n_values)
    for n in n_values:
        corr_df["max_size_diff"][n] = 0
        corr_df["mean_size_diff"][n] = 0
        corr_df["std_size_diff"][n] = 0
        corr_df["min"][n] = 1
        corr_df["MSE"][n] = 0
        corr_df["mean_corr"][n] = 0
        corr_df["std_corr"][n] = 0
        corr_df["n_under_0.9"][n] = 0
        corr_df["n_under_0.8"][n] = 0
        corr_df["n_under_0.7"][n] = 0
        corr_df["n_under_0.5"][n] = 0
    # Loop over the n values and resample the dataframe for each n
    for df in df_list:
        for n in n_values:
            resampled_df = resample_dataframe(df, n_samples = n, interpolate = interpolate)
            # Compute the correlation coefficients between the original and resampled data for each feature
            corr_coeffs = []
            diffs = []
            mse = 0
            for feature in df.columns:
                diff = 0
                orig_data = df[feature].dropna()
                # Skip object features
                if orig_data.dtypes == "object":
                    continue
                resampled_data = resampled_df[feature]
                corr = 1
                if orig_data.shape[0] > resampled_data[0]:
                    diff = orig_data.shape[0] > resampled_data[0]
                    merged_data = pd.merge_asof(orig_data.reset_index(), resampled_data.reset_index(), on='index', direction='nearest', suffixes=('_orig', '_resampled'))
                    corr = np.corrcoef(merged_data[feature + '_orig'], merged_data[feature + '_resampled'])[0, 1]
                    mse += ((merged_data[feature + '_orig'] - merged_data[feature + '_resampled'])**2).mean()
                for treshold in [0.9, 0.8, 0.7, 0.5]:
                    if corr < treshold:
                        corr_df["n_under_{}".format(str(treshold))][n] += 1
                corr_coeffs.append(corr)
                diffs.append(diff)
                if diff > corr_df["max_size_diff"][n]:
                    corr_df["max_size_diff"][n] = diff
                if abs(corr) < corr_df["min"][n]:
                    corr_df["min"][n] = abs(corr)
            # Compute MSE
            corr_df["MSE"][n] += mse/len(df.columns)
            # Compute mean and std diff
            mean_diff = np.mean(diffs)
            std_diff = np.std(diffs)
            # Compute the mean correlation coefficient across all features
            mean_corr = np.nanmean(corr_coeffs)
            # Compute std
            std_corr = np.nanstd(corr_coeffs)
            corr_df["n"][n] = n
            corr_df["std_size_diff"][n] = std_diff
            corr_df["mean_size_diff"][n] = mean_diff
            if not math.isnan(mean_corr):
                corr_df["mean_corr"][n] += mean_corr
            if not math.isnan(std_corr):
                corr_df["std_corr"][n] += std_corr
    # Save the correlation DataFrame to a CSV file 
    for n in n_values:
        corr_df["mean_corr"][n] /= n_files
        corr_df["std_corr"][n] /= n_files
    corr_df.to_csv('results/correlation_results_nfiles{}_interpolate{}.csv'.format(str(n_files), str(interpolate)), index=False)

def get_feature_names():
    all_features = list(pd.read_csv("results/feature_statistics.csv").columns)
    mobileye = ["/mobileye/{}/data".format(str(i)) for i in range(20, 1500)]
    unique = []
    for feature in all_features:
        if feature not in mobileye:
            unique.append(feature)
    print(unique)
    print(len(unique))

if __name__ == "__main__":
    path = "/work5/share/NEDO/nedo-2019/data/processed_rosbags_topickles/fixed_pickles"
    #plot_statistics()
    #get_dataset_statistics()
    #get_n_samples()
    #compute_sample_correlation(n_files = 500)
    save_single_df(path, n_samples = 200, test=False, interpolate = True, normalize = True)
    #get_feature_names()