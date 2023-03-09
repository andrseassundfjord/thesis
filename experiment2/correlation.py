import pandas as pd 
import numpy as np
import os
import seaborn as sns
import math
import matplotlib.pyplot as plt
import csv
from functools import reduce

def check_for_similar_feature_names():
    feature_stats = pd.read_csv("../experiment1/results/feature_statistics.csv")
    similar = []
    for idx, col in enumerate(feature_stats.columns):
        for idx2, col2 in enumerate(feature_stats.columns):
            if col in col2 and idx != idx2:
                if not col in similar:
                    similar.append(col)
                if not col2 in similar:
                    similar.append(col2)
    return similar

def corr_for_one_video(idx = 0, only_similar = False):
    path = "../experiment1/resampled_pickles"
    files = os.listdir(path)
    df = pd.read_pickle(path + "/" + files[idx])
    if only_similar:
        df = df[check_for_similar_feature_names()]
    # Get correlation matrix
    corr_matrix = df.corr()
    # Plot the correlation matrix as a heatmap
    fig, ax = plt.subplots(figsize=(15, 12))
    sns.set(font_scale=1.2)
    sns.set_style({'font.family': 'serif', 'font.serif': ['Times New Roman']})
    plt.subplots_adjust(top=0.92, bottom=0.15, left=0.15, right=0.95)
    sns.heatmap(corr_matrix, 
        cmap='coolwarm', annot=True, annot_kws={'size': 12}, 
        fmt='.2f', linewidths=1, cbar=False, ax=ax, 
        xticklabels=corr_matrix.columns, yticklabels=corr_matrix.columns) 
    plt.tick_params(axis='x', rotation=45)
    plt.tick_params(axis='y', rotation=45)
    fig.savefig("results/correlation_matrix_idx{}_similar{}".format(str(idx), str(only_similar)), dpi=200, bbox_inches='tight')

def compute_correlation():
    path = "../experiment1/resampled_pickles"
    files = os.listdir(path)
    df_list = [pd.read_pickle(path + "/" + filename) for filename in files]
    # Compute the correlation matrix for each dataframe
    #corr_list = [df.corr() for df in df_list]
    #corr_matrix = reduce(lambda x, y: x.add(y, fill_value=0), corr_list) / len(corr_list)
    #corr_matrix.to_csv("results/corr_matrix_full.csv")
    concatenated_df = pd.concat(dataframes, axis=0)
    # compute the correlation between columns
    correlation_df = concatenated_df.corr()

    # remove any rows that contain missing data
    correlation_df = correlation_df.dropna()
    # Create a list of tuples, where each tuple contains the name of two features and their correlation coefficient
    corr_list = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            feature1 = corr_matrix.columns[i]
            feature2 = corr_matrix.columns[j]
            corr_coef = corr_matrix.iloc[i, j]
            if math.isnan(corr_coef):
                corr_coef = 0
            corr_list.append((feature1, feature2, corr_coef))
    corr_list_sorted = sorted(corr_list, key=lambda x: abs(x[2]), reverse=False)
    with open('results/correlation_matrix2.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Feature 1', 'Feature 2', 'Correlation Coefficient'])
        for idx, row in enumerate(corr_list_sorted):
            writer.writerow(row)

if __name__ == "__main__":
    #check_for_similar_feature_names()
    #corr_for_one_video(idx = 100, only_similar = False)
    compute_correlation()