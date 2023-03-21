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

def compute_correlation(test = True):
    path = "../experiment1/resampled_pickles"
    files = os.listdir(path)
    df_list = [pd.read_pickle(path + "/" + filename) for filename in files]
    all_features = set().union(*[set(df.columns) for df in df_list])
    # Add missing columns
    for idx, df in enumerate(df_list):
        missing_features = all_features - set(df.columns)
        for feature in missing_features:
            df_list[idx][feature] = pd.Series(dtype='float64')
            df_list[idx] = df_list[idx].copy()
        print(df['/driver/blink/data'].isnull().sum())
        return
    # Sort by column name
    df_list_sorted = [df.sort_index(axis = 1) for df in df_list]
    # Get correlation matrices
    corr_m_list = [df.corr() for df in df_list_sorted]
    feature_counts = reduce(lambda x, y: x.add(y.notna().astype(int), fill_value=0), df_list_sorted)
    # Compute the sum of correlations for each feature across all dataframes
    corr_sum = reduce(lambda x, y: x.add(y, fill_value=0), corr_m_list)
    # Divide each correlation by the number of occurrences of the corresponding feature
    corr_matrix = corr_sum.div(feature_counts, fill_value=0)
    corr_matrix.to_csv("results/corr_matrix_full.csv")
    # Create a list of tuples, where each tuple contains the name of two features and their correlation coefficient
    corr_list = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if i != j: 
                feature1 = corr_matrix.columns[i]
                feature2 = corr_matrix.columns[j]
                corr_coef = corr_matrix.iloc[i, j]
                if math.isnan(corr_coef):
                    corr_coef = 0
                corr_list.append((feature1, feature2, corr_coef))
    corr_list_sorted = sorted(corr_list, key=lambda x: abs(x[2]), reverse=False)
    with open('results/correlation_list.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Feature 1', 'Feature 2', 'Correlation Coefficient'])
        for idx, row in enumerate(corr_list_sorted):
            writer.writerow(row)

if __name__ == "__main__":
    #check_for_similar_feature_names()
    #corr_for_one_video(idx = 100, only_similar = False)
    compute_correlation(test = False)