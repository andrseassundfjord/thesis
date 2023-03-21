import os
import glob
import random
import pickle
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, ListDataset

def load_data(folder_path):
    class_dict = {}
    for file_path in glob.glob(os.path.join(folder_path, '*.pkl')):
        with open(file_path, 'rb') as f:
            sample = pickle.load(f)
            data.append(sample)
            print(sample)
            print(type(sample))
        return
        label = file_path.split(".")[0].split("_")[-1]
        if label not in class_dict:
            class_dict[label] = []
        class_dict[label].append(sample)
    return class_dict

def split_data(class_dict, train_ratio=0.7):
    train_data = []
    train_labels = []
    test_data = []
    test_labels = []
    
    for label in class_dict:
        samples = class_dict[label]
        n_train = int(len(samples) * train_ratio)
        random.shuffle(samples)
        train_data += samples[:n_train]
        train_labels += [label] * n_train
        test_data += samples[n_train:]
        test_labels += [label] * (len(samples) - n_train)
        
    # Define the data loaders
    train_dataset = ListDataset(train_data, train_labels)
    test_dataset = ListDataset(test_data, test_labels)
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    return train_loader, test_loader

def get_dataloaders():
    class_dict = load_data('../experiment1/resampled_pickles')
    return split_data(class_dict)


if __name__ == "__main__":
    print(torch.__version__)
    class_dict = load_data('../experiment1/resampled_pickles')
    #train_dataloader, test_dataloader = split_data(class_dict)




