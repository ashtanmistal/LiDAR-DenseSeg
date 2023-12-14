import os
import os.path as osp
import numpy as np
import pandas as pd
import torch.utils.data as data


def convert_to_pkl(las, output_path):
    """
    Convert the given LiDAR data to a .pkl file.
    :param las: the input LiDAR dataset
    :param output_path: the path to save the .pkl file
    """
    points = {
        "X": las.x,
        "Y": las.y,
        "Z": las.z,
        "r": las.red,
        "g": las.green,
        "b": las.blue,
    }

    labels = {
        "classification": las.classification
    }
    if not osp.exists(osp.dirname(output_path)):
        os.makedirs(osp.dirname(output_path))

    pd.DataFrame(points).to_pickle(output_path.replace(".pkl", "_points.pkl"))
    pd.DataFrame(labels).to_pickle(output_path.replace(".pkl", "_labels.pkl"))


def merge_pkl_files(directory_path, output_pkl_path):
    """
    Merges all PKL files in a directory into a single PKL file.

    Parameters:
    directory_path (str): Path to the directory containing the .pkl files.
    output_pkl_path (str): Path to the output .pkl file.
    """
    # List to hold dataframes
    points = []
    labels = []

    # Iterate through all files in the directory
    for filename in os.listdir(directory_path):
        if filename.endswith('_points.pkl'):
            file_path = os.path.join(directory_path, filename)
            # Read the PKL file and append to the list
            df = pd.read_pickle(file_path)
            points.append(df)
        elif filename.endswith('_labels.pkl'):
            file_path = os.path.join(directory_path, filename)
            # Read the PKL file and append to the list
            df = pd.read_pickle(file_path)
            labels.append(df)

    # Concatenate all dataframes
    merged_df = pd.concat(points, ignore_index=True)
    merged_df_labels = pd.concat(labels, ignore_index=True)

    # Save the merged dataframe to a PKL file
    merged_df.to_pickle(output_pkl_path.replace(".pkl", "_points.pkl"))
    merged_df_labels.to_pickle(output_pkl_path.replace(".pkl", "_labels.pkl"))


def train_test_split(config, points, labels):
    """
    Split the data into training and testing sets based on config.ratio_tr_data.
    :param config: ml_collections.ConfigDict() object containing the configuration parameters
    :param points: pickle file containing the points
    :param labels: pickle file containing the labels
    :return: (train_points, train_labels, test_points, test_labels) corresponding to the split data
    """

    # shuffle all the data
    indices = np.random.permutation(len(points))
    points = points[indices]
    labels = labels[indices]

    split_index = int(config.ratio_tr_data * len(points))
    train_points = data.Subset(points, indices[:split_index])
    train_labels = data.Subset(labels, indices[:split_index])
    test_points = data.Subset(points, indices[split_index:])
    test_labels = data.Subset(points, indices[split_index:])

    return train_points, train_labels, test_points, test_labels


