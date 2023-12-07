import numpy as np
import glob
import os
import sys
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)

DATA_PATH = os.path.join(ROOT_DIR, 'data')
g_classes = [x.rstrip() for x in open(os.path.join(BASE_DIR, 'meta/class_names_lidar.txt'))]
g_class2label = {cls: i for i, cls in enumerate(g_classes)}


def collect_point_label(out_filename, file_format='txt'):
    """
    Convert the LiDAR dataset to a data_label file where each line is XYZRGBL.
    Data is already one
    :param out_filename: path to save the data_label file
    :param file_format: txt or numpy, determines what format to save the data in
    :return: None
    """
    points_file = os.path.join(DATA_PATH, 'classified_merged_points.pkl')
    labels_file = os.path.join(DATA_PATH, 'classified_merged_labels.pkl')

# load and ignore header
    points = pickle.load(open(points_file, 'rb'))
    labels = pickle.load(open(labels_file, 'rb'))
    # right now these are pandas dataframes. we do NOT need the index column
    points = points.values
    labels = labels.values

    # xyz_min = np.amin(points, axis=0)[0:3]
    # points[:, 0:3] -= xyz_min
    xyzrgbl = np.concatenate((points, labels), axis=1)
    # remove all points that are noise; we do not want to train on noise
    # NOTE that this is making a big assumption that all of the noise
    # has already been removed.
    xyzrgbl = xyzrgbl[xyzrgbl[:, 6] != 6]  # 6 is the label for noise

    if file_format == 'txt':
        fout = open(out_filename, 'w')
        for i in range(xyzrgbl.shape[0]):
            fout.write('%f %f %f %d %d %d %d\n' % (
                xyzrgbl[i, 0], xyzrgbl[i, 1], xyzrgbl[i, 2], xyzrgbl[i, 3], xyzrgbl[i, 4], xyzrgbl[i, 5],
                xyzrgbl[i, 6]))
        fout.close()
    elif file_format == 'numpy':
        np.save(out_filename, xyzrgbl)
    else:
        raise ValueError('Unsupported file format. Currently only support txt and numpy.')
