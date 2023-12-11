import json
import os
import sys
import numpy as np
import random
import time
import torch
from shapely.geometry import shape, Point
from torch.utils.data import Dataset
from tqdm import tqdm


def filter_points_in_polygon(points_with_index, geometry):
    """
    Filter points that are within a polygon.
    Usage: points = filter_points_in_polygon(points, polygon)
    It is helpful to have already filtered the points by the bounding box of the polygon for efficiency.
    :param points_with_index: the points to filter
    :param geometry: the geometry to filter by
    :return: the filtered points
    """

    filtered = [(i, p) for i, p in points_with_index if Point(p[:2]).within(geometry)]

    # Separating indices and points for return
    indices, within_geometry = zip(*filtered) if filtered else ([], [])
    return list(within_geometry), list(indices)


def calculate_label_weights(labels, num_classes):
    class_counts = np.bincount(labels.astype(np.int32), minlength=num_classes)
    class_weights = np.zeros(num_classes)
    for i in range(num_classes):
        if class_counts[i] == 0:
            class_weights[i] = 0
        else:
            class_weights[i] = 1 / class_counts[i]
    class_weights = class_weights / np.sum(class_weights)
    class_weights = np.power(np.amax(class_weights) / class_weights, 1 / 3.0)  # What this does is it makes the weights
    # for the classes that are underrepresented larger, and the weights for the classes that are overrepresented
    # smaller. This is because we want to make sure that the loss function is not dominated by the overrepresented
    # classes.
    return class_weights
    # return np.array([1.0, 1.0])


class AerialLiDARDatasetWholeScene(Dataset):
    """
    Loader for whole scene data from las files.
    Code attribution: This code is significantly based on the PyTorch
    implementation of PointNet++ by Charles R. Qi, adapted here to work robustly with aerial .las files instead of
    geared towards the S3DIS dataset. The original code can be found here:
    https://github.com/yanx27/Pointnet_Pointnet2_pytorch/blob/master/data_utils/S3DISDataLoader.py
    """

    NUM_CLASSES = 2  # now it is 2 because of our binary classification
    OFFSET_X = 480000
    OFFSET_Y = 5455000

    def __init__(self, split='train', file_path="data/data_label", data_root="data", num_point=32768, block_size=64.0,
                 process_data=False):
        """
        Initialize the dataset with the path to the .las file
        :param file_path:
        :param num_point:
        :param block_size:
        """
        super().__init__()
        self.num_point = num_point
        self.block_size = block_size
        self.process_data = process_data
        file_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        file_path = os.path.join(file_path, 'data', 'data_label', 'classified_merged_points.npy')
        sys.path.append(file_path)
        data_all = np.load(file_path)
        self.points, self.labels = data_all[:, :6], data_all[:, 6]  # xyzrgb, label

        self.labels[self.labels != 5] = 0
        self.labels[self.labels == 5] = 1
        self.points[:, 3:6] = self.points[:, 3:6].astype(np.float32)
        self.points[:, 3:6] = self.points[:, 3:6] * (255.0 / 65535.0)
        self.points[:, 3:6] = self.points[:, 3:6].astype(np.int32)
        if self.points.shape[0] < self.num_point:
            raise ValueError('Dataset size must be larger than num_point.')

        self.save_path = os.path.join(data_root, 'buildings_split')
        buffered_geojson = os.path.join(data_root, 'buffer.geojson')
        buildings = json.load(open(buffered_geojson))['features']
        self.list_of_points = np.empty(len(buildings), dtype=object)
        self.list_of_labels = np.empty(len(buildings), dtype=object)
        if self.process_data:
            """
            In order to process the data, we need to go through each building in the geojson, calculate the points 
            that intersect with the building, and then save those points to a file
            corresponding to the building. We will also create `self.list_of_points` and `self.list_of_labels` to be able
            to access the data in the future in the __getitem__ function.
            """
            print('Processing data %s (only running in the first time)...' % self.save_path)
            for index in tqdm(range(len(buildings)), total=len(buildings)):
                building = buildings[index]
                building_uid = building['properties']['BLDG_UID']
                building_polygon = shape(building['geometry'])
                if building_polygon.geom_type == 'Polygon':
                    raise NotImplementedError('Polygon type not implemented yet; only MultiPolygon')
                elif building_polygon.geom_type == 'MultiPolygon':
                    bbox = building_polygon.bounds
                    points_array = np.array(self.points)
                    mask = (points_array[:, 0] >= bbox[0]) & (points_array[:, 0] <= bbox[2]) & \
                           (points_array[:, 1] >= bbox[1]) & (points_array[:, 1] <= bbox[3])
                    filtered_points = points_array[mask]
                    points_with_index = list(enumerate(filtered_points))
                    points_within_building, within_building = filter_points_in_polygon(points_with_index,
                                                                                       building_polygon)
                    building_file = os.path.join(self.save_path, building_uid + "_points.npy")
                    os.makedirs(os.path.dirname(building_file), exist_ok=True)
                    np.save(building_file, points_within_building)
                    building_labels = self.labels[within_building]
                    building_labels_file = os.path.join(self.save_path, building_uid + "_labels.npy")
                    np.save(building_labels_file, building_labels)
                    # save the points and labels to the list of points and labels
                    self.list_of_points[index] = points_within_building
                    self.list_of_labels[index] = building_labels
                else:
                    raise NotImplementedError('Geometry type not implemented yet')
        else:
            print('Load processed data from %s...' % self.save_path)
            for index in tqdm(range(len(buildings)), total=len(buildings)):
                building = buildings[index]
                building_uid = building['properties']['BLDG_UID']
                building_file = os.path.join(self.save_path, building_uid + "_points.npy")
                building_labels_file = os.path.join(self.save_path, building_uid + "_labels.npy")
                self.list_of_points[index] = np.load(building_file)
                self.list_of_labels[index] = np.load(building_labels_file)

        # train/test split
        SPLIT_RATIO = 0.8
        # create random booleans the length of the number of buildings, such that SPLIT_RATIO of them are True
        train_mask = np.random.rand(len(buildings)) < SPLIT_RATIO
        test_mask = np.logical_not(train_mask)
        if split == 'train':
            self.list_of_points = self.list_of_points[train_mask]
            self.list_of_labels = self.list_of_labels[train_mask]
        elif split == 'test':
            self.list_of_points = self.list_of_points[test_mask]
            self.list_of_labels = self.list_of_labels[test_mask]

        # calculate label weights
        labelweights = np.zeros(self.NUM_CLASSES)

        num_point_all = []

        for index in tqdm(range(len(self.list_of_labels)), total=len(self.list_of_labels)):
            points, labels = self.list_of_points[index], self.list_of_labels[index]
            tmp, _ = np.histogram(labels, range(self.NUM_CLASSES + 1))
            labelweights += tmp
            num_point_all.append(labels.shape[0])
        labelweights = labelweights.astype(np.float32)
        labelweights = labelweights / np.sum(labelweights)
        self.labelweights = np.power(np.amax(labelweights) / labelweights, 1 / 3.0)
        print(self.labelweights)

    def __len__(self):
        return len(self.list_of_points)

    def __getitem__(self, idx):
        points, labels = self.list_of_points[idx], self.list_of_labels[idx]
        N_points = points.shape[0]

        while True:
            center = points[np.random.choice(N_points)][:3]
            block_min = center - [self.block_size / 2.0, self.block_size / 2.0, 0]
            block_max = center + [self.block_size / 2.0, self.block_size / 2.0, 0]
            point_idxs = np.where(
                (points[:, 0] >= block_min[0]) & (points[:, 0] <= block_max[0]) & (points[:, 1] >= block_min[1]) & (
                        points[:, 1] <= block_max[1]))[0]
            if (point_idxs.size > self.num_point // 4) or (point_idxs.size == N_points):
                break

        if point_idxs.size >= self.num_point:
            selected_point_idxs = np.random.choice(point_idxs, self.num_point, replace=False)
        else:
            selected_point_idxs = np.random.choice(point_idxs, self.num_point, replace=True)

        # code isn't really working, let's add some asserts to see what's going on
        assert selected_point_idxs.size == self.num_point

        selected_points = points[selected_point_idxs, :]  # num_point * 6
        # offset the x and y values
        selected_points[:, 0] = selected_points[:, 0] - self.OFFSET_X
        selected_points[:, 1] = selected_points[:, 1] - self.OFFSET_Y
        # offset the center
        center[0] = center[0] - self.OFFSET_X
        center[1] = center[1] - self.OFFSET_Y
        current_points = np.zeros((self.num_point, 9))  # num_point * 9
        coord_max = np.max(selected_points[:, :3], axis=0)
        current_points[:, 6] = selected_points[:, 0] / coord_max[0]  # x
        current_points[:, 7] = selected_points[:, 1] / coord_max[1]  # y
        current_points[:, 8] = selected_points[:, 2] / coord_max[2]  # z
        selected_points[:, 0] = selected_points[:, 0] - center[0]  # x
        selected_points[:, 1] = selected_points[:, 1] - center[1]  # y
        selected_points[:, 3:6] /= 255.0
        current_points[:, 0:6] = selected_points
        current_labels = labels[selected_point_idxs]  # num_point * 1

        assert current_points.shape == (self.num_point, 9)
        assert current_labels.shape == (self.num_point,)

        return current_points, current_labels


manual_seed = 123


def worker_init_fn(worker_id):
    random.seed(manual_seed + worker_id)


if __name__ == '__main__':
    # file_path = 'data/data_label/classified_merged_points.npy'
    file_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_path = os.path.join(file_path, 'data', 'data_label', 'classified_merged_points.npy')
    sys.path.append(file_path)
    # Initialize the dataset with the path to the file
    aerial_data = AerialLiDARDatasetWholeScene(file_path=file_path)

    print('Aerial LiDAR data size:', aerial_data.__len__())
    print('Aerial LiDAR data 0 shape:', aerial_data.__getitem__(0)[0].shape)
    print('Aerial LiDAR label 0 shape:', aerial_data.__getitem__(0)[1].shape)

    random.seed(manual_seed)
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)

    # Create the DataLoader for the aerial LiDAR data
    train_loader = torch.utils.data.DataLoader(aerial_data, batch_size=16, shuffle=True, num_workers=8, pin_memory=True,
                                               worker_init_fn=worker_init_fn)

    # Iterate through the DataLoader
    for idx in range(4):
        end = time.time()
        for i, (input, target) in enumerate(train_loader):
            print('time: {}/{}--{}'.format(i + 1, len(train_loader), time.time() - end))
            end = time.time()
