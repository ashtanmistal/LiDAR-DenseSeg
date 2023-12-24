"""
Author: Ashtan Mistal (derived from Charles R. Qi's implementation of PointNet++ in PyTorch)
Date: Dec 2023
"""
import json
import os
import numpy as np
import pylas
import torch
from shapely.geometry import shape, Point
from torch.utils.data import Dataset
from tqdm import tqdm


def read_las_file(file_path):
    """
    Reads a .las file and extracts the necessary point cloud data.

    :param file_path: Path to the .las file to be read.
    :return: A tuple of numpy arrays (points, colors, labels).
    """

    # Open the .las file
    las_data = pylas.read(file_path)

    # Extract xyz coordinates
    points = np.vstack((las_data.x, las_data.y, las_data.z)).transpose()

    # Extract classification labels
    labels = las_data.classification

    # Check if color information is present and extract it
    if hasattr(las_data, 'red') and hasattr(las_data, 'green') and hasattr(las_data, 'blue'):
        colors = np.vstack((las_data.red, las_data.green, las_data.blue)).transpose()
    else:
        colors = np.zeros_like(points)  # If no color info, create a dummy array with zeros

    # Normalize or preprocess the points if needed
    # This would be based on the preprocessing done in the original S3DISDataLoader

    # Return the extracted data
    return points, colors, labels


def write_ply(filename, data, create=True):
    if create:
        # if the directory doesn't exist, create it
        os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write("element vertex {}\n".format(len(data)))
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for row in data:
            f.write("{} {} {} {} {} {}\n".format(row[0], row[1], row[2], int(row[3]), int(row[4]), int(row[5])))


def debug_to_ply(data_path, xyz, rgb):
    filename = os.path.join(data_path, 'debug.ply')
    data = np.concatenate((xyz, rgb), axis=1)
    write_ply(filename, data)


def read_ply(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    # Extracting the number of vertices from the header
    num_vertices = int(next(line for line in lines if line.startswith("element vertex")).split()[-1])

    # Find the start of the vertex data
    start_idx = lines.index("end_header\n") + 1

    # Read and convert the vertex data
    data = np.array([list(map(float, line.split())) for line in tqdm(lines[start_idx:start_idx + num_vertices])])

    return data


def filter_points_in_multipolygon(points_with_index, geometry):
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


class UBCDataset(Dataset):
    """
    Loader for whole scene data from las files.
    Code attribution: This code is significantly based on the PyTorch
    implementation of PointNet++ by Charles R. Qi, adapted here to work robustly with aerial .las files instead of
    geared towards the S3DIS dataset. The original code can be found here:
    https://github.com/yanx27/Pointnet_Pointnet2_pytorch/blob/master/data_utils/S3DISDataLoader.py
    """

    DEBUG_MODE = False  # set to True to debug
    NUM_CLASSES = 2  # now it is 2 because of our binary classification
    CLASS_OF_INTEREST = 6
    OFFSET_X = 480000
    OFFSET_Y = 5455000
    ROTATION_DEGREES = 28.000  # This is the rotation of UBC's roads relative to true north.
    ROTATION_RADIANS = np.radians(ROTATION_DEGREES)
    INVERSE_ROTATION_MATRIX = np.array([[np.cos(ROTATION_RADIANS), np.sin(ROTATION_RADIANS), 0],
                                        [-np.sin(ROTATION_RADIANS), np.cos(ROTATION_RADIANS), 0],
                                        [0, 0, 1]])

    def __init__(self, split='train', data_path="data/las", data_root="data", num_point=32768, block_size=32.0,
                 process_data=False):
        """
        Initialize the dataset with the path to the .las file
        :param split: 'train' or 'test'
        :param data_path: path to the .las folder
        :param data_root: path to the root of the data
        :param num_point: number of points to sample
        :param block_size: size of the block to sample
        :param process_data: whether to process the data or not
        """
        super().__init__()
        self.num_point = num_point
        self.block_size = block_size
        self.process_data = process_data

        self.save_path = os.path.join(data_root, 'buildings_split')
        buffered_geojson = os.path.join(data_root, 'buffer.geojson')
        # buffered_geojson = os.path.join(data_root, 'singlebuilding.geojson')  # overfitting to a single building
        buildings = json.load(open(buffered_geojson))['features']
        # remove any buildings with a height less than 2
        buildings = [building for building in buildings if (building['properties']['BLDG_HEIGHT'] is not None) and
                     (building['properties']['BLDG_HEIGHT'] >= 3)]
        # take only the first 10 buildings for debugging
        # buildings = buildings[:1]
        self.list_of_points = np.empty(len(buildings), dtype=object)
        self.list_of_labels = np.empty(len(buildings), dtype=object)
        self.list_of_building_uids = []
        if self.process_data:
            """
            In order to process the data, we need to go through each building in the geojson, calculate the points 
            that intersect with the building, and then save those points to a file
            corresponding to the building. We will also create `self.list_of_points` and `self.list_of_labels` to be
            able to access the data in the future in the __getitem__ function.
            """
            print('Processing data %s (only running in the first time)...' % self.save_path)
            points_data = []
            colors_data = []
            label_data = []
            print("las processing...")
            for filename in tqdm(os.listdir(data_path)):
                if filename.endswith(".las"):
                    xyz, rgb, l = read_las_file(os.path.join(data_path, filename))
                    # remove never classified, unclassified, and noise (0, 1, 7)
                    valid_indices = np.where((l != 0) & (l != 1) & (l != 7))
                    xyz, rgb, l = xyz[valid_indices], rgb[valid_indices], l[valid_indices]
                    rgb = rgb * (255.0 / 65535.0)
                    l = (l == self.CLASS_OF_INTEREST).astype(int)  # make the labels binary
                    points_data.append(xyz)
                    colors_data.append(rgb)
                    label_data.append(l)

            # flatten the arrays to make them 2D
            ply_data = np.concatenate(points_data, axis=0)
            ply_data = np.concatenate((ply_data, np.concatenate(colors_data, axis=0)), axis=1)
            label_data = np.concatenate(label_data, axis=0)
            print("building processing...")
            for index in tqdm(range(len(buildings)), total=len(buildings)):
                building = buildings[index]
                building_uid = building['properties']['BLDG_UID']
                building_polygon = shape(building['geometry'])
                if building_polygon.geom_type == 'MultiPolygon':
                    bbox = building_polygon.bounds
                    mask = (ply_data[:, 0] >= bbox[0]) & (ply_data[:, 0] <= bbox[2]) & \
                           (ply_data[:, 1] >= bbox[1]) & (ply_data[:, 1] <= bbox[3])
                    masked_labels = label_data[mask]
                    points_with_index = list(enumerate(ply_data[mask]))
                    points_in_bldg, idx_within_building = filter_points_in_multipolygon(points_with_index,
                                                                                        building_polygon)
                    # subtract offsets from x and y
                    points_in_bldg = np.array(points_in_bldg)
                    tmp = np.matmul(self.INVERSE_ROTATION_MATRIX,
                                    np.array([points_in_bldg[:, 0] - self.OFFSET_X,
                                              points_in_bldg[:, 1] - self.OFFSET_Y,
                                              points_in_bldg[:, 2]]))
                    points_in_bldg[:, 0], points_in_bldg[:, 1], points_in_bldg[:, 2] = tmp[0], tmp[1], tmp[2]
                    # building_file = os.path.join(self.save_path, building_uid + "_points.ply")
                    # os.makedirs(os.path.dirname(building_file), exist_ok=True)
                    # write_ply(building_file, points_in_bldg)  # more useful than .npy for visualizing
                    building_file = os.path.join(self.save_path, building_uid + "_points.npy")
                    np.save(building_file, points_in_bldg)
                    building_labels = masked_labels[idx_within_building]
                    building_labels_file = os.path.join(self.save_path, building_uid + "_labels.txt")
                    os.makedirs(os.path.dirname(building_labels_file), exist_ok=True)
                    np.savetxt(building_labels_file, building_labels, fmt="%d")
                    # save the points and labels to the list of points and labels
                    self.list_of_points[index] = points_in_bldg
                    self.list_of_labels[index] = building_labels
                    self.list_of_building_uids.append(building_uid)
                else:
                    raise NotImplementedError('Geometry type not implemented yet')
        else:
            print('Load processed data from %s...' % self.save_path)
            for index in tqdm(range(len(buildings)), total=len(buildings)):
                building = buildings[index]
                building_uid = building['properties']['BLDG_UID']
                # building_file = os.path.join(self.save_path, building_uid + "_points.ply")
                # self.list_of_points[index] = read_ply(building_file)
                # using .npy instead
                building_file = os.path.join(self.save_path, building_uid + "_points.npy")
                self.list_of_points[index] = np.load(building_file)
                building_labels_file = os.path.join(self.save_path, building_uid + "_labels.txt")
                self.list_of_labels[index] = np.loadtxt(building_labels_file)

        # train/test split
        if len(buildings) == 1:
            # This is done when we are overfitting to a single building
            split_ratio = 1.0
            train_mask = np.random.rand(len(buildings)) < split_ratio
            test_mask = train_mask
        else:
            # real train/test split
            split_ratio = 0.7
            train_mask = np.random.rand(len(buildings)) < split_ratio
            # train_mask = torch.rand(len(buildings)) < split_ratio
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
        self.labelweights = self.labelweights / np.sum(self.labelweights)
        print(self.labelweights)

    def __len__(self):
        return len(self.list_of_points)

    def __getitem__(self, idx):
        # idx = 0  # using this for debugging
        points, labels = self.list_of_points[idx], self.list_of_labels[idx]
        labels = labels.astype(int)
        n_points = points.shape[0]

        while True:
            center = points[np.random.choice(n_points)][:3]
            # just pick the center of the building for single batch
            # center = np.mean(points[:, :3], axis=0)
            block_min = center - [self.block_size / 2.0, self.block_size / 2.0, 0]
            block_max = center + [self.block_size / 2.0, self.block_size / 2.0, 0]
            point_idxs = np.where(
                (points[:, 0] >= block_min[0]) & (points[:, 0] <= block_max[0]) & (points[:, 1] >= block_min[1]) & (
                        points[:, 1] <= block_max[1]))[0]
            if point_idxs.size > self.num_point // 2:
                break

        if point_idxs.size >= self.num_point:
            selected_point_idxs = np.random.choice(point_idxs, self.num_point, replace=False)
            # selected_point_idxs = torch.randperm(point_idxs.size)[:self.num_point]

        else:
            selected_point_idxs = np.random.choice(point_idxs, self.num_point, replace=True)
            # selected_point_idxs = torch.randint(0, point_idxs.size, (self.num_point,))

        selected_points = points[selected_point_idxs, :]  # num_point * 6
        min_z = np.min(selected_points[:, 2])

        current_points = np.zeros((self.num_point, 9))  # num_point * 9

        selected_points[:, 0] = selected_points[:, 0] - center[0]  # x
        selected_points[:, 1] = selected_points[:, 1] - center[1]  # y
        selected_points[:, 2] = selected_points[:, 2] - min_z  # z
        coord_max = np.max(selected_points[:, :3], axis=0)

        current_points[:, 6] = selected_points[:, 0] / coord_max[0]  # x
        current_points[:, 7] = selected_points[:, 1] / coord_max[1]  # y
        current_points[:, 8] = selected_points[:, 2] / coord_max[2]  # z
        # ensure they are between -1 and 1
        current_points[:, 6] = current_points[:, 6] * 2 - 1
        current_points[:, 7] = current_points[:, 7] * 2 - 1
        current_points[:, 8] = current_points[:, 8] * 2 - 1
        current_points[:, 0:6] = selected_points
        current_labels = labels[selected_point_idxs]  # num_point * 1

        # if self.DEBUG_MODE:
        #     assert current_points.shape == (self.num_point, 9)
        #     assert current_labels.shape == (self.num_point,)
        #
        #     # save to test.ply for debugging (keeping xyzrgb. write a header)
        #     test_ply = os.path.join(self.save_path, 'test.ply')
        #     write_ply(test_ply, current_points)
        #
        #     # write to a .ply all the points with 1 as a label
        #     buildings_ply = os.path.join(self.save_path, 'buildings.ply')
        #     building_idx = np.where(current_labels == 1)
        #     building_points = current_points[building_idx]
        #     write_ply(buildings_ply, building_points)

        # current_labels = np.ones(self.num_point)

        return current_points, current_labels


class UBCDatasetWholeScene():
    """
    Loader for whole scene data from las files.
    """

    DEBUG_MODE = False  # set to True to debug
    NUM_CLASSES = 2  # now it is 2 because of our binary classification
    CLASS_OF_INTEREST = 6
    OFFSET_X = 480000
    OFFSET_Y = 5455000
    ROTATION_DEGREES = 28.000  # This is the rotation of UBC's roads relative to true north.
    ROTATION_RADIANS = np.radians(ROTATION_DEGREES)
    INVERSE_ROTATION_MATRIX = np.array([[np.cos(ROTATION_RADIANS), np.sin(ROTATION_RADIANS), 0],
                                        [-np.sin(ROTATION_RADIANS), np.cos(ROTATION_RADIANS), 0],
                                        [0, 0, 1]])
    def __init__(self, root, block_points=8192, split='unclassified', stride=16.0, block_size=32.0, padding=0.001,
                 preprocess=False):
        self.block_points = block_points
        self.block_size = block_size
        self.padding = padding
        self.root = root
        self.split = split
        self.stride = stride
        self.preprocess = preprocess
        self.scene_points_num = []
        assert split in ['classified', 'unclassified', 'merged']
        self.save_path = os.path.join(self.root, 'buildings_split')
        self.las_path = os.path.join(self.root, 'las')
        buffered_geojson = os.path.join(self.root, 'buffer.geojson')
        buildings = json.load(open(buffered_geojson))['features']
        # remove any buildings with a height less than 2
        buildings = [building for building in buildings if (building['properties']['BLDG_HEIGHT'] is not None) and
                     (building['properties']['BLDG_HEIGHT'] >= 3)]
        self.list_of_building_uids = []
        if self.preprocess:
            """we follow a similar preprocessing step as above, but we need the *unclassified* data to be stored in 
            the same folder as the *classified* data but with _unlabeled.npy instead of _points.npy"""
            print('Preprocessing data %s (only running in the first time)...' % self.save_path)
            points_data = []
            colors_data = []
            label_data = []
            print("las processing...")
            for filename in tqdm(os.listdir(self.las_path)):
                if filename.endswith(".las"):
                    xyz, rgb, l = read_las_file(os.path.join(self.las_path, filename))
                    # only keeping unclassified and never classified points
                    if self.split == 'classified':
                        valid_indices = np.where((l != 0) & (l != 1) & (l != 7))
                    elif self.split == 'unclassified':
                        valid_indices = np.where((l == 0) | (l == 1))
                    else:
                        valid_indices = np.where((l != 7))
                    xyz, rgb, l = xyz[valid_indices], rgb[valid_indices], l[valid_indices]
                    rgb = rgb * (255.0 / 65535.0)
                    l = (l == self.CLASS_OF_INTEREST).astype(int)  # make the labels binary
                    points_data.append(xyz)
                    colors_data.append(rgb)
                    label_data.append(l)

            # flatten the arrays to make them 2D
            ply_data = np.concatenate(points_data, axis=0)
            ply_data = np.concatenate((ply_data, np.concatenate(colors_data, axis=0)), axis=1)
            label_data = np.concatenate(label_data, axis=0)
            print("building processing...")
            if self.split == 'classified':
                raise NotImplementedError('Use the UBCDataset class for processing the classified data')
            for index in tqdm(range(len(buildings)), total=len(buildings)):
                building = buildings[index]
                building_uid = building['properties']['BLDG_UID']
                building_polygon = shape(building['geometry'])
                if building_polygon.geom_type == 'MultiPolygon':
                    bbox = building_polygon.bounds
                    mask = (ply_data[:, 0] >= bbox[0]) & (ply_data[:, 0] <= bbox[2]) & \
                           (ply_data[:, 1] >= bbox[1]) & (ply_data[:, 1] <= bbox[3])
                    masked_labels = label_data[mask]
                    points_with_index = list(enumerate(ply_data[mask]))
                    points_in_bldg, idx_within_building = filter_points_in_multipolygon(points_with_index,
                                                                                        building_polygon)
                    # subtract offsets from x and y
                    points_in_bldg = np.array(points_in_bldg)
                    tmp = np.matmul(self.INVERSE_ROTATION_MATRIX,
                                    np.array([points_in_bldg[:, 0] - self.OFFSET_X,
                                              points_in_bldg[:, 1] - self.OFFSET_Y,
                                              points_in_bldg[:, 2]]))
                    points_in_bldg[:, 0], points_in_bldg[:, 1], points_in_bldg[:, 2] = tmp[0], tmp[1], tmp[2]
                    if self.split == 'unclassified':
                        building_file = os.path.join(self.save_path, building_uid + "_unlabeled.npy")
                    else:
                        building_file = os.path.join(self.save_path, building_uid + "_merged.npy")
                    np.save(building_file, points_in_bldg)

        # get the list of files
        if self.split == 'classified':
            # get the list of training buildings (same buildigns as above)
            self.file_list = [os.path.join(self.save_path, filename) for filename in os.listdir(self.save_path) if
                              filename.endswith('_points.npy')]
            self.label_list = [os.path.join(self.save_path, filename) for filename in os.listdir(self.save_path) if
                               filename.endswith('_labels.txt')]

        elif self.split == 'unclassified':
            # the testing data ends in "unlabeled.npy"
            self.file_list = [os.path.join(self.save_path, filename) for filename in os.listdir(self.save_path) if
                              filename.endswith('_unlabeled.npy')]
        else:
            # the merged data ends in "merged.npy"
            self.file_list = [os.path.join(self.save_path, filename) for filename in os.listdir(self.save_path) if
                              filename.endswith('_merged.npy')]

        self.scene_points_list = []
        # sort the file list and label list by the building uid
        self.file_list.sort(key=lambda x: x.split('/')[-1].split('_')[0])
        if self.split == 'classified':
            self.label_list.sort(key=lambda x: x.split('/')[-1].split('_')[0])
        self.semantic_labels_list = []
        self.building_coord_min, self.building_coord_max = [], []
        for index in tqdm(range(len(self.file_list)), total=len(self.file_list)):
            data = np.load(self.file_list[index])
            xyz = data[:, :3]
            self.scene_points_list.append(data[:, :6])  # xyzrgb
            if self.split == 'classified':
                self.semantic_labels_list.append(np.loadtxt(self.label_list[index]).astype(np.uint8))
            else:
                # we don't have labels for the test data... we're trying to predict them!
                self.semantic_labels_list.append(np.zeros(xyz.shape[0]).astype(np.uint8))
            coord_min, coord_max = np.amin(xyz, axis=0)[:3], np.amax(xyz, axis=0)[:3]
            self.building_coord_min.append(coord_min), self.building_coord_max.append(coord_max)
        assert len(self.scene_points_list) == len(self.semantic_labels_list)

        # calculate label weights
        if self.split == 'classified':
            labelweights = np.zeros(2)
            for seg in self.semantic_labels_list:
                tmp, _ = np.histogram(seg, range(3))
                self.scene_points_num.append(seg.shape[0])
                labelweights += tmp
            labelweights = labelweights.astype(np.float32)
            labelweights = labelweights / np.sum(labelweights)
            self.labelweights = np.power(np.amax(labelweights) / labelweights, 1 / 3.0)
        else:
            # [0.4782623, 0.52173764] are the label weights for the overall training data
            self.labelweights = np.array([0.4782623, 0.52173764])

    def __getitem__(self, index):
        point_set_init = self.scene_points_list[index]
        points = point_set_init[:, :6]
        labels = self.semantic_labels_list[index]
        coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
        grid_x = int(np.ceil(float(coord_max[0] - coord_min[0] - self.block_size) / self.stride) + 1)
        grid_y = int(np.ceil(float(coord_max[1] - coord_min[1] - self.block_size) / self.stride) + 1)
        data_building, label_building, sample_weight, index_building = np.array([]), np.array([]), np.array([]),  np.array([])
        for index_y in range(0, grid_y):
            for index_x in range(0, grid_x):
                s_x = coord_min[0] + index_x * self.stride
                e_x = min(s_x + self.block_size, coord_max[0])
                s_x = e_x - self.block_size
                s_y = coord_min[1] + index_y * self.stride
                e_y = min(s_y + self.block_size, coord_max[1])
                s_y = e_y - self.block_size
                point_idxs = np.where(
                    (points[:, 0] >= s_x - self.padding) & (points[:, 0] <= e_x + self.padding) & (
                                points[:, 1] >= s_y - self.padding) & (
                            points[:, 1] <= e_y + self.padding))[0]
                if point_idxs.size == 0:
                    continue
                num_batch = int(np.ceil(point_idxs.size / self.block_points))
                point_size = int(num_batch * self.block_points)
                replace = False if (point_size - point_idxs.size <= point_idxs.size) else True
                point_idxs_repeat = np.random.choice(point_idxs, point_size - point_idxs.size, replace=replace)
                point_idxs = np.concatenate((point_idxs, point_idxs_repeat))
                np.random.shuffle(point_idxs)
                data_batch = points[point_idxs, :]
                min_z = np.min(data_batch[:, 2])
                data_batch[:, 0] = data_batch[:, 0] - (s_x + self.block_size / 2.0)
                data_batch[:, 1] = data_batch[:, 1] - (s_y + self.block_size / 2.0)
                data_batch[:, 2] = data_batch[:, 2] - min_z
                normalized_xyz = np.zeros((point_size, 3))
                coord_max_tmp = np.max(data_batch[:, 0:3], axis=0)
                normalized_xyz[:, 0] = data_batch[:, 0] / coord_max_tmp[0]
                normalized_xyz[:, 1] = data_batch[:, 1] / coord_max_tmp[1]
                normalized_xyz[:, 2] = data_batch[:, 2] / coord_max_tmp[2]

                data_batch = np.concatenate((data_batch, normalized_xyz), axis=1)
                label_batch = labels[point_idxs].astype(int)
                batch_weight = self.labelweights[label_batch]

                data_building = np.vstack([data_building, data_batch]) if data_building.size else data_batch
                label_building = np.hstack([label_building, label_batch]) if label_building.size else label_batch
                sample_weight = np.hstack([sample_weight, batch_weight]) if sample_weight.size else batch_weight
                index_building = np.hstack([index_building, point_idxs]) if index_building.size else point_idxs
        data_building = data_building.reshape([-1, self.block_points, data_building.shape[1]])
        label_building = label_building.reshape([-1, self.block_points])
        sample_weight = sample_weight.reshape([-1, self.block_points])
        index_building = index_building.reshape([-1, self.block_points])
        return data_building, label_building, sample_weight, index_building

    def __len__(self):
        return len(self.scene_points_list)