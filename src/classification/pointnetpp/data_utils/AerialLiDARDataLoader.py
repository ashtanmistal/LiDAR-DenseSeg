import os
import sys
import torch, time, random
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
import json
from shapely.geometry import Polygon, MultiPolygon, shape, Point


def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:, :3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point, centroids.astype(np.int32)


def is_within_multipolygon(point, multipolygon):
    """Check if a point is within any of the polygons in a MultiPolygon."""
    return any([point.within(polygon) for polygon in multipolygon])


def filter_points_in_polygon(points_with_index, geometry):
    """
    Filter points that are within a polygon.
    Usage: points = filter_points_in_polygon(points, polygon)
    It is helpful to have already filtered the points by the bounding box of the polygon for efficiency.
    :param points_with_index: the points to filter
    :param geometry: the geometry to filter by
    :return: the filtered points
    """

    # if isinstance(geometry, MultiPolygon):
    #     filtered = [(i, p) for i, p in points_with_index if is_within_multipolygon(Point(p[:2]), geometry)]
    # else:
    filtered = [(i, p) for i, p in points_with_index if Point(p[:2]).within(geometry)]

        # Separating indices and points for return
    indices, within_geometry = zip(*filtered) if filtered else ([], [])
    return list(within_geometry), list(indices)


class AerialLiDARDatasetWholeScene(Dataset):
    """
    Loader for whole scene data from las files.
    Code attribution: This code is significantly based on the PyTorch
    implementation of PointNet++ by Charles R. Qi, adapted here to work robustly with aerial .las files instead of
    geared towards the S3DIS dataset. The original code can be found here:
    https://github.com/yanx27/Pointnet_Pointnet2_pytorch/blob/master/data_utils/S3DISDataLoader.py
    """

    NUM_CLASSES = 17

    def __init__(self, split='train', file_path="data/data_label", data_root="data", num_point=1024, block_size=16.0,
                 padding=0.001, process_data=False):
        # TODO figure out what to do for the split
        """
        Initialize the dataset with the path to the .las file
        :param file_path: the path to the .las file
        :param num_point:
        :param block_size:
        :param padding:
        """
        super().__init__()
        self.num_point = num_point
        self.block_size = block_size
        self.padding = padding
        self.process_data = process_data
        file_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        file_path = os.path.join(file_path, 'data', 'data_label', 'classified_merged_points.npy')
        sys.path.append(file_path)
        data_all = np.load(file_path)
        self.points, self.labels = data_all[:, :6], data_all[:, 6]  # xyzrgb, label
        # rgb right now is between 0 and 65535, so we need to make it between 0 and 255
        # make it a float first, then divide by 65535.0, then multiply by 255.0
        self.points[:, 3:6] = self.points[:, 3:6].astype(np.float32)
        self.points[:, 3:6] = self.points[:, 3:6] / 65535.0
        self.points[:, 3:6] = self.points[:, 3:6] * 255.0
        # turn the rgb values into integers
        self.points[:, 3:6] = self.points[:, 3:6].astype(np.int32)
        if self.points.shape[0] < self.num_point:
            raise ValueError('Dataset size must be larger than num_point.')
        self.labelweights = self.calculate_label_weights(self.labels)

        self.save_path = os.path.join(data_root, 'buildings_split')
        if self.process_data:
            """
            In order to process the data, we need to go through each building in the geojson, calculate the points 
            that intersect with the building, and then save those points to a file
            corresponding to the building. We will also create `self.list_of_points` and `self.list_of_labels` to be able
            to access the data in the future in the __getitem__ function.
            """
            print('Processing data %s (only running in the first time)...' % self.save_path)
            buffered_geojson = os.path.join(data_root, 'buffer.geojson')
            buildings = json.load(open(buffered_geojson))['features']

            self.list_of_points = [None] * len(buildings)
            self.list_of_labels = [None] * len(buildings)
            for index in tqdm(range(len(buildings)), total=len(buildings)):
                building = buildings[index]
                building_uid = building['properties']['BLDG_UID']
                building_polygon = shape(building['geometry'])
                if building_polygon.geom_type == 'Polygon':
                    raise NotImplementedError('Polygon type not implemented yet; only MultiPolygon')
                    # polygon is not implemented because we have buffered all the geometries
                    # there does not exist a single Polygon in the geojson we are working with
                elif building_polygon.geom_type == 'MultiPolygon':
                    # get bounds of the building
                    bbox = building_polygon.bounds
                    points_with_index = [(i, p) for i, p in enumerate(self.points) if
                                         bbox[0] <= p[0] <= bbox[2] and bbox[1] <= p[1] <= bbox[3]]
                    # filter points within the building
                    points_within_building, within_building = filter_points_in_polygon(points_with_index,
                                                                                       building_polygon)
                    # save the points to a file
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
            buildings = json.load(open(os.path.join(data_root, 'buffer.geojson')))
            buildings = buildings['features']
            self.list_of_points = [None] * len(buildings)
            self.list_of_labels = [None] * len(buildings)
            for index in tqdm(range(len(buildings)), total=len(buildings)):
                building = buildings[index]
                building_uid = building['properties']['BLDG_UID']
                building_file = os.path.join(self.save_path, building_uid + "_points.npy")
                building_labels_file = os.path.join(self.save_path, building_uid + "_labels.npy")
                self.list_of_points[index] = np.load(building_file)
                self.list_of_labels[index] = np.load(building_labels_file)



    def calculate_label_weights(self, labels, num_classes=NUM_CLASSES):
        epsilon = 1e-6  # Small constant to prevent divide by zero
        labelweights = np.histogram(labels, range(num_classes + 1))[0]
        labelweights = labelweights.astype(np.float32)
        labelweights += epsilon  # Add epsilon to each label weight
        labelweights = labelweights / np.sum(labelweights)
        # Adjust the weights to account for the class imbalance
        labelweights = np.power(np.amax(labelweights) / labelweights, 1 / 3.0)
        print("labelweights: ", labelweights)
        return labelweights

    def __len__(self):
        return self.points.shape[0] // self.num_point

    def __getitem__(self, idx):
        # TODO: We need to do the following:
        # - Train / test split
        # - Separate the dataset into each building (this will also assist with the train / test split!!)
        # - This requires dataset preprocessing similar to ModelNetDataLoader.py
        # - Then on the __getitem__ function,
        #   - we query the idx of the building, and *then* we can do the farthest point stuff

        selected_points, selected_labels = self.list_of_points[idx], self.list_of_labels[idx]
        sampled_points, sampled_labels = farthest_point_sample(selected_points, self.num_point)
        center = np.mean(sampled_points, axis=0)


        # normalize the points; TODO check if this is the right way to do it
        coord_max = np.max(selected_points[:, :3], axis=0)
        current_points = np.zeros((self.num_point, 9))  # num_point * 9
        current_points[:, 6] = selected_points[:, 0] / coord_max[0]  # x
        current_points[:, 7] = selected_points[:, 1] / coord_max[1]  # y
        current_points[:, 8] = selected_points[:, 2] / coord_max[2]  # z
        selected_points[:, 0] = selected_points[:, 0] - center[0]  # x
        selected_points[:, 1] = selected_points[:, 1] - center[1]  # y
        selected_points[:, 3:6] /= 255.0
        current_points[:, 0:6] = selected_points

        current_labels = np.zeros((self.num_point, 1))  # num_point * 1
        current_labels[:, 0] = selected_labels

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
