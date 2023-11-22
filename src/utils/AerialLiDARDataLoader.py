import os
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset

from src.utils.data_helper import read_las_file


class AerialLiDARDatasetWholeScene(Dataset):
    """
    Loader for whole scene data from las files.
    Code attribution: This code is significantly based on the PyTorch
    implementation of PointNet++ by Charles R. Qi, adapted here to work robustly with aerial .las files instead of
    geared towards the S3DIS dataset. The original code can be found here:
    https://github.com/yanx27/Pointnet_Pointnet2_pytorch/blob/master/data_utils/S3DISDataLoader.py
    """

    NUM_CLASSES = 17  # TODO Adjust the number of classes if the dataset changes

    def __init__(self, file_path, block_points=4096, block_size=1.0, padding=0.001):
        """
        Initialize the dataset with the path to the .las file
        :param file_path: the path to the .las file
        :param block_points:
        :param block_size:
        :param padding:
        """
        super().__init__()  # TODO is this necessary?
        self.block_points = block_points
        self.block_size = block_size
        self.padding = padding
        self.points, self.colors, self.labels = self.load_las_data(file_path)
        self.labelweights = self.calculate_label_weights(self.labels)

    def load_las_data(self, file_path):
        return read_las_file(file_path)

    def calculate_label_weights(self, labels, num_classes=NUM_CLASSES):
        labelweights = np.zeros(num_classes)
        labelweights = np.histogram(labels, range(num_classes + 1))[0]
        labelweights = labelweights.astype(np.float32)
        labelweights = labelweights / np.sum(labelweights)
        # Now, we need to adjust the weights to account for the class imbalance
        # This method is what PointNet++ uses (TODO check what the method is called)
        labelweights = np.power(np.amax(labelweights) / labelweights, 1 / 3.0)
        print("labelweights: ", labelweights)
        return labelweights

    def __len__(self):
        return len(self.points)

    def __getitem__(self, idx):
        start_idx = idx * self.block_points
        end_idx = (idx + 1) * self.block_points
        points_sample = self.points[start_idx:end_idx]
        labels_sample = self.labels[start_idx:end_idx]
        weights_sample = self.labelweights[labels_sample]
        # TODO Any additional preprocessing
        return points_sample, labels_sample, weights_sample


if __name__ == '__main__':
    # TODO Update the file_path to the .las file location
    file_path = '/replace/with/path/to/las/file.las'

    # Initialize the dataset with the path to the .las file
    aerial_data = AerialLiDARDatasetWholeScene(file_path=file_path)

    print('Aerial LiDAR data size:', aerial_data.__len__())
    print('Aerial LiDAR data 0 shape:', aerial_data.__getitem__(0)[0].shape)
    print('Aerial LiDAR label 0 shape:', aerial_data.__getitem__(0)[1].shape)

    import torch, time, random

    manual_seed = 123
    random.seed(manual_seed)
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)


    def worker_init_fn(worker_id):
        random.seed(manual_seed + worker_id)


    # Create the DataLoader for the aerial LiDAR data
    train_loader = torch.utils.data.DataLoader(aerial_data, batch_size=16, shuffle=True, num_workers=16,
                                               pin_memory=True, worker_init_fn=worker_init_fn)

    # Iterate through the DataLoader
    for idx, (data, labels, weights) in enumerate(train_loader):
        end = time.time()
        for i, (input, target) in enumerate(train_loader):
            print('time: {}/{}--{}'.format(i + 1, len(train_loader), time.time() - end))
            end = time.time()
