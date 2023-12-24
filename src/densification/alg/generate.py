import torch
# import torch.distributions as dist
import os
import shutil
import argparse
from tqdm import tqdm
import time
import trimesh
import random
from collections import defaultdict
import fd.config
import fn.config
import fn.checkpoints
import fd.checkpoints
from generation import Generator3D6
import numpy as np


def farthest_point_sample(xyz, pointnumber):
    device = 'cuda'
    N, C = xyz.shape
    torch.seed()
    xyz = torch.from_numpy(xyz).float().to(device)
    centroids = torch.zeros(pointnumber, dtype=torch.long).to(device)

    distance = torch.ones(N).to(device) * 1e32
    farthest = torch.randint(0, N, (1,), dtype=torch.long).to(device)
    farthest[0] = N / 2
    for i in tqdm(range(pointnumber)):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids.detach().cpu().numpy().astype(int)


def subdivide_and_upsample(cloud):
    """
    The generator can only handle 5000 points at a time.
    So we need to subdivide the pointcloud into smaller cubic chunks (8 subclouds)
    and then upsample each subcloud, re-assemble them.
    :param cloud:
    :return:
    """
    if cloud.shape[0] < 100:  # k is from tree1.query(p_split[i], 100) in generateiopoint
        return cloud
    bbox = np.zeros((2, 3))
    bbox[0][0] = np.min(cloud[:, 0])
    bbox[0][1] = np.min(cloud[:, 1])
    bbox[0][2] = np.min(cloud[:, 2])
    bbox[1][0] = np.max(cloud[:, 0])
    bbox[1][1] = np.max(cloud[:, 1])
    bbox[1][2] = np.max(cloud[:, 2])
    loc = (bbox[0] + bbox[1]) / 2
    scale = (bbox[1] - bbox[0]).max()
    scale1 = 1 / scale
    # if the number of points is less than 5000 (max KDtree size), upsample it directly
    # Greater performance is found via recursive subdivision than trying to
    # significantly increase KDtree size (to 131072 or higher)
    if cloud.size <= 5000:
        for x in range(cloud.shape[0]):
            cloud[x] = cloud[x] - loc
            cloud[x] = cloud[x] * scale1
        np.savetxt("test.xyz", cloud)
        cloud = np.expand_dims(cloud, 0)

        # upsampling
        pointcloud = np.array(generator.upsample(cloud))

        # farthest_point_sample
        print("farthest point sample")
        for x in range(pointcloud.shape[0]):
            pointcloud[x] = pointcloud[x] * scale
            pointcloud[x] = pointcloud[x] + loc

        pointnumber = UPSAMPLE_AMOUNT * pointcloud.shape[0]

        centroids = farthest_point_sample(pointcloud, pointnumber)
        return pointcloud[centroids]
    else:
        # recursively subdivide the pointcloud into 8 subclouds
        subclouds = []
        for x in range(2):
            for y in range(2):
                for z in range(2):
                    subcloud = cloud[np.where((cloud[:, 0] >= bbox[0][0] + x * scale / 2) &
                                              (cloud[:, 0] < bbox[0][0] + (x + 1) * scale / 2) &
                                              (cloud[:, 1] >= bbox[0][1] + y * scale / 2) &
                                              (cloud[:, 1] < bbox[0][1] + (y + 1) * scale / 2) &
                                              (cloud[:, 2] >= bbox[0][2] + z * scale / 2) &
                                              (cloud[:, 2] < bbox[0][2] + (z + 1) * scale / 2))]
                    subclouds.append(subcloud)
        subclouds = np.array(subclouds, dtype=object)
        pointcloud = []
        for x in range(8):
            pointcloud.append(subdivide_and_upsample(subclouds[x]))
        pointcloud = np.concatenate(pointcloud, axis=0)
        return pointcloud


UPSAMPLE_AMOUNT = 2
datalist = []
outlist = []

out_dir = 'buildings_out'

for root, dirs, files in os.walk("buildings_split"):
    for file in files:
        if file.endswith(".xyz"):
            datalist.append(os.path.join(root, file))
            outlist.append(os.path.join(out_dir, file))

datalist, outlist = zip(*sorted(zip(datalist, outlist), key=lambda x: os.path.getsize(x[0])))

is_cuda = (torch.cuda.is_available())
device = torch.device("cuda" if is_cuda else "cpu")

out_dir = 'out/pointcloud/opu'

cfg1 = fn.config.load_config('configs/fn.yaml')
cfg2 = fd.config.load_config('configs/fd.yaml')

model = fn.config.get_model(cfg1, device)
model2 = fd.config.get_model(cfg2, device)

checkpoint_io1 = fn.checkpoints.CheckpointIO('out/fn', model=model)
load_dict = checkpoint_io1.load('model_best.pt')

checkpoint_io2 = fd.checkpoints.CheckpointIO('out/fd', model=model2)
load_dict = checkpoint_io2.load('model_best.pt')

model.eval()
model2.eval()

generator = Generator3D6(model, model2, device)

for k in range(len(datalist)):
    print("processing " + datalist[k])
    # clear test.xyz and target.xyz (make them empty)
    open("test.xyz", 'w').close()
    open("target.xyz", 'w').close()
    xyzname = datalist[k]
    cloud = np.loadtxt(xyzname)
    cloud = cloud[:, 0:3]

    # subdivide and upsample
    pointcloud = subdivide_and_upsample(cloud)

    np.savetxt(outlist[k], pointcloud)
    print("done")
