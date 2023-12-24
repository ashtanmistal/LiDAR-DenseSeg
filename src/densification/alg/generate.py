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
    So we need to subdivide the pointcloud into smaller blocks based on a median plane,
    and then upsample each subcloud, re-assemble them.
    :param cloud:
    :return:
    """
    if cloud.shape[0] < 100:  # k is from tree1.query(p_split[i], 100) in generateiopoint
        return cloud

    bbox = np.array([np.min(cloud, axis=0), np.max(cloud, axis=0)])
    # if the number of points is less than 5000 (max KDtree size), upsample it directly
    # Greater performance is found via recursive subdivision than trying to
    # significantly increase KDtree size (to 131072 or higher)
    if cloud.shape[0] < 5000:

        loc = np.mean(bbox, axis=0)
        scale = np.max(bbox[1] - bbox[0])
        for x in range(cloud.shape[0]):
            cloud[x] = cloud[x] - loc
            cloud[x] = cloud[x] / scale
        np.savetxt("test.xyz", cloud)
        cloud = np.expand_dims(cloud, 0)  # (5000, 3) -> (1, 5000, 3)

        # upsampling
        pointcloud = np.array(generator.upsample(cloud))

        print("farthest point sample")
        # ndarray (1, 5000, 3)

        pointnumber = UPSAMPLE_AMOUNT * cloud.shape[1]

        centroids = farthest_point_sample(pointcloud, pointnumber)

        # undo the scaling
        for x in range(pointcloud.shape[0]):
            pointcloud[x] = pointcloud[x] * scale
            pointcloud[x] = pointcloud[x] + loc

        return pointcloud[centroids]
    else:
        # subdivide the pointcloud into 2 based on a median plane that splits x, y, or z in half based on
        # the largest dimension
        largest_dim = np.argmax(bbox[1] - bbox[0])
        median = np.median(cloud[:, largest_dim])
        pc1 = subdivide_and_upsample(cloud[cloud[:, largest_dim] < median])
        pc2 = subdivide_and_upsample(cloud[cloud[:, largest_dim] >= median])
        return np.concatenate((pc1, pc2), axis=0)


def obj_to_xyz(obj_file):
    # data is of the form:
    # 1825.157331 -152.869169 67.399000 0 255 0
    # Only keep the point if the color is red
    xyz = []
    for x in range(obj_file.shape[0]):
        if obj_file[x][3] == 255 and obj_file[x][4] == 0 and obj_file[x][5] == 0:
            xyz.append([obj_file[x][0], obj_file[x][1], obj_file[x][2]])
    xyz = np.array(xyz, dtype=np.float32)
    return xyz


def xyz_to_obj(xyz):
    # add the v and color to the xyz
    obj = []
    for x in range(xyz.shape[0]):
        obj.append(['v', xyz[x][0], xyz[x][1], xyz[x][2], '255', '0', '0'])
    obj = np.array(obj)
    return obj


UPSAMPLE_AMOUNT = 2

datalist = []
outlist = []

out_dir = 'data_out'

for root, dirs, files in os.walk("data_in"):
    for file in files:
        if file.endswith("_pred.obj"):
            datalist.append(os.path.join(root, file))
            outlist.append(os.path.join(out_dir, file))



# sort by file size (ensure datalist and outlist are in the same order)
# this will assist with faster debugging
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
    # normalization
    xyzname = datalist[k]
    cloud = np.loadtxt(xyzname, usecols=(1, 2, 3, 4, 5, 6))
    cloud = obj_to_xyz(cloud)
    cloud = cloud[:, 0:3]

    # subdivide and upsample
    print("total points: " + str(cloud.shape[0]))
    pointcloud = subdivide_and_upsample(cloud)

    # save pointcloud
    pc_obj = xyz_to_obj(pointcloud)
    np.savetxt(outlist[k], pc_obj, fmt='%s')
    print("done")