import os

import numpy as np
import torch
from tqdm import tqdm

import fd.checkpoints
import argparse
import fd.config
import fn.checkpoints
import fn.config
from generation import Generator3D6


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


def subdivide_and_upsample(cloud, upsample_amount, generator):
    """
    The generator can only handle 5000 points at a time.
    So we need to subdivide the pointcloud into smaller blocks based on a median plane,
    and then upsample each subcloud, re-assemble them.
    :param cloud: The pointcloud to upsample
    :param upsample_amount: The amount of upsampling to do (as a ratio)
    :param generator: The generator to use
    :return: The upsampled pointcloud
    """
    if cloud.shape[0] < 100:  # k is from tree1.query(p_split[i], 100) in `generateiopoint`
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

        pointnumber = upsample_amount * cloud.shape[1]

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
        pc1 = subdivide_and_upsample(cloud[cloud[:, largest_dim] < median], upsample_amount, generator)
        pc2 = subdivide_and_upsample(cloud[cloud[:, largest_dim] >= median], upsample_amount, generator)
        return np.concatenate((pc1, pc2), axis=0)


def obj_to_xyz(obj_file, filter_color):
    # data is of the form:
    # 1825.157331 -152.869169 67.399000 0 255 0
    xyz = []
    r, g, b = filter_color
    for x in range(obj_file.shape[0]):
        if obj_file[x][3] == r and obj_file[x][4] == g and obj_file[x][5] == b:
            xyz.append([obj_file[x][0], obj_file[x][1], obj_file[x][2]])
    xyz = np.array(xyz, dtype=np.float32)
    return xyz


def xyz_to_obj(xyz, filter_color):
    # add the v and color to the xyz
    obj = []
    for x in range(xyz.shape[0]):
        obj.append(['v', xyz[x][0], xyz[x][1], xyz[x][2], filter_color[0], filter_color[1], filter_color[2]])
    obj = np.array(obj)
    return obj


def parse_tuple(string):
    try:
        return tuple(map(int, string.split(',')))
    except:
        raise argparse.ArgumentTypeError("Tuple must be x,y")



def parse_args():
    parser = argparse.ArgumentParser(description='Recursive SAPCU Upsampling')
    parser.add_argument('--input', type=str, default='data_in', help='input path')
    parser.add_argument('--output', type=str, default='data_out', help='output path')
    parser.add_argument('--upsample', type=int, default=2, help='upsample amount')
    parser.add_argument('--cuda', type=bool, default=True, help='use cuda')
    parser.add_argument('--color', type=parse_tuple, default='255, 0, 0', help='filter color')
    args = parser.parse_args()
    return args


def main(args):
    data_list = []
    out_list = []

    # out_dir = 'data_out'
    out_dir = args.output
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for root, dirs, files in os.walk(args.input):
        for file in files:
            if file.endswith("_pred.obj"):
                data_list.append(os.path.join(root, file))
                out_list.append(os.path.join(out_dir, file))

    # sort by file size (assists with faster debugging)
    data_list, out_list = zip(*sorted(zip(data_list, out_list), key=lambda x: os.path.getsize(x[0])))

    is_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if is_cuda else "cpu")

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

    for k in range(len(data_list)):
        upsample_amount_total = args.upsample
        print("processing " + data_list[k])
        # clear test.xyz and target.xyz (make them empty)
        open("test.xyz", 'w').close()
        open("target.xyz", 'w').close()
        # normalization
        xyzname = data_list[k]
        main_cloud = np.loadtxt(xyzname, usecols=(1, 2, 3, 4, 5, 6))
        filter_color = args.color
        main_cloud = obj_to_xyz(main_cloud, filter_color)
        main_cloud = main_cloud[:, 0:3]

        # subdivide and upsample
        print("total points: " + str(main_cloud.shape[0]))
        pointcloud = subdivide_and_upsample(main_cloud, upsample_amount_total, generator)

        # save pointcloud
        pc_obj = xyz_to_obj(pointcloud, filter_color)
        np.savetxt(out_list[k], pc_obj, fmt='%s')
        print("done")


if __name__ == '__main__':
    """
    Example command line usage (using default values):
    python generate.py --input data_in --output data_out --upsample 2 --cuda True --color 255,0,0
    """
    args = parse_args()
    print(args)
    main(args)
