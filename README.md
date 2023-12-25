# SAPCU-Recursive

This is a recursive implementation of the SAPCU algorithm. The original implementation has a limit of 5000 points. This implementation can handle point clouds of any size without the need to modify the size of the kdtree, and recursively subdivides the point cloud based on the median of the largest dimension on each split, and stitches the results together.

This also adapts the algorithm to run on `.obj` files as opposed to `.ply` files; this does require the point cloud to have either the same color for all points, or for the upsampling to only be ran on one color.

## Usage

Intended usage is for large point clouds (e.g. buildings; tested on 100k points). 
To run the algorithm, run `python generate.py`. The following arguments are available:
```
--input: input path (directory to .obj files)
--output: output path (directory to save .obj files)
--upsample: upsample amount (e.g. 2 for 2x upsampling)
--cuda: use cuda
--color: filter color (Algorithm will upsample points of this color)
```


The color motivation of the color handling is for the use case of color being used as labels (e.g. output from a semantic segmentation algorithm like PointNet++). In this case, the user should specify the color corresponding to the label, and the algorithm will upsample the points belonging to that label.

## Notes

- The recursive nature does leave some visible seams in the split. A fix is planned to be implemented in the future; this will involve a more complex stitching process and a blending of the seams as opposed to a simple concatenation.
- Further efficiency improvements are planned.

___

# SAPCU
【Code of CVPR 2022 paper】 

Self-Supervised Arbitrary-Scale Point Clouds Upsampling via Implicit Neural Representation

Paper address: https://arxiv.org/abs/2204.08196

## Environment
Pytorch 1.9.0

CUDA 10.2

## Evaluation
### a. Download models
Download the pretrained models from the link and unzip it to  `./out/`
```
https://pan.baidu.com/s/1OPVnCHq129DBMWh5BA2Whg 
access code: hgii 

or

https://drive.google.com/file/d/12TifjDW2L7r2LK3AGDgabHdN3f0xY5DR/view?usp=share_link
```
### b. Compilation
Run the following command for compiling dense.cpp which generates dense seed points.
Note that the size of the input point cloud is currently limited to 5000, if you want to change this limit, you need to modify the size of the kdtree.
```
g++ -std=c++11 dense.cpp -O2 -o dense
```
### c. Evaluation
You can now test our code on the provided point clouds in the `test` folder. To this end, simply run
```
python generate.py
```
The 4X upsampling results will be created in the `testout` folder.

Ground truth are provided by [Meta-PU](https://drive.google.com/file/d/1dnSgI1UXBPucZepP8bPhfGYJEJ6kY6ig/view?usp=sharing)

## Training
Download the training dataset from the link and unzip it to the working directory
```
https://pan.baidu.com/s/1VQ-3RFO02fQfcLBfqvCBZA 
access code: vpfm

or

https://1drv.ms/f/s!AsP2NtMX-kUTml4U3DYUD6Hy9FJn?e=8QfJTH
```

Then run the following commands for training our network
```
python trainfn.py
python trainfd.py
```
## Dataset
We present a fast implementation for building the dataset, which is based on [occupancy_networks](https://github.com/autonomousvision/occupancy_networks/).
### a. Preprocessing
Follow the link [occupancy_networks](https://github.com/autonomousvision/occupancy_networks#building-the-dataset) to obtain pointclouds and watertight meshes, notice that we only use ShapeNet dataset v1.

### b. Building and installing
Then move scripts to occupancy_networks/scripts run the following commands for building traindata for fd and fn:
```
bash dataset_shapenet/build-fd.sh
bash dataset_shapenet/build-fn.sh
bash dataset_shapenet/installfd.sh
bash dataset_shapenet/installfn.sh
```
## Evaluation
The code for evaluation can be download from:
```
https://github.com/pleaseconnectwifi/Meta-PU/tree/master/evaluation_code
https://github.com/jialancong/3D_Processing
```
## Citation
If the code is useful for your research, please consider citing:
  
    @inproceedings{sapcu,
      title = {Self-Supervised Arbitrary-Scale Point Clouds Upsampling via Implicit Neural Representation},
      author = {Wenbo Zhao, Xianming Liu, Zhiwei Zhong, Junjun Jian, Wei Gao, Ge Li, Xiangyang Ji},
      booktitle = {Proceedings IEEE Conf. on Computer Vision and Pattern Recognition (CVPR)},
      month     = {June},
      year      = {2022},
      pages     = {1999-2007}
    }


## Acknowledgement
The code is based on [occupancy_networks](https://github.com/autonomousvision/occupancy_networks/) and [DGCNN](https://github.com/WangYueFt/dgcnn), If you use any of this code, please make sure to cite these works.
