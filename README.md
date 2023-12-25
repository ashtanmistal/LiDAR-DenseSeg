# LiDAR-DenseSeg
A pipeline for point cloud densification and architectural semantic segmentation for improving voxelization and mesh reconstruction quality of airborne LiDAR data. 

> *This paper introduces LiDAR-DenseSeg, a novel framework designed to enhance voxelization and mesh reconstruction of airborne LiDAR data. The process involves three pivotal steps: Semantic segmentation, densification, and planar flattening of the point cloud. Using a modified PointNet++ architecture, the framework effectively segments point cloud data, focusing primarily on building structures and achieving an evaluation accuracy of 94.1% and a mean IoU of 0.89. Following segmentation, a recursive median split algorithm based on the SAPCU architecture densifies the point cloud, addressing the inherent sparsity in airborne LiDAR data. Planar flattening is proposed to further refine the process, reducing noise and enhancing voxelization quality. The paper presents empirical results demonstrating significant improvements in the voxelization and mesh reconstruction of airborne LiDAR data, contributing to the field of LiDAR data processing and 3D reconstruction.*


This project contains submodules. To clone the repository, use the following command:

```bash
git clone --recurse-submodules https://github.com/ashtanmistal/LiDAR-DenseSeg.git
```
        

___

## Introduction

This repository contains the code for my CPSC 533Y (Deep Learning with Visual Geometry) final project of the same name. The pipeline was developed with the intention of improving the mesh reconstruction and voxelization of LiDAR data taken of the UBC Vancouver Campus. The latter was a task undertaken for visualizing the UBC campus in a Minecraft world; the proof-of-concept can be found [here](https://github.com/ashtanmistal/minecraftUBC).


## Pipeline

The semantic segmentation of the point cloud is performed using a PointNet++ architecture that is trained on the already labelled data. The data is a partially labelled point cloud of the University of British Columbia campus, with the main labels of buildings, trees, water, and ground. This framework focuses on labelling buildings, as these are the most important features to capture in the voxelization and mesh reconstruction process and the ones that suffer the most from the aforementioned issues. The semantic segmentation problem is thus formulated as a binary problem, with the two classes being _buildings_ and _not buildings_.

The densification of the point cloud is performed using a recursive median split implementation of the [SAPCU architecture](https://github.com/xnowbzhao/sapcu) using pre-trained weights from the authors. A weighted nearest neighbours algorithm is used to transfer the colour information from the original point cloud to the densified point cloud. The densification task is performed only on the buildings, and primarily enhances mesh reconstruction of the buildings given the large voxel size in the voxelization process.

To resolve the issue of the irregularity of the point cloud and fix the amplification of noise that occurs during the voxelization process, planar flattening is performed on the densified point cloud. For points that lie on a plane, they are brought to the plane along their normal. As a result, points are much more likely to get voxelized to the same voxel as their neighbours. This leads to a massive reduction in the noise of the point cloud, and an increase in voxelization quality.
