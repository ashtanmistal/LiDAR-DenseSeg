# LiDAR-DenseSeg
A pipeline for point cloud densification and architectural semantic segmentation for improving voxelization and mesh reconstruction quality of airborne LiDAR data. 

___

## Introduction

This repository contains the code for my CPSC 533Y (Deep Learning with Visual Geometry) final project of the same name. The pipeline was developed with the intention of improving the mesh reconstruction and voxelization of LiDAR data taken of the UBC Vancouver Campus. The latter was a task undertaken for visualizing the UBC campus in a Minecraft world; the proof-of-concept can be found [here](https://github.com/ashtanmistal/minecraftUBC).
Consequently, the project is split into two parts: the first part is a point cloud densification pipeline, and the second part is a semantic segmentation pipeline. The two parts are independent of each other, and can be run separately.
Nonetheless it is recommended to run the densification pipeline first, as a higher quality point cloud dataset will improve the quality of the segmentation pipeline.


## Dependencies


See `requirements.txt` for a list of software dependencies. The code was developed and tested on Windows 11 with Python 3.11 and CUDA 12.3. The code is not guaranteed to work on other platforms, but to my knowledge there is nothing in the code that is platform-specific.


Hardware-wise, the code was developed and the models were trained on the following hardware:

- CPU: Intel Core i7-13700K

- GPU: NVIDIA GeForce RTX 4070 Ti (12 GB VRAM)

- RAM: 48 GB

Minimum storage requirements are around 100 GB, almost entirely due to the size of the LiDAR dataset.

## Densification Pipeline

The densification is outlined as follows:

1. Classification of the unclassified points in the point cloud dataset using a PointNet++ model trained on the data that was classified by the original LiDAR sensor. The majority of points are classified already, however a significant number of points that would otherwise be buildings are unclassified. Without this step, the skeleton of some buildings is suboptimally reconstructed.
2. Division of the LiDAR data into individual buildings. The division into individual buildings is done with the aid of UBC operational geospatial data, which contains the outlines of buildings on campus in a geojson file. The division is done using a built-in QGIS tool. Should similar polygons not be available for other datasets, division can be done using a basic DBSCAN clustering algorithm. Each building is saved as a separate point cloud dataset. 
3. Point Cloud Densification. A point cloud upsampling algorithm using a graph convolutional neural network is applied to the LiDAR data. The algorithm is based on the paper [Point Cloud Upsampling using Graph Convolutional Networks](https://arxiv.org/abs/1912.03264) by Qian et al, and is re-trained on the UBC LiDAR dataset. The code for this step is derived from the corresponding [GitHub repository](https://github.com/guochengqian/PU-GCN). The output of this step is a densified point cloud dataset for each building.


The densification step will directly improve the voxelization of the LiDAR data, as we will have more data to work with when creating the voxelization. Mesh reconstruction is another step that directly benefits from additional points. 

## Semantic Segmentation Pipeline

It is still TBD whether this step will be implemented as a component of the final course project; this is due to time constraints as well as potential training time in the other components of the pipeline.

The pipeline for this approach would be based on the [GeoSegNet](https://link.springer.com/article/10.1007/s00371-023-02853-7) paper by Chen et al, which is a semantic segmentation model that uses geometric encoder-decoder models and is trained on the [S3DIS](http://buildingparser.stanford.edu/dataset.html) dataset. The code for this step is derived from the corresponding [GitHub repository](https://github.com/Chen-yuiyui/GeoSegNet).

Currently, there is no pre-trained model available for *outdoor* building data, and the research papers that outlined such a task do not publicly share their code. GeoSegNet is trained on indoor data. There are two options for this step:

1. Training a model from scratch on the UBC LiDAR dataset. This is the most time-consuming option, as manual labelling of some of the dataset will be required. However, it is also the most flexible option, and the option that is likely to work best.
2. Using the pre-trained model that is trained on indoor data. Even though the data is not the same, the model may still be able to generalize to outdoor data. The lighting on each point is likely to be the biggest hindrance to this approach, as indoor data is usually taken with a fixed light source, whereas outdoor data is taken with the sun as the light source. 
    a. If this approach is taken, it is unclear whether the existing model requires the mesh to be closed or not; if this is the case, we will need to densify and close the corresponding digital elevation model (DEM) of the UBC campus. This was done in the proof-of-concept, but was done after the voxelization had been imported into a Minecraft map. As a result a different approach will need to be addressed. If time does not permit for a quality closed DEM, then we will perform a linear interpolation between the lowest points on a per-building basis to close the mesh.

The primary motivation behind semantic segmentation is that it will allow for a smart selection of blocks when voxelizing the model. Segmentation will allow for the selection of blocks based on the semantic class of the point, instead of just the color data and other LiDAR-inherent data. We will also be able to color the mesh based on the semantic class. 

## Current Hindrances

The biggest hindrance to this project is the lack of a pre-trained model for outdoor building data. I'm still working on finding a pre-trained model that can be used for this project, but it is likely that I will have to train a model from scratch. This is a time-consuming process, and I'm not sure if I will have enough time to do this. 
If that does become the case, I will likely have to cut out the semantic segmentation step of the pipeline, and focus on the densification step. Even if only the densification step is completed, the pipeline will still be useful for improving the quality of the voxelization and mesh reconstruction of the UBC campus. Further steps will be taken in the densification if we are unable to perform semantic segmentation to improve the voxelization.
