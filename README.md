# Auto Annotation of Pathology Images

Columbia Data Science Institute Capstone Project, Fall 2020

Mentor: Dr. Adler Perotte

Instructor: Dr. Adam S. Kelleher

Team member:

Yihao Li, Chao Huang, Yufeng Ma, Xiaoyun Zhu, Shuo Yang

This project aims to create a machine learning-driven user interface for the annotation of very large
pathology images. Each image may be 10s of thousands by 10s of thousands of pixels. As a result,
annotation of the entire slide for object recognition or semantic/instance segmentation can be time
consuming when entities are only a few pixels in diameter. This project aims to build a framework for
maximally leveraging expert annotator (clinician) time by interleaving annotation (label generation) with
inference to provide an intuitive notion of model fit and the minimal amount of labeling required for
acceptable model performance.

## Installation
1. Required packages can be found in the [requirements](requirements.txt) file,
it's recommended to use a virtual environment to install all required packages through pip.
2. Note that although [detectron2](https://github.com/facebookresearch/detectron2) is used in this repository,
it's **NOT** explicitly listed in the requirements due to its complex dependencies on the version of PyTorch and CUDA. 
Therefore, it's better to build it from source by following the [official guide](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md#build-detectron2-from-source).

## Repository Structure
1. [Collage Generator](Collage_generator): the module for generating synthetic whole slide images (a.k.a, collages) from vignettes,
which utilize a complex algorithm. The algorithm is fully described and explained in the sub-directory called [illustration](Collage_generator/illustration).

2. [Vignettes Data](data): contains vignettes used for generating synthetic whole slide images.

3. [COCO-Format Converter](format_converter): the module for generating instance segmentation datasets from 
collages using COCO-compatible format.

4. [Core ML Components](ml_core): the module storing essential functions and tools for training and serving UNet models for segmentation.
    * [preprocessing](ml_core/preprocessing): contains functions for the preprocessing pipeline, namely cropping images as patches, saving patches as HDF5 files 
    and loading data as PyTorch Datasets with augmentations. 
    * [modeling](ml_core/modeling): contains UNet model architecture, which is wrapped as a PyTorch Lightning model. Also, essential functions
    for postprocessing are also provided.
    * [utils](ml_core/utils): contains essential utility functions for manipulating slides and annotations.
    * [api](ml_core/api.py): high level APIs exposed for the model serving component.
    * [config](ml_core/label_info.ini): a configuration file denoting target classes and parameters for the segmentation task.
    
5. [Scripts](ml_core): contains useful scripts for tuning (using [Optuna](https://github.com/optuna/optuna/)) and testing
models. Can also be used as a reference for calling low-level functions.

6. [Demo Notebooks](notebooks): contains several useful demo notebooks showing
the usage of core components. 

