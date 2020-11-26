import PIL.Image as Img
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from typing import List, Dict, Tuple
from PIL import ImageFilter
import tables
import time
import os
import albumentations as A
import cv2
from torch.utils.data import Dataset

from ._augmentation import _augmentation
from ._functional import _functional
from ._generative import _generative
from ._insertion import _insertion
from ._utils import _utils
from ._generating_dataset import _generating_dataset

class collage_generator(_augmentation, _functional, _generative, _insertion, _utils):

    def __init__(self,
                 label_list: List[str] = ["class_one", "class_two"],
                 canvas_size: Tuple[int,int] = (3000,3000),
                 cluster_size: Tuple[int,int] = (1800,1800),
                 example_image: str = "",
                 patience: int = 100,
                 gaussian_noise_constant: float = 5.0,
                 scanning_constant: int = 25):
        """
        the initiator
        args:
          label_list:
            list of string, the list of label names
          canvas_size:
            tuple of int length 2, the 2d size of the collage
          cluster_size:
            tuple of int length 2, the 2d size of the cluster
          example_image:
            str/np.ndarray/PIL.imageobject, example_image for background
          patience:
            int, the retry time for each insert if overlap
          gaussian_noise_constant:
            float, define the strength of gaussian noise onto the background
          scanning_constant:
            int, the relocation step length for overlapping
        """
        super(collage_generator, self).__init__()

        self.label_list = label_list
        self.label_dict = dict(zip(label_list, range(1, len(label_list)+1)))
        self.image_list = [[] for i in range(len(label_list))]
        self.canvas_size = canvas_size
        self.patience = patience
        self.gaussian_noise_constant = gaussian_noise_constant
        self.example_img = self._unify_image_format(example_image)
        self.scanning_constant = scanning_constant
        self.max_component_size = np.array([0,0])
        self.cluster_size = cluster_size

        @property
        def patience(self):
            """
            makes the canvas_size can be access from .patience
            """
            return self._patience

        @patience.setter
        def patience(self, patience: int):
            """
            enforce the update of patience some legal value, the update procedure is
            still from collage_generator.patience = n
            """
            assert type(patience) == int
            assert patience > 0
            self._patience = patience

        @property
        def canvas_size(self):
            """
            makes the canvas_size can be access from .canvas_size
            """
            return self._canvas_size

        @canvas_size.setter
        def canvas_size(self, canvas_size: Tuple[int,int]):
            """
            enforce the update of canvas size some legal value, the update still from
            collage_generator.canvas_size = (x,y)
            """
            assert len(canvas_size) == 2
            assert canvas_size[0] > 0
            assert canvas_size[1] > 0
            self._canvas_size = (canvas_size[1], canvas_size[0])

        @property
        def cluster_size(self):
            """
            makes the canvas_size can be access from .canvas_size
            """
            return self._cluster_size

        @cluster_size.setter
        def cluster_size(self, cluster_size: Tuple[int,int]):
            """
            enforce the update of canvas size some legal value, the update still from
            collage_generator.cluster_size = (x,y)
            """
            assert len(cluster_size) == 2
            assert cluster_size[0] > 0
            assert cluster_size[1] > 0
            self._cluster_size = (cluster_size[1], cluster_size[0])

        @property
        def scanning_constant(self):
            """
            makes the canvas_size can be access from .scanning_constant
            """
            return self._scanning_constant

        @scanning_constant.setter
        def scanning_constant(self, scanning_constant: int):
            """
            enforce the update of patience some legal value, the update procedure is
            still from collage_generator.scanning_constant = n
            """
            assert type(scanning_constant) == int
            assert scanning_constant > 0
            self._scanning_constant = scanning_constant

class generating_dataset(Dataset, _generating_dataset):
    """
    a supervised dataset specifically designed for this task, deprecated
    """
    def __init__(self,
                 collage_generator,
                 dataset_size,
                 vignettes_ratio_list = [0.8,0.07,0.06,0.07],
                 background_color_ratio = 1.0):
        """
        the init of dataset:
        """
        super(generating_dataset, self).__init__()
        self._image_transformation = transforms.ToTensor()
        # load the collage_generator inside the dataset
        self._col_gen = collage_generator
        self._dataset_size = dataset_size
        self._vignettes_ratio_list = vignettes_ratio_list
        self._background_color_ratio = background_color_ratio
        self._collage_data = None
        self._mask_data = None

        for ctr in tqdm(range(dataset_size), desc = "generating..."):
            self._add_image()

        def __len__(self):
            return self._dataset_size

        def __getitem__(self, idx):
            return (self._collage_data[idx], self._mask_data[idx])
