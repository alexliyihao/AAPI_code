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

from ._augmentation import _augmentation
from ._functional import _functional
from ._generative import _generative
from ._insertion import _insertion
from ._utils import _utils

class collage_generator(_augmentation, _functional, _generative, _insertion, _utils):

    def __init__(self,
                 label_list: List[str] = ["one", "two"],
                 canvas_size: Tuple[int,int] = (3000,3000),
                 cluster_size: Tuple[int,int] = (2200,2200),
                 example_image: str = "",
                 patience: int = 100,
                 gaussian_noise_constant: float = 5.0,
                 scanning_constant: int = 25,
                 num_proximal_per_cluster: int = 300,
                 num_distal_per_image: int = 3000):
        """
        the initiator of collage_generator
        Args:
          label_list:
            list of string, the list of label names
          canvas_size:
            tuple of int length 2, the 2d size of the collage
          cluster_size:
            tuple of int length 2, the 2d size of the cluster
          example_image:
            str/np.ndarray/PIL.image object, example_image for background
          patience:
            int, the retry time for each insert if overlap
          gaussian_noise_constant:
            float, define the strength of gaussian noise onto the image
          scanning_constant:
            int, the relocation step length for overlapping
        """
        super(collage_generator, self).__init__()

        self._label_list = label_list
        self._label_dict = dict(zip(label_list, range(1, len(label_list)+1)))
        self._image_list = [[] for i in range(len(label_list))]
        self._canvas_size = canvas_size
        self._patience = patience
        self._gaussian_noise_constant = gaussian_noise_constant
        self._example_img = self._unify_image_format(example_image)
        self._scanning_constant = scanning_constant
        self._max_component_size = np.array([0,0])
        self._cluster_size = cluster_size
        self._num_proximal_per_cluster = num_proximal_per_cluster
        self._num_distal_per_image = num_distal_per_image
        
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
        def gaussian_noise_constant(self):
            """
            getter of gaussian_noise_constant
            """
            return self._gaussian_noise_constant

        @gaussian_noise_constant.setter
        def gaussian_noise_constant(self, gaussian_noise_constant):
            """
            setter for gaussian_noise_constant
            """
            assert gaussian_noise_constant >=0
            self._gaussian_noise_constant = gaussian_noise_constant

        @property
        def example_img(self):
            """
            returns the example_img using to generate background
            """
            plt.imshow(self._example_img)

        @example_img.setter
        def example_img(self, img):
            """
            setter of example img, img can be string path, np.ndarray or PIL object
            """
            self._example_img = self._unify_image_format(img = img)

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

        @property
        def num_proximal_per_cluster(self):
            """
            makes the canvas_size can be access from .scanning_constant
            """
            return self._num_proximal_per_cluster

        @num_proximal_per_cluster.setter
        def num_proximal_per_cluster(self, scanning_constant: int):
            """
            enforce the update of patience some legal value, the update procedure is
            still from collage_generator.scanning_constant = n
            """
            assert type(scanning_constant) == int
            assert scanning_constant > 0
            self._num_proximal_per_cluster = num_proximal_per_cluster
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
        def label_dict(self):
            """
            The getter for label_dict
            """
            return self._label_dict
