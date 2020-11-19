import datetime
import json
import os
import glob
import numpy as np
import datetime
from pytz import timezone
import json
import math
from detectron2.structures import BoxMode
from tqdm.notebook import tqdm
from PIL import Image
from skimage import measure
from shapely.geometry import Polygon, MultiPolygon
from ._coco_format import _coco_converter
from ._detectron_parser import _detectron_parser

class format_converter(_detectron_parser):
    """
    format_converter saving COCO_PANOPTIC and detectron format
    """
    def __init__(self, root_path = "COCO"):
        super(format_converter, self).__init__()
        self.root_path = root_path
        self.image_path = os.path.join(self.root_path, "images")
        self.segmentation_path = os.path.join(self.root_path, "segmentations")
        for i in [self.root_path, self.image_path, self.segmentation_path]:
            os.makedirs(i, exist_ok=True)
