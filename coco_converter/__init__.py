import datetime
import json
import os
import glob
import PIL.Image as Img
import numpy as np
import datetime
from pytz import timezone
import json
from skimage import measure
from shapely.geometry import Polygon, MultiPolygon
from ._coco_format import _coco_converter


class coco_converter(_coco_converter):
    """
    coco_converter saving COCO_PANOPTIC format
    """
    def __init__(self, root_path = "COCO"):
        super(coco_converter, self).__init__()
        self.root_path = root_path
        self.image_path = os.path.join(self.root_path, "images")
        self.segmentation_path = os.path.join(self.root_path, "segmentations")
        for i in [self.root_path, self.image_path, self.segmentation_path]:
            os.makedirs(i, exist_ok=True)
