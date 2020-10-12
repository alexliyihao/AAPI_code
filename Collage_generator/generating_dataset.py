import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from .collage_generator import collage_generator

"""
a torch dataset completely based on a collage_generator, just saved for references for now
"""

class generating_dataset(Dataset):
    """
    a supervised dataset specifically designed for this task
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

    def _mask_to_binary(self, mask):
        """
        convert a integer mask to binary form
        """
        return np.stack([np.where(mask == i, 1,0) for i in range(len(self._col_gen.label_dict)+1)])

    def _add_image(self):
        _collage, _mask= self._col_gen.generate(ratio_list = self._vignettes_ratio_list,
                                                background_color = \
                                                np.random.binomial(size=1,
                                                                  n=1,
                                                                  p=self._background_color_ratio)[0],
                                                return_dict = False
                                                )
        #torch channel first [0,1] tensor
        _collage = self._image_transformation(_collage).type(torch.float)
        # load all the collages as
        if self._collage_data == None:
          self._collage_data = torch.unsqueeze(_collage, dim = 0)
        else:
          self._collage_data = torch.cat([self._collage_data, torch.unsqueeze(_collage, dim = 0)])

        #torch channel first binary mask
        _mask = torch.tensor(_mask, dtype = torch.long)#self._mask_to_binary(_mask))
        # load all the collages as
        if self._mask_data == None:
          self._mask_data = torch.unsqueeze(_mask, dim = 0)
        else:
          self._mask_data = torch.cat([self._mask_data, torch.unsqueeze(_mask, dim = 0)])

    def add_addition_image(self, length):
        self._dataset_size += length
        for ctr in tqdm(range(length), desc = "adding additional images"):
            self._add_image()
