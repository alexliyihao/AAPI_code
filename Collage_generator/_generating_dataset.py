import torch
import numpy as np
from torchvision import transforms

"""
a torch dataset completely based on a collage_generator, deprecated, just saved for references for now
"""

class _generating_dataset():
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
