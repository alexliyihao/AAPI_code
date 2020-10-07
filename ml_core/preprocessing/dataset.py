import torch

import numpy as np
import tables
import scipy as sp


class Dataset(torch.utils.data.Dataset):
    """
    Create a custom dataset object compatible with torch dataloader
    """
    def __init__(self,
                 hdf5_fpath,
                 img_transform=None,
                 mask_transform=None,
                 use_edge_mask=False,
                 seed=42):
        """

        Parameters
        ----------
        hdf5_fpath : str or Path
            location to load the hdf5 file
        img_transform :
            a series of transformation function for images, composed by
            torchvision.transforms.Compose. Should start with ToPILImage()
            as the input is numpy array.
       mask_transform :
            similar to img_transform, but transformations are used for masks,
            so no color jitter transformation is applied.
        use_edge_mask : bool
            True indicates creating a special mask for edges of annotations,
            so that the loss function can take special care of these regions
        seed : int
            seed for making results from RNG reproducible
        """

        self.hdf5_fpath = hdf5_fpath
        self.img_transform = img_transform
        self.mask_transform = mask_transform
        self.use_edge_mask = use_edge_mask

        # set random seed for reproducibility
        # TODO: verify whether it's needed to set cudnn backend parameters
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        with tables.open_file(self.hdf5_fpath, 'r') as hdf5_file:
            self.length = hdf5_file.root.img.shape[0]

    def __getitem__(self, index):
        # opening should be done in __init__ but seems to be
        # an issue with multithreading so doing here
        with tables.open_file(self.hdf5_fpath, 'r') as hdf5_file:
            img = hdf5_file.root.img[index, ...]
            mask = hdf5_file.root.mask[index, ...]

        # the original UNet paper assignes increased weights to the edges of the annotated objects
        # their method is more sophistocated, but this one is faster, we simply dilate the mask and
        # highlight all the pixels which were "added"
        # TODO: see original UNet paper to find out new ways of adding edge weights
        if self.use_edge_mask:
            edge_mask = sp.ndimage.morphology.binary_dilation(mask == 1, iterations=2) & ~mask
        else:  # otherwise the edge weight is all ones and thus has no affect
            edge_mask = np.ones(mask.shape, dtype=mask.dtype)

        img = self.img_transform(img) if self.img_transform is not None else img
        mask = self.mask_transform(mask) if self.mask_transform is not None else mask
        mask = mask.long()  # change label to int type
        edge_mask = self.mask_transform(edge_mask) if self.mask_transform is not None else edge_mask

        return img, torch.squeeze(mask), torch.squeeze(edge_mask)

    def __len__(self):
        return self.length
