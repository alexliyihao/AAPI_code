from typing import Union, List
from pathlib import Path
import configparser

import tables
from PIL import Image
import cv2
import numpy as np
from numpy import ndarray
from sklearn.feature_extraction.image import extract_patches_2d, extract_patches


class Extractor:
    """
    A generic extractor class with size-specific settings
    and unified interface for extracting patches from images.
    Can be used to extract patches from both images and masks.

    """

    def __init__(self,
                 resize: float = 0.125,
                 mirror_pad_size: int = 128,
                 patch_size: int = 256,
                 stride_size: int = 64,
                 normalize_mask: bool = False,
                 config_section_name: str = None):
        """
        Configure the extractor with specific size arguments;

        Parameters
        ----------
        resize : float
            resize factor of original slide images, e.g.,
            in order to get a desired 5x magnification patches
            from 40x magnification slides, the resize factor should
            be 0.125, since (0.125) * 40 = 5.
        mirror_pad_size : int
            size of padding regions in front of or behind the first two axis
            mirror/reflecting padding is used here
        patch_size : int
            height and width of a patch;
            for now only square patches are extracted
        stride_size : int
            stride used in patches extraction
        normalize_mask : bool
            reshape the mask to add the third channel and rescale our labels to
            continuous integers, i.e, from 0 to len(unique_labels)-1.
        config_section_name : str
            section name in config file "extractor_param.ini";
            if not None (default value), it will be used to initialize the extractor,
            ignoring all the above input arguments
        """
        if config_section_name is None:
            self.resize = resize
            self.mirror_pad_size = mirror_pad_size
            self.patch_size = patch_size
            self.stride_size = stride_size
            self.normalize_mask = normalize_mask

        else:
            config = configparser.ConfigParser()
            config.read(Path(__file__).parent / "extractor_param.ini")

            assert config_section_name in config, f"{config_section_name} is not a valid section name.\n" \
                                                  f"Valid sections: {config.sections()}"
            section = config[config_section_name]

            self.resize = section.getfloat("resize")
            self.mirror_pad_size = section.getint("mirror_pad_size")
            self.patch_size = section.getint("patch_size")
            self.stride_size = section.getint("stride_size")
            self.normalize_mask = section.getboolean("normalize_mask")

    def extract_patches(self, img, interp_method=Image.BICUBIC) -> ndarray:
        """
        Interface for extracting patches from an image after resizing

        Parameters
        ----------
        img : opencv-format image
            image to extract patches from, loaded with opencv
        interp_method : callable, PIL.Image.BICUBIC by default
            a function specifies the interpolation method for resizing,
            can only be chosen from PIL.Image.Filters.
            See https://pillow.readthedocs.io/en/stable/handbook/concepts.html#filters

        Returns
        -------
        img_patches: np.array with shape (ntile, patch_size, patch_size, 3)
            extracted patches from input image, without any filtering;
            filtering is delegated to the caller function
        """
        img = cv2.resize(img,
                         (0, 0),
                         fx=self.resize,
                         fy=self.resize,
                         interpolation=interp_method)  # resize it as specified above

        # apply mirror padding on front and end of first two axis
        # to make sure the border pixels are all preserved
        # TODO: make sure the mirror_pad_size is large enough to cover every original pixel
        pad_front_end = (self.mirror_pad_size, self.mirror_pad_size)
        img = np.pad(img,
                     pad_width=[pad_front_end, pad_front_end, (0, 0)],
                     mode="reflect")

        # convert input image into overlapping tiles,
        # size is ntiler * ntilec * 1 * patch_size x patch_size x 3
        # TODO: fix the deprecation warnings
        img_patches: np.ndarray = extract_patches(img,
                                                  (self.patch_size, self.patch_size, 3),
                                                  self.stride_size)
        # reshape size to ntile * patch_size x patch_size x 3
        img_patches = img_patches.reshape((-1, self.patch_size, self.patch_size, 3))  # type: ndarray

        return img_patches


def extract_img_patches(img_path: Union[Path, str],
                        extractor: Extractor):
    """
    Helper function for extracting patches from raw images
    with a configured extractor instance.
    Filtering is applied to patches, only keeping patches
    with tissues in it.

    Parameters
    ----------
    img_path : str or Path
        path to the image file
    extractor: Extractor
        an instance of Extractor class with parameter specified;
        caller function should use the same one with
        the one used in the "extract_mask_patches" function

    Returns
    -------
    img_patches: np.array
        Extracted patches after applying filtering strategy
    keep_indices: List[int]
        indices for patches satisfying filtering criterion


    """
    img_path = str(img_path)
    # change the order of channels from cv2 format to numpy format
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

    img_patches = extractor.extract_patches(img)

    # only keep patches with tissue in it;
    # indices for valid patches are returned
    # TODO: determine a more robust way of tissue detection;
    # currently, keep all patches whose mean(R_channel) <= 220

    keep_indices = filter(lambda patch_id: np.mean(img_patches[patch_id, ..., 0]) <= 220,
                          range(len(img_patches)))
    keep_indices = list(keep_indices)

    return img_patches, keep_indices


def extract_mask_patches(mask_path: Union[Path, str],
                         extractor: Extractor):
    """
    Helper function for extracting patches from masks
    with a configured extractor instance.
    Filtering is applied to patches, only keeping positive patches
    with annotations or negative patches passing the random sampling

    Parameters
    ----------
    mask_path : Union[Path, str]
        path to the mask file
    extractor: Extractor
        an instance of Extractor class with parameter specified;
        caller function should use the same one with
        the one used in the "extract_image_patches" function

    Returns
    -------

    """
    # make sure that input mask has three channels
    # and labels for every class is represented as a uint8 label
    mask_path = str(mask_path)
    mask = cv2.imread(mask_path)  # load as uint8 labels
    unique_labels = np.unique(np.array(mask))

    if extractor.normalize_mask:
        # rescale labels
        if np.max(unique_labels) != len(unique_labels) - 1:
            label_mappings = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
            for old_label in label_mappings:
                new_label = label_mappings[old_label]
                if old_label != new_label:
                    mask[mask == old_label] = new_label

        # reshape mask to 3D
        if len(mask.shape) == 2:
            mask = np.repeat(mask[:, np.newaxis], 3, axis=2)

    assert mask.dtype == np.uint8, \
        f"Only supports uint8 labels, but the current dtype for mask {mask_path} is {mask.dtype}"

    assert len(mask.shape) == 3 and mask.shape[-1] == 3, \
        f"Mask must also have three channels. Current mask {mask_path} has shape {mask.shape}."

    assert mask.max() == len(unique_labels) - 1, \
        f"Labels are not continuous integer from 0 to {len(unique_labels) - 1}; Max: {mask.max()}"

    # want to use nearest;
    # otherwise resizing may cause non-existing classes
    # to be produced via interpolation (e.g., ".25")
    mask_patches = extractor.extract_patches(mask, interp_method=Image.NEAREST)

    # keep all positive patches and sample negative patches;
    # TODO: determine a more intuitive sampling strategy;
    # currently, a normal distribution is used, selecting N(0,1) > 0.8

    random_flags = np.random.rand(len(mask_patches))
    keep_indices = filter(lambda patch_id: np.any(mask_patches[patch_id, ..., 0])
                                           or random_flags[patch_id] > 0.8,
                          range(len(mask_patches)))
    keep_indices = list(keep_indices)

    return mask_patches, keep_indices


def crop_and_save_patches_to_hdf5(hdf5_dataset_fname, images, masks, extractor: Extractor):
    """
    Crop images and masks and save all extracted patches to a hdf5 file.
    The resulting hdf5 file has three main columns, namely "src_image_fname",
    "img" and "mask". Can load each of them by using the syntax of
    <h5_file>.root.<col_name>[index, ...], and it will return a numpy array.

    Parameters
    ----------
    hdf5_dataset_fname : str or Path
        target location to create and write the hdf5 file
    images : List[str] or List[Path]
        list of paths to image files
    masks : List[str] or List[Path]
        list of paths to mask files;
        note that the mask can contain multiple classes,
        every unique non-zero value stands for a unique class.
        Currently the mapping is {tubules:1, artery:2, glomerulus:3, arteriole:4}
    extractor : Extractor
        instance of Extractor class, containing extraction parameters

    Returns
    -------
    None, but with the side effect of writing a hdf5 file to the target location
    """

    img_dtype = tables.UInt8Atom()
    filename_dtype = tables.StringAtom(itemsize=255)
    img_shape = (extractor.patch_size, extractor.patch_size, 3)
    mask_shape = (extractor.patch_size, extractor.patch_size) # mask is just a 2D matrix

    with tables.open_file(hdf5_dataset_fname, mode='w') as hdf5_file:

        # use blosc compression
        filters = tables.Filters(complevel=1, complib='blosc')

        # filenames, images, masks are saved as three separate
        # earray in the hdf5 file tree

        src_img_fnames = hdf5_file.create_earray(
            hdf5_file.root,
            name="src_image_fname",
            atom=filename_dtype,
            shape=(0, ))

        img_array = hdf5_file.create_earray(
            hdf5_file.root,
            name="img",
            atom=img_dtype,
            shape=(0, *img_shape),
            chunkshape=(1, *img_shape),
            filters=filters)

        mask_array = hdf5_file.create_earray(
            hdf5_file.root,
            name="mask",
            atom=img_dtype,
            shape=(0, *mask_shape),
            chunkshape=(1, *mask_shape),
            filters=filters)

        for img_path, mask_path in zip(images, masks):
            # append newly created patches for every pair image and mask
            # and flush them incrementally to the hdf5 file

            img_patches, img_keep_indices = extract_img_patches(img_path, extractor)
            mask_patches, mask_keep_indices = extract_mask_patches(mask_path, extractor)

            # take intersection of both indices for images and masks
            keep_indices = np.intersect1d(img_keep_indices, mask_keep_indices)

            img_array.append(img_patches[keep_indices, ...])
            mask_array.append(mask_patches[keep_indices, ..., 0].squeeze())

            src_img_fnames.append([img_path] * len(img_array))
