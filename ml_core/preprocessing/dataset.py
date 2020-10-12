import torch
import torchvision

from matplotlib import pyplot as plt
import numpy as np
import tables
import cv2


class Dataset(torch.utils.data.Dataset):
    """
    Create a custom dataset object compatible with torch dataloader
    """
    def __init__(self,
                 hdf5_fpath,
                 aug_transform=None,
                 use_edge_mask=False):
        """

        Parameters
        ----------
        hdf5_fpath : str or Path
            location to load the hdf5 file
        aug_transform : Callable(image, mask)
            a series of transformation function from albumentations package,
            Input for transformations are numpy arrays and outputs are still numpy
            arrays, so don't add any data type casting in the transformation, e.g.,
            toTensor(), as they will be handled internally by __getitem__ method.

            Also albumentations package guarantees that the mask and image will
            receive the same spatial augmentations, and color augmentations will
            only be applied to images.

        use_edge_mask : bool
            True indicates creating a special mask for edges of annotations,
            so that the loss function can take special care of these regions

        """

        self.hdf5_fpath = hdf5_fpath
        self.aug_transform = aug_transform
        self.use_edge_mask = use_edge_mask

        with tables.open_file(self.hdf5_fpath, 'r') as hdf5_file:
            self.length = hdf5_file.root.img.shape[0]

    def __getitem__(self, index):
        # opening should be done in __init__ but seems to be
        # an issue with multithreading so doing here
        with tables.open_file(self.hdf5_fpath, 'r') as hdf5_file:
            img = hdf5_file.root.img[index, ...]
            mask = hdf5_file.root.mask[index, ...]

        if self.aug_transform is not None:
            # apply augmentation transformation here
            augmented = self.aug_transform(image=img, mask=mask)
            img, mask = augmented["image"], augmented["mask"]

        # the original UNet paper assigns increased weights to the edges of the annotated objects
        # their method is more sophisticated, but this one is faster, we simply dilate the mask and
        # highlight all the pixels which were "added"
        # TODO: see original UNet paper to find out new ways of adding edge weights
        if self.use_edge_mask:
            binary_mask = np.array((mask != 0) * 255, dtype=np.uint8)
            edge_mask = cv2.Canny(binary_mask, 100, 200)
        else:  # otherwise the edge weight is all ones and thus has no affect
            edge_mask = np.ones(mask.shape, dtype=mask.dtype)

        # transform to torch.tensor
        img = torchvision.transforms.ToTensor()(img)
        mask = torch.tensor(mask, dtype=torch.long)
        edge_mask = torch.tensor(edge_mask, dtype=torch.float32)
        return img, mask, edge_mask

    def __len__(self):
        return self.length


def visualize_one_sample(dataset: Dataset, index: int, positive_only=False):
    assert index >= 0, "Index must be a non-negative number."
    if positive_only:
        cnt = index
        found = False

        for i, sample in enumerate(dataset):
            img, mask, edge_mask = sample
            if np.any(mask.numpy()) and cnt <= 0:
                print(f"Sample {i} has the {index}-th positive mask.")
                found = True
                break
            cnt -= 1

        if not found:
            sample = dataset[-1]
            print("[Warning] Fail to retrieve a sample with positive mask; "
                  "use the last sample instead.")

    else:
        sample = dataset[index]

    img, mask, edge_mask = list(map(lambda x: x.numpy(), sample))
    fig, axes = plt.subplots(1, 3)

    if img.shape[-1] != 3:
        # change from CHW to HWC
        img = np.transpose(img, (1, 2, 0))

    axes[0].imshow(img.reshape(*img.shape[:2], img.shape[2]))
    axes[0].set_title("image")
    axes[1].imshow(mask)
    axes[1].set_title("mask")
    axes[2].imshow(edge_mask)
    axes[2].set_title("edge_mask")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.close()
    return fig
