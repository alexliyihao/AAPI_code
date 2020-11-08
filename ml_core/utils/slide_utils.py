from openslide import open_slide
from pathlib import Path
import pandas as pd
from skimage import measure, color
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from itertools import product
from functools import reduce
from PIL import Image
from skimage.filters import threshold_otsu
from tqdm import tqdm

from .annotations import create_covering_rectangles, annotation_to_mask


LEVEL_BASE = 4
HIGHEST_LEVEL = 2


def read_slide(path):
    return open_slide(str(path)) if path is not None and Path(path).exists() else None


def get_slide_and_mask_paths(data_dir):
    data_dir = Path(data_dir)
    slide_paths = sorted(filter(lambda x: "mask" not in str(x), data_dir.glob("**/*.tif")))
    mask_paths = [data_dir / str(p.name).replace(".tif", "_mask.tif") for p in slide_paths]
    return slide_paths, mask_paths


def read_region_from_slide(slide, x, y, level, width, height,
                           relative_coordinate=True):
    if relative_coordinate:
        # x,y are relative to the current output level
        # i.e., the top left pixel in the level ${level} reference frame.
        factor = int(slide.level_downsamples[level])
        offset_x, offset_y = x * factor, y * factor
    else:
        # the top left pixel in the level 0 reference frame.
        offset_x, offset_y = x, y

    im = slide.read_region((offset_x, offset_y), level, (width, height))
    im = im.convert('RGB') # drop the alpha channel
    return im


def read_full_slide_by_level(slide_path, level):
    slide = open_slide(str(slide_path))
    return read_region_from_slide(slide, x=0, y=0, level=level,
                                  width=slide.level_dimensions[level][0],
                                  height=slide.level_dimensions[level][1])


def get_slides_meta_info(slide_paths, mask_paths=None, output_path=None):
    if mask_paths is None:
        mask_paths = [None for _ in slide_paths]
    
    max_level = max([len(open_slide(str(s)).level_dimensions) for s in slide_paths])

    meta_data = {"id": [], "has_tumor": []}
    meta_data.update({f"Level_{k}": [] for k in range(max_level)})

    for slide_path, mask_path in zip(slide_paths, mask_paths):
        slide_name = slide_path.name
        slide = open_slide(str(slide_path))
        meta_data["id"].append(slide_name)

        mask = read_slide(mask_path) if mask_path is not None else None
        has_tumor = mask is not None
        meta_data["has_tumor"].append(has_tumor)

        for i in range(max_level):
            meta = [-1, [], []] # (downsample_factor, slide_level_dims, mask_level_dims)

            if i < len(slide.level_dimensions):
                meta[:2] = slide.level_downsamples[i], slide.level_dimensions[i]

            if mask is not None:
                if i < len(mask.level_dimensions):
                    meta[-1] = mask.level_dimensions[i]
            else:
                meta[-1] = "<No Mask>"

            meta_data[f"Level_{i}"].append(meta)

    meta_info = pd.DataFrame(meta_data)
    if output_path is not None:
        meta_info.to_json(Path(output_path) / "meta_info.json")
    return meta_info


def get_connected_regions_from_tumor_slides(mask_path, verbose=False, mask_level=HIGHEST_LEVEL):
    mask = read_full_slide_by_level(mask_path, level=mask_level)
    mask = np.array(mask.getchannel(0))
    mask = mask.T # transpose to be consistent with the shape of slide
    labeled_mask = measure.label(mask, connectivity=2)
    bbox_list = []
    for region in measure.regionprops(labeled_mask):
        bbox_list.append(region.bbox)

    if verbose:
        # show the bounding box and region on the mask
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
        label_overlay = color.label2rgb(labeled_mask, image=mask)
        ax.imshow(label_overlay)

        for bbox in bbox_list:
            # draw rectangle around segmented coins
            minr, minc, maxr, maxc = bbox
            rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                      fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(rect)

        plt.show()

    return bbox_list


def check_patch_include_tissue(patch, otsu_thresh):
    # detect tissue regions having pixels with H, S, V >= otsu_threshold
    hsv_patch = patch.convert("HSV")
    channels = [np.asarray(hsv_patch.getchannel(c)) for c in ["H", "S", "V"]]
    tissue_mask = np.bitwise_and(*[c >= otsu_thresh[i] for i, c in enumerate(channels)])
    return np.sum(tissue_mask) > 0.15 * reduce(lambda a,b: a*b, tissue_mask.shape)


def crop_ROI_from_slide(slide_path, save_dir, size, stride, level, annotation_file=None):

    slide = read_slide(str(slide_path))
    save_dir.mkdir(parents=True, exist_ok=True)

    thumbnail = read_full_slide_by_level(slide_path, 2).convert("HSV")
    otsu_thresh = [threshold_otsu(np.asarray(thumbnail.getchannel(c)))
                   for c in ["H", "S", "V"]]

    level_dims = slide.level_dimensions[level]
    row_count, col_count = (level_dims[0] // stride + 1,
                            level_dims[1] // stride + 1)

    list_of_row_id_col_id = product(range(row_count), range(col_count))
    for row_id, col_id in tqdm(list_of_row_id_col_id, total=row_count*col_count):
        offset_x, offset_y = int(row_id * stride), int(col_id * stride)
        slide_patch = read_region_from_slide(slide, offset_x, offset_y, level,
                                             width=size, height=size)
        if check_patch_include_tissue(slide_patch, otsu_thresh):
            slide_patch.save(save_dir / f"ROI_{(offset_x, offset_y)}.png")


def crop_ROI_using_annotations(slide_path,
                               save_dir,
                               annotations,
                               class_name,
                               section_size,
                               verbose):
    filtered_annotations = list(filter(lambda a: a.group_name == class_name, annotations))
    if len(filtered_annotations) == 0:
        print(f"Cannot find any annotations with class {class_name}")
        return None

    upper_left_coords = create_covering_rectangles(filtered_annotations, verbose=verbose, size=section_size)

    # currently supports binary mask only
    label_info = pd.DataFrame({"label_name": [class_name],
                               "label": [1],
                               "color": "#ff0000"})

    slide = read_slide(str(slide_path))

    class_root = save_dir / class_name
    class_root.mkdir(exist_ok=True)

    for i, upper_left in enumerate(upper_left_coords):
        mask = annotation_to_mask(filtered_annotations,
                                  label_info=label_info,
                                  upper_left_coordinates=upper_left,
                                  mask_shape=(section_size, section_size),
                                  level=0)
        assert mask.max() > 0
        #         assert np.all(mask[..., 0] == mask[..., 1]) and np.all(mask[..., 0] == mask[..., 2])

        img = read_region_from_slide(slide,
                                     x=upper_left[0],
                                     y=upper_left[1],
                                     level=0,
                                     width=section_size,
                                     height=section_size,
                                     relative_coordinate=True)

        Image.fromarray(mask).save(class_root / f"HE_{i:03d}_mask_{upper_left}.png")
        img.save(class_root / f"HE_{i:03d}_{upper_left}.png")