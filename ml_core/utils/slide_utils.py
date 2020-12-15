from openslide import open_slide
from pathlib import Path
import pandas as pd
import numpy as np
from itertools import product
from functools import reduce
from PIL import Image
from skimage.filters import threshold_otsu
import cv2
from tqdm import tqdm
from shapely.geometry import Polygon, box


from .annotations import create_covering_rectangles_using_clusters, create_covering_rectangles_greedy
from .annotations import annotation_to_mask, mask_to_polygon, generate_colorful_mask


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


def check_patch_include_tissue(patch, otsu_thresh):
    # detect tissue regions having pixels with H, S, V >= otsu_threshold
    hsv_patch = patch.convert("HSV")
    channels = [np.asarray(hsv_patch.getchannel(c)) for c in ["H", "S", "V"]]
    tissue_mask = np.bitwise_and(*[c >= otsu_thresh[i] for i, c in enumerate(channels)])
    return np.sum(tissue_mask) > 0.15 * reduce(lambda a,b: a*b, tissue_mask.shape)


def crop_ROI_from_slide(slide_path, save_dir, size, stride, level,
                        apply_otsu=True,
                        fname_formatter=lambda offset_x, offset_y, *args: f"ROI_{(offset_x, offset_y)}.png",
                        overwrite_exist=False):

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
        patch_name = fname_formatter(offset_x, offset_y, size)

        if overwrite_exist or not Path(save_dir / patch_name).exists():
            # generate and write a new patch
            slide_patch = read_region_from_slide(slide, offset_x, offset_y, level,
                                                 width=size, height=size)
            if not apply_otsu or check_patch_include_tissue(slide_patch, otsu_thresh):
                slide_patch.save(save_dir / patch_name)


def generate_patches_coords(width, height, patch_size, stride, pad_size):
    def get_edge_count(length):
        return (2 * pad_size + length - patch_size) // stride + 1

    row_count = get_edge_count(height)
    col_count = get_edge_count(width)

    output_indices = np.unravel_index(range(row_count * col_count), (row_count, col_count))  # (hids, wids)
    output_indices = np.column_stack(output_indices)  # (hid, wid) pairs
    output_coords = output_indices * stride - pad_size # coord = id * stride - pad
    output_coords = np.column_stack([output_coords[:, 1], output_coords[:, 0]]) # change (y, x) to (x, y)
    return output_coords


def get_biopsy_contours(image):
    # detect biopsy regions: Otsu's thresholding after Gaussian filtering
    blur = cv2.GaussianBlur(np.array(image), (25, 25), 0)
    grayscale_blur = cv2.cvtColor(blur, cv2.COLOR_RGB2GRAY)
    ret3, th3 = cv2.threshold(grayscale_blur, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cv2.bitwise_not(th3, th3)  # invert the foreground and background
    region_polygons = mask_to_polygon(th3, min_area=10000)
    return region_polygons


def get_biopsy_covered_patches_coords(biopsy_contour, width, height, patch_size, stride, pad_size):
    # get patches coords
    all_patch_coords = generate_patches_coords(width, height, patch_size, stride, pad_size)
    patch_boxes = [box(coords[0], coords[1], coords[0] + patch_size, coords[1] + patch_size)
                   for coords in all_patch_coords]

    biopsy_covered_patches_coords = []

    for patch_box, patch_coord in zip(patch_boxes, all_patch_coords):
        if patch_box.intersects(biopsy_contour):
            biopsy_covered_patches_coords.append(patch_coord)

    return biopsy_covered_patches_coords


def crop_ROI_using_annotations(slide_path,
                               save_dir,
                               annotations,
                               class_name,
                               section_size):

    filtered_annotations = list(filter(lambda a: a.group_name == class_name, annotations))
    if len(filtered_annotations) == 0:
        print(f"Cannot find any annotations with class {class_name}")
        return None

    try:
        upper_left_coords = create_covering_rectangles_using_clusters(filtered_annotations, size=section_size)
    except RuntimeError as e:
        print(f"[Warning] {e}; use a greedy algorithm to cover all polygons instead")
        upper_left_coords = create_covering_rectangles_greedy(filtered_annotations, size=section_size)

    # currently supports binary mask only
    label_info = pd.DataFrame({"label_name": [class_name],
                               "label": [1],
                               "color": "#ffffff"})

    slide = read_slide(str(slide_path))
    slide_name = Path(slide_path).with_suffix("").name

    class_root = save_dir / class_name
    class_root.mkdir(exist_ok=True, parents=True)

    saved_paths = []

    for i, upper_left in enumerate(upper_left_coords):
        ROI_path = class_root / f"{slide_name}_ROI_{i:03d}_{upper_left}.png"
        mask_path = class_root / f"{ROI_path.with_suffix('').name}_mask.png"
        saved_paths.append((ROI_path, mask_path))

        if not mask_path.exists():
            mask = annotation_to_mask(filtered_annotations,
                                      label_info=label_info,
                                      upper_left_coordinates=upper_left,
                                      mask_shape=(section_size, section_size),
                                      level=0)
            assert mask.max() > 0
            mask = generate_colorful_mask(mask, label_info)
            Image.fromarray(mask).save(mask_path)

        if not ROI_path.exists():
            img = read_region_from_slide(slide,
                                         x=upper_left[0],
                                         y=upper_left[1],
                                         level=0,
                                         width=section_size,
                                         height=section_size,
                                         relative_coordinate=True)

            img.save(ROI_path)

    return saved_paths