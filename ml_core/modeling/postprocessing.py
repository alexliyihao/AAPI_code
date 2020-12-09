import configparser
from typing import List

from torch.utils.data import TensorDataset, DataLoader
import torch
import torchvision
from pytorch_lightning import LightningModule
from itertools import chain
from PIL import Image
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from shapely.geometry import Polygon
from shapely.affinity import affine_transform

from ..preprocessing.patches_extraction import extract_img_patches, Extractor
from ..utils.annotations import mask_to_annotation
from .unet import UNet
from ..utils.slide_utils import read_full_slide_by_level, generate_patches_coords, get_biopsy_contours, \
    get_biopsy_covered_patches_coords, get_biopsy_mask

"""
========= Morphological Postprocessing on Masks ====================
"""


def enhance_heatmap(heatmap):
    # Remove Hole or noise through the use of opening, closing in Morphology module
    kernel = np.ones((7, 7), np.uint8)

    # remove noise (small bright regions) in background
    binary_heatmap = np.where(np.array(heatmap) >= 127, 255, 0)
    binary_heatmap = np.array(binary_heatmap, dtype=np.uint8)
    enhanced_heatmap = cv2.morphologyEx(binary_heatmap,
                                        cv2.MORPH_OPEN,
                                        kernel,
                                        iterations=3)

    return enhanced_heatmap


"""
========= Produce Inferenced Masks from Trained Models ==========
"""


def construct_inference_dataloader_from_patches(patches, batch_size):
    patches_tensor = torch.stack(list(map(lambda img: torchvision.transforms.ToTensor()(img), patches)))
    dataset = TensorDataset(patches_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=True)
    return dataloader


def construct_inference_dataloader_from_ROI(ROI, extractor, batch_size, extract_foreground=False):
    if extract_foreground:
        biopsy_mask = get_biopsy_mask(ROI)
        ROI = np.array(ROI)
        ROI = cv2.bitwise_and(ROI, ROI, mask=biopsy_mask)
    patches, indices = extract_img_patches(img=ROI, extractor=extractor)
    return construct_inference_dataloader_from_patches(patches, batch_size)


def load_model_from_checkpoint(ckpt_path, model_class: LightningModule = UNet):
    model = model_class.load_from_checkpoint(ckpt_path) if Path(ckpt_path).exists() else None
    if model is not None and torch.cuda.is_available():
        model.to(torch.device("cuda"))
    return model


def predict_with_model(model: LightningModule, dataloader):
    """
    Generating predictions using model

    Parameters
    ----------
    model
    dataloader

    Returns
    -------

    """
    model_output = []
    device = model.device

    for img, *_ in dataloader:
        img_ = img.to(device) if img.device != device else img
        output = model(img_)
        output = torch.nn.functional.softmax(output, dim=1)
        model_output.append(output.detach().cpu().numpy()[:, 1, ...])

    model_output = list(chain.from_iterable(model_output))  # remove the batch axis
    model_output = np.array(model_output)

    return model_output


def generate_heatmap(size, output, extractor: Extractor, output_coords=None, threshold=0.5):
    width, height = size
    resized_width, resized_height = int(width * extractor.resize), int(height * extractor.resize)

    if output_coords is None:
        output_coords = generate_patches_coords(resized_width,
                                                resized_height,
                                                extractor.patch_size,
                                                extractor.stride_size,
                                                extractor.mirror_pad_size)

    assert len(output_coords) == len(output), f"{len(output_coords)} != {len(output)}."

    heatmap = Image.new("L", (resized_width, resized_height))

    for output_prob, (x, y) in zip(output, output_coords):
        existing_area = np.array(heatmap.crop((x, y, x + output_prob.shape[0], y + output_prob.shape[1])))
        # output_prob *= 255
        output_prob[output_prob < threshold] = 0
        output_prob *= 255

        paste_area = np.maximum(existing_area, output_prob)
        paste_area = Image.fromarray(paste_area)
        paste_coordinates = x, y
        heatmap.paste(paste_area, paste_coordinates)

    heatmap = heatmap.resize((width, height), resample=Image.NEAREST)

    return heatmap


"""
=========== Higher Level API =======================
"""


def load_label_info_from_config(config_file):
    config = configparser.ConfigParser()
    config.read(config_file)
    label_info = pd.DataFrame()
    for section_name in config.sections():
        section = config[section_name]
        model_root = Path(config_file).parent / Path(config[section_name]["model_root"])
        section["model_path"] = str(model_root / f"{section_name}_best_model.ckpt")
        section["label_name"] = section_name
        section["extractor_config_name"] = f"AAPI_{section_name}"
        label_info = label_info.append(dict(section), ignore_index=True)

    label_info["model"] = label_info["model_path"].apply(load_model_from_checkpoint)
    label_info["label"] = label_info["label"].apply(int)
    label_info["min_size"] = label_info["min_size"].apply(int)
    label_info["threshold"] = label_info["threshold"].apply(float)
    label_info = label_info.dropna()
    return label_info


def predict_on_single_ROI(model,
                          data,
                          heatmap_size,
                          extractor,
                          threshold,
                          output_coords=None):

    outputs = predict_with_model(model, data)
    heatmap: Image = generate_heatmap(heatmap_size, outputs, extractor,
                               output_coords=output_coords,
                               threshold=threshold)
    heatmap: np.ndarray = np.array(heatmap)
    binary_enhanced_heatmap = enhance_heatmap(heatmap)
    heatmap[np.where(binary_enhanced_heatmap == 0)] &= 0
    return heatmap


def predict_on_batch_ROIs(ROIs,
                          upper_left_coords,
                          label_info,
                          batch_size=64,
                          verbose=False,
                          input_ROI_level=0,
                          level_base=4):
    annotations = []
    predicted_masks = []

    ROI_iter = zip(ROIs, upper_left_coords)
    if verbose:
        ROI_iter = tqdm(ROI_iter, total=len(ROIs))

    resize_factor = (level_base ** input_ROI_level)
    label_info["min_area"] = label_info["min_size"] // resize_factor

    for ROI, upper_left in ROI_iter:
        # infer every ROI
        # coordinate_pattern = re.compile(r".*ROI_(\(\d+,[ ]?\d+\)).*")
        # match = re.match(coordinate_pattern, str(ROI_path))

        # upper_left = eval(match.group(1))
        # assert type(upper_left) == tuple, f"Match Failed with {upper_left}."
        width, height = ROI.size
        agg_heatmap = np.zeros((height, width, len(label_info) + 1)) # add a layer for background
        layer_mapping = [0] + label_info["label"].to_list()

        # annotation = []
        # predicted_mask = []

        # predict for every class
        for layer_index, row in enumerate(label_info.itertuples()):
            extractor = Extractor(config_section_name=row.extractor_config_name)
            # resize factor in extractor is based on level 0
            # to accommodate input level, should times the level zoom factor (4 as default)
            extractor.resize = extractor.resize * resize_factor

            model = row.model
            model.freeze()

            data = construct_inference_dataloader_from_ROI(ROI=ROI,
                                                                 extractor=extractor,
                                                                 batch_size=batch_size)

            heatmap = predict_on_single_ROI(model,
                                            data,
                                            heatmap_size=ROI.size,
                                            extractor=extractor,
                                            threshold=row.threshold,
                                            output_coords=None)

            agg_heatmap[..., layer_index + 1] = heatmap

            # mask = np.zeros_like(heatmap)
            # mask[heatmap != 0] = row.label

            # min_area = row.min_size // (level_base ** input_ROI_level)
            # annotation.extend(mask_to_annotation(mask, label_info, upper_left, mask_level=0, min_area=min_area))
            # predicted_mask.append(mask)

        mask = np.argmax(agg_heatmap, axis=2).squeeze()
        for layer, label in enumerate(layer_mapping):
            if layer != 0 and label != 0:
                mask[mask == layer] = label

        annotation = mask_to_annotation(mask, label_info, upper_left, mask_level=0)

        annotations.append(annotation)
        predicted_masks.append(mask)

    return predicted_masks, annotations


def predict_on_WSI(slide_path,
                   label_info,
                   batch_size=64,
                   level_base=4,
                   callback=None):

    whole_slide_level_1: Image = read_full_slide_by_level(slide_path, level=1)
    level_1_width, level_1_height = whole_slide_level_1.size
    thumbnail = whole_slide_level_1.resize((level_1_width // level_base, level_1_height // level_base))

    biopsy_contours: List[Polygon] = get_biopsy_contours(thumbnail)
    # remap to level 1 coordinates
    biopsy_contours = [affine_transform(c, [level_base, 0, 0, level_base, 0, 0]) for c in biopsy_contours]

    annotations = [[] for _ in range(len(biopsy_contours))]
    predicted_masks = [[] for _ in range(len(biopsy_contours))]

    for biopsy_id, biopsy_contour in enumerate(tqdm(biopsy_contours)):

        bbox = biopsy_contour.bounds

        maxx_level_0 = bbox[0] * level_base
        maxy_level_0 = bbox[1] * level_base

        ROI: Image = whole_slide_level_1.crop(bbox)
        width, height = ROI.size

        for layer_index, row in enumerate(label_info.itertuples()):
            extractor = Extractor(config_section_name=row.extractor_config_name)
            extractor.resize = extractor.resize * level_base  # change level from 0 to 1

            resized_width, resized_height = int(width * extractor.resize), int(height * extractor.resize)
            resized_ROI = ROI.resize((resized_width, resized_height))

            resized_contour = affine_transform(biopsy_contour, [1, 0, 0, 1, -bbox[0], -bbox[1]])
            resized_contour = affine_transform(resized_contour, [extractor.resize, 0, 0, extractor.resize, 0, 0])

            # resize level: same as resized_contour
            # origin point: upper left of resized_contour
            covered_coords = get_biopsy_covered_patches_coords(resized_contour,
                                                               resized_width,
                                                               resized_height,
                                                               extractor.patch_size,
                                                               extractor.stride_size,
                                                               0)

            patches = [resized_ROI.crop((coord[0],
                                         coord[1],
                                         coord[0] + extractor.patch_size,
                                         coord[1] + extractor.patch_size))
                       for coord in covered_coords]

            data = construct_inference_dataloader_from_patches(patches, batch_size)

            model: LightningModule = row.model
            model.freeze()

            heatmap = predict_on_single_ROI(model, data, ROI.size, extractor,
                                            threshold=row.threshold,
                                            output_coords=covered_coords)

            mask = np.zeros_like(heatmap)
            mask[heatmap != 0] = row.label

            min_area = row.min_size // level_base

            annotations[biopsy_id].extend(mask_to_annotation(mask,
                                                             label_info,
                                                             (maxx_level_0, maxy_level_0),
                                                             mask_level=1,
                                                             min_area=min_area,
                                                             level_factor=level_base))
            predicted_masks[biopsy_id].append(mask)

            del patches, resized_ROI, data

    return predicted_masks, annotations
