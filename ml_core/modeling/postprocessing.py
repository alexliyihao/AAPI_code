import configparser

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

from ml_core.preprocessing.patches_extraction import extract_img_patches, Extractor
from ml_core.utils.annotations import mask_to_annotation
from ml_core.modeling.unet import UNet
from ml_core.utils.slide_utils import read_full_slide_by_level


"""
========= Morphological Postprocessing on Masks ====================
"""


def enhance_heatmap(heatmap):
    # Remove Hole or noise through the use of opening, closing in Morphology module
    kernel = np.ones((7, 7), np.uint8)

    # remove noise (small bright regions) in background
    enhanced_heatmap = cv2.morphologyEx(np.array(heatmap), cv2.MORPH_OPEN, kernel, iterations=3)

    return enhanced_heatmap


"""
========= Produce Inferenced Masks from Trained Models ==========
"""

def construct_inference_dataloader(ROI, extractor, batch_size):
    """
    Create PyTorch dataloader using image in memory

    Parameters
    ----------
    ROI
    extractor
    batch_size

    Returns
    -------

    """
    patches, indices = extract_img_patches(img=ROI, extractor=extractor)  # don't apply any filtering
    patches_tensor = torch.stack(list(map(lambda img: torchvision.transforms.ToTensor()(img), patches)))
    dataset = TensorDataset(patches_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=True)
    return dataloader


def load_model_from_checkpoint(ckpt_path, model_class: LightningModule = UNet):
    return model_class.load_from_checkpoint(ckpt_path) if Path(ckpt_path).exists() else None


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


def generate_heatmap(image, output, extractor: Extractor, threshold=0.5):
    width, height = image.size
    resized_width, resized_height = int(width * extractor.resize), int(width * extractor.resize)

    def get_edge_count(length):
        return (2 * extractor.mirror_pad_size + length - extractor.patch_size) // extractor.stride_size + 1

    row_count = get_edge_count(resized_height)
    col_count = get_edge_count(resized_width)
    assert row_count * col_count == len(output), "Patch count changes after unraveling."

    row_count = int(np.sqrt(len(output)))
    col_count = row_count
    output_indices = np.unravel_index(range(len(output)), (row_count, col_count), "F")  # (rows, cols)
    output_indices = np.column_stack(output_indices)  # (row, col) pairs
    output_indices = output_indices * extractor.stride_size - extractor.mirror_pad_size

    heatmap = Image.new("L", (resized_width, resized_height))

    for output_prob, (x, y) in zip(output, output_indices):
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
        section["model_path"] = str(Path(config[section_name]["model_root"]) / f"{section_name}_best_model.ckpt")
        section["label_name"] = section_name
        section["extractor_config_name"] = f"AAPI_{section_name}"
        label_info = label_info.append(dict(section), ignore_index=True)

    label_info["model"] = label_info["model_path"].apply(load_model_from_checkpoint)
    label_info["label"] = label_info["label"].apply(int)
    label_info["min_size"] = label_info["min_size"].apply(int)
    label_info["threshold"] = label_info["threshold"].apply(float)
    label_info = label_info.dropna()
    return label_info


def predict_on_ROI(ROIs,
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

    for ROI, upper_left in ROI_iter:
        # infer every ROI
        # coordinate_pattern = re.compile(r".*ROI_(\(\d+,[ ]?\d+\)).*")
        # match = re.match(coordinate_pattern, str(ROI_path))

        # upper_left = eval(match.group(1))
        # assert type(upper_left) == tuple, f"Match Failed with {upper_left}."

        # agg_heatmap = np.zeros((*ROI.size, len(label_info) + 1)) # add a layer for background
        # layer_mapping = [0] + label_info["label"].to_list()

        annotation = []
        predicted_mask = []

        # predict for every class
        for layer_index, row in enumerate(label_info.itertuples()):
            extractor = Extractor(config_section_name=row.extractor_config_name)
            # resize factor in extractor is based on level 0
            # to accommodate input level, should times the level zoom factor (4 as default)
            extractor.resize = extractor.resize * level_base ** (input_ROI_level)

            model = row.model
            model.freeze()

            dataloader = construct_inference_dataloader(ROI, extractor, batch_size)
            output = predict_with_model(model, dataloader)
            heatmap = generate_heatmap(ROI, output, extractor, row.threshold)
            heatmap = enhance_heatmap(heatmap)
            mask = np.zeros_like(heatmap)
            mask[heatmap != 0] = row.label

            min_area = row.min_size // (level_base ** input_ROI_level)
            annotation.extend(mask_to_annotation(mask, label_info, upper_left, level=0, min_area=min_area))
            predicted_mask.append(mask)

        # mask = np.argmax(agg_heatmap, axis=2).squeeze()
        # for layer, label in enumerate(layer_mapping):
        #     if layer != 0 and label != 0:
        #         mask[mask == layer] = label
        # annotation = mask_to_annotation(mask, label_info, upper_left, level=0, min_area=10)

        annotations.append(annotation)
        predicted_masks.append(predicted_mask)

    return predicted_masks, annotations


def predict_on_WSI(slide_path,
                   label_info,
                   batch_size=64,
                   verbose=False):

    whole_slide_ROI = read_full_slide_by_level(slide_path, 1)
    return predict_on_ROI([whole_slide_ROI], [(0, 0)], label_info,
                          batch_size=batch_size,
                          verbose=verbose,
                          input_ROI_level=1)
