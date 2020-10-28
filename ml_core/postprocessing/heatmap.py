from torch.utils.data import TensorDataset, DataLoader
import torch
import torchvision
from pytorch_lightning import LightningModule
from itertools import chain
from PIL import Image
import numpy as np
import re
from tqdm import tqdm

from ..preprocessing.patches_extraction import extract_img_patches, Extractor
from ..utils.annotations import mask_to_annotation

ROI_SIZE = 3000


def construct_inference_dataloader(ROI_path, extractor, batch_size):
    patches, indices = extract_img_patches(ROI_path, extractor)
    patches_tensor = torch.stack(list(map(lambda img: torchvision.transforms.ToTensor()(img), patches)))
    dataset = TensorDataset(patches_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=True)
    return dataloader


def predict_with_model(model: LightningModule, dataloader):
    model_output = []

    for img, *_ in dataloader:
        output = model(img)
        output = torch.nn.functional.softmax(output, dim=1)
        model_output.append(output.detach().cpu().numpy()[:, 1, ...])

    model_output = list(chain.from_iterable(model_output))
    model_output = np.array(model_output)

    return model_output


def generate_heatmap_and_annotations(model_class: LightningModule,
                                     ckpt_path,
                                     ROI_paths,
                                     extractor_config_section_name,
                                     label_info,
                                     batch_size=64):
    extractor = Extractor(config_section_name=extractor_config_section_name)
    pad_size = extractor.mirror_pad_size
    stride = extractor.stride_size
    resize = extractor.resize

    model = model_class.load_from_checkpoint(ckpt_path)
    model.freeze()

    heatmap_group = []
    annotations_group = []

    for ROI_path in tqdm(ROI_paths):
        assert ROI_path.exists(), f"{ROI_path} doesn't exist."
        dataloader = construct_inference_dataloader(ROI_path, extractor, batch_size)
        model_output = predict_with_model(model, dataloader)

        output_length = len(model_output)
        one_edge_count = int(np.sqrt(output_length))
        output_indices = range(output_length)
        output_indices = np.unravel_index(output_indices, (one_edge_count, one_edge_count), "F")  # (rows, cols)
        output_indices = np.column_stack(output_indices)  # (row, col) pairs
        output_indices = output_indices * stride - pad_size

        resized_ROI_size = int(ROI_SIZE * resize)
        heatmap = Image.new("L", (resized_ROI_size, resized_ROI_size))

        for output_prob, (x, y) in zip(model_output, output_indices):
            existing_area = np.array(heatmap.crop((x, y, x + output_prob.shape[0], y + output_prob.shape[1])))
            paste_area = np.maximum(existing_area, output_prob)
            paste_area = Image.fromarray(paste_area)
            paste_coordinates = x, y
            heatmap.paste(paste_area, paste_coordinates)

        heatmap = heatmap.resize((ROI_SIZE, ROI_SIZE), resample=Image.NEAREST)

        coordinate_pattern = re.compile(r".*ROI_(\(\d+,[ ]?\d+\)).*")
        match = re.match(coordinate_pattern, str(ROI_path))

        if match:
            upper_left = eval(match.group(1))
            assert type(upper_left) == tuple, f"Match Failed with {upper_left}."
            annotations = mask_to_annotation(heatmap, label_info, upper_left, level=0)
        else:
            annotations = []

        heatmap_group.append(heatmap)
        annotations_group.append(annotations)

    return heatmap_group, annotations_group
