import sys
sys.path.append("../")

import argparse
import configparser
from pathlib import Path
from functools import partial

from pytorch_lightning import Trainer
from PIL import Image
import numpy as np
from tqdm import tqdm
import pandas as pd
import cv2
from sklearn.metrics import f1_score, precision_score, recall_score

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg

from ml_core.preprocessing.patches_extraction import Extractor
from ml_core.preprocessing.dataset import create_dataloader
from ml_core.modeling.unet import UNet
from ml_core.modeling.postprocessing import construct_inference_dataloader_from_ROI, predict_on_single_ROI
from ml_core.utils.slide_utils import get_biopsy_mask
from format_converter import format_converter


CLASS_NAMES = ("Glomerulus", "Artery", "Tubules", "Arteriole")


def test_models(args):
    config = configparser.ConfigParser()
    config.read(args.model_config_path)

    is_unet = True
    if args.use_collage:
        section_name = "Collage"
    elif args.use_multistain:
        section_name = "MultiStain"
    elif args.use_maskrcnn:
        section_name = "MaskRCNN"
        is_unet = False
    else:
        raise RuntimeError(f"None of '--use_collage', '--use_multistain' or '--use_maskrcnn' is specified in args.")

    section = config[section_name]
    args.dataset_name = section_name

    for class_name in args.class_names:
        print("="*25)
        print(f"Start testing class: {class_name}")
        print("=" * 25)

        if is_unet:
            # model loading for UNet models
            version = section.get(class_name)
            ckpt_dir = (args.log_dir /
                        Path(f"{section_name}/{class_name}/lightning_logs/{version}/checkpoints"))
            try:
                model_path = next(ckpt_dir.glob("*.ckpt"))
            except StopIteration:
                print(f"[Warning] No saved checkpoints found at {ckpt_dir}.")
                continue

            model = UNet.load_from_checkpoint(str(model_path))
            gpu_cnt = 1 if args.use_gpu else 0
            trainer = Trainer(gpus=gpu_cnt, default_root_dir=None)
            pred_closure = unet_pred_closure(model)

            if args.test_mode in [0, 2]:
                patch_level_tests(model, trainer, class_name, args)

        else:
            ckpt_path = args.log_dir / section_name / section.get("CKPT_REL_PATH")
            pred_closure = maskrcnn_pred_closure(ckpt_path)

        if args.test_mode in [1, 2]:
            roi_level_tests(pred_closure, class_name, args)

        print("=" * 25 + "\n")


def patch_level_tests(model, trainer, class_name, args):
    test_fname = args.data_root / Path(f"AAPI/hdf5_data/"
                                       f"patch_{args.patch_size}/{class_name}_test.h5")
    test_dataloader = create_dataloader(test_fname, batch_size=32, shuffle=False, num_workers=0)
    metrics = trainer.test(model, test_dataloader)
    metrics = pd.DataFrame(metrics)

    save_dir = args.output_dir / Path(f"{args.dataset_name}/")
    if not save_dir.exists():
        save_dir.mkdir(parents=True)

    metrics.to_csv(save_dir / f"{class_name}_patch_report.csv")
    del test_dataloader
    return metrics


def unet_pred_closure(model):
    def unet_predict_roi(class_name, roi):
        extractor = Extractor(config_section_name=f"AAPI_{class_name}")
        data = construct_inference_dataloader_from_ROI(roi, extractor, 32)
        heatmap = predict_on_single_ROI(model, data, roi.size, extractor, 0.5)
        biopsy_mask = get_biopsy_mask(roi)
        heatmap = cv2.bitwise_and(heatmap, heatmap, mask=biopsy_mask)
        del data
        return heatmap

    return unet_predict_roi


def maskrcnn_pred_closure(ckpt_path):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 32  # Check this, default 512
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 6
    cfg.MODEL.WEIGHTS = str(ckpt_path.resolve())
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2  # set a custom testing threshold

    predictor = DefaultPredictor(cfg)
    fc = format_converter(root_path="/tmp")

    def maskrcnn_predict_roi(class_name, roi):
        outputs = predictor(np.array(roi))
        instances = outputs["instances"]
        mask = fc.parse_np_masks(pred_instance=instances)
        class_label = CLASS_NAMES.index(class_name) + 1
        mask[mask != class_label] = 0
        mask[mask == class_label] = 255
        return mask

    return maskrcnn_predict_roi


def roi_level_tests(model_pred_closure, class_name, args):
    """

    Parameters
    ----------
    model_pred_closure: callable
        input: class_name, ROI
        output: predicted heatmap
    class_name
    args

    Returns
    -------

    """
    roi_data_root = args.data_root / Path(f"AAPI/ROI_data/test/{class_name}/")
    image_paths = list(filter(lambda path: "mask" not in str(path.name), roi_data_root.glob("*.png")))
    rois = []
    masks = []

    heatmap_cache_dir = args.output_dir / Path(f"{args.dataset_name}/{class_name}/")
    if not heatmap_cache_dir.exists():
        heatmap_cache_dir.mkdir(parents=True)

    for path in image_paths:
        mask_path = path.parent / Path(path.name.replace(".png", "_mask.png"))
        if mask_path.exists():
            rois.append(Image.open(path))
            masks.append(Image.open(mask_path))

    if len(rois) != len(image_paths):
        raise RuntimeError(f"Only {len(rois)}/{len(image_paths)} ROIs have masks.")

    report_dict = {
        "target": [],
        "f1_score": [],
        "precision": [],
        "recall": []
    }

    def apply_flatten_metric(metric, bin_heatmap, bin_mask):
        res = metric(y_pred=bin_heatmap.reshape(-1),
                     y_true=bin_mask.reshape(-1),
                     pos_label=True)
        return float(f"{res:.5f}")

    pbar = tqdm(zip(rois, masks))
    for roi, mask in pbar:
        mask_2d = np.reshape(mask, (*reversed(mask.size), -1))[..., 0]
        binary_mask = np.asarray(mask_2d, dtype=np.bool)

        heatmap = model_pred_closure(class_name, roi)
        binary_heatmap = np.asarray(heatmap, dtype=np.bool)

        report_dict["target"].append(Path(roi.filename).name)
        report_dict["f1_score"].append(apply_flatten_metric(f1_score, binary_heatmap, binary_mask))
        report_dict["precision"].append(apply_flatten_metric(precision_score, binary_heatmap, binary_mask))
        report_dict["recall"].append(apply_flatten_metric(recall_score, binary_heatmap, binary_mask))

        pbar.set_description(f"Avg F1: {np.mean(report_dict['f1_score']):.5f}")
        Image.fromarray(heatmap).save(heatmap_cache_dir / Path(roi.filename).name)

    report = pd.DataFrame(report_dict)
    report = report.append(report.describe().loc[["mean", "50%"]])
    report.to_csv(heatmap_cache_dir / Path(f"../{class_name}_roi_report.csv"))
    return report


def parse_arguments(input_args=None):
    parser = argparse.ArgumentParser()

    # positional arguments
    parser.add_argument("data_root",
                        help="root directory containing data; relative to the script")
    parser.add_argument("log_dir",
                        help="root directory saving logs and model checkpoints;"
                             "relative to the script;"
                             "dataset_name and class_name will be appended to it.")
    parser.add_argument("output_dir",
                        help="root directory for saving testing results;"
                             "relative to the script;"
                             "dataset_name and class_name will be appended to it.")
    parser.add_argument("model_config_path",
                        help="the config .ini file containing model versions.")

    # optional arguments: dataset settings
    dataset_name_group = parser.add_mutually_exclusive_group()
    dataset_name_group.add_argument("--use_collage",
                                    action="store_true",
                                    help="test UNet models trained on collage data,"
                                         "false as default;")
    dataset_name_group.add_argument("--use_multistain",
                                    action="store_true",
                                    help="test UNet models trained on multistain data,"
                                         "false as default;")
    dataset_name_group.add_argument("--use_maskrcnn",
                                    action="store_true",
                                    help="test maskrcnn models trained on collage data,"
                                         "false as default")

    parser.add_argument("--patch_size", default=256, type=int,
                        help='size of cropped patches')
    parser.add_argument("--use_gpu",
                        action="store_true",
                        help="Use GPU for inference.")
    parser.add_argument("--test_mode",
                        default=1,
                        type=int,
                        choices=[0,1,2],
                        help="0: patch level testings [Only available for UNet];"
                             "1: ROI level testings;"
                             "2: both patch and ROI level testings")

    parser.add_argument("--class_names",
                        nargs="*",
                        help="Target classes for testing;"
                             "separated by space;"
                             "Use all four classes by default.")

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    args.data_root = (Path.cwd() / Path(args.data_root)).resolve()
    args.output_dir = (Path.cwd() / Path(args.output_dir)).resolve()
    args.log_dir = (Path.cwd() / Path(args.log_dir)).resolve()
    args.model_config_path = (Path.cwd() / args.model_config_path).resolve()
    if not args.class_names:
        args.class_names = CLASS_NAMES

    return args


if __name__ == "__main__":
    args = parse_arguments()
    test_models(args)