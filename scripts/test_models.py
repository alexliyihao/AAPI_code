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
from sklearn.metrics import f1_score, precision_score, recall_score

from ml_core.preprocessing.patches_extraction import Extractor
from ml_core.preprocessing.dataset import create_dataloader
from ml_core.modeling.unet import UNet
from ml_core.modeling.postprocessing import construct_inference_dataloader_from_ROI, predict_on_single_ROI


CLASS_NAMES = ("Glomerulus", "Artery", "Tubules", "Arteriole")


def test_models(args):
    config = configparser.ConfigParser()
    config.read(args.model_config_path)

    if args.use_collage:
        section_name = "Collage"
    elif args.use_multistain:
        section_name = "MultiStain"
    else:
        raise RuntimeError(f"Neither --use_collage nor --use_multistain is used in args.")

    section = config[section_name]
    args.dataset_name = section_name

    for class_name in args.class_names:
        print("="*25)
        print(f"Start testing class: {class_name}")
        print("=" * 25)

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

        if args.test_mode in [0, 2]:
            patch_level_tests(model, trainer, class_name, args)

        if args.test_mode in [1, 2]:
            roi_level_tests(model, class_name, args)

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


def roi_level_tests(model, class_name, args):
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

    extractor = Extractor(config_section_name=f"AAPI_{class_name}")
    pbar = tqdm(zip(rois, masks))
    for roi, mask in pbar:
        data = construct_inference_dataloader_from_ROI(roi, extractor, 32)

        mask_2d = np.reshape(mask, (*reversed(mask.size), -1))[..., 0]
        binary_mask = np.asarray(mask_2d, dtype=np.bool)

        heatmap = predict_on_single_ROI(model, data, roi.size, extractor, 0.5)
        binary_heatmap = np.asarray(heatmap, dtype=np.bool)

        report_dict["target"].append(Path(roi.filename).name)
        report_dict["f1_score"].append(apply_flatten_metric(f1_score, binary_heatmap, binary_mask))
        report_dict["precision"].append(apply_flatten_metric(precision_score, binary_heatmap, binary_mask))
        report_dict["recall"].append(apply_flatten_metric(recall_score, binary_heatmap, binary_mask))

        pbar.set_description(f"Avg F1: {np.mean(report_dict['f1_score']):.5f}")
        del data
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
                                    help="use collage data as auxiliary dataset, false if not specified;"
                                         "mutually exclusive with --use_multistain")
    dataset_name_group.add_argument("--use_multistain",
                                    action="store_true",
                                    help="use multistain data as auxiliary dataset, false if not specified;"
                                         "mutually exclusive with --use_collage")

    parser.add_argument("--patch_size", default=256, type=int,
                        help='size of cropped patches')
    parser.add_argument("--use_gpu",
                        action="store_true",
                        help="Use GPU for inference.")
    parser.add_argument("--test_mode",
                        default=0,
                        type=int,
                        choices=[0,1,2],
                        help="0: patch level testings;"
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