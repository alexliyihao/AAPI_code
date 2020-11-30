from pathlib import Path
import sys
sys.path.append("../")
import argparse
from functools import partial

from ml_core.preprocessing.dataset import create_dataloader
from ml_core.modeling.unet import UNet
from ml_core.modeling.utils import get_augmentation_transforms, create_pl_trainer, get_checkpoint_callback

from torch.optim import Adam
import optuna
from optuna.integration import PyTorchLightningPruningCallback


CLASS_NAMES = ("Glomerulus", "Artery", "Tubules", "Arteriole")


def train_binary_unet_model(args, callbacks=None):

    # parse dataset settings
    class_name = args.class_name
    dataset_name = args.dataset_name
    data_root = args.data_root
    log_dir = args.log_dir
    aapi_hdf5_dir = data_root / Path(f"AAPI/hdf5_data/patch_{args.patch_size}")

    if dataset_name != "AAPI":
        aux_data_hdf5_dir = data_root / Path(f"{dataset_name}/hdf5_data/patch_{args.patch_size}")
    else:
        aux_data_hdf5_dir = None

    train_fnames = [aapi_hdf5_dir / f"{class_name}_train.h5"]
    val_fnames = [aapi_hdf5_dir / f"{class_name}_val.h5"]

    if class_name == "Tubules" or dataset_name == "Collage":
        # only use the auxiliary dataset for collage or training for Tubules class
        train_fnames = [aux_data_hdf5_dir / f"{class_name}_train.h5"]
        val_fnames = [aux_data_hdf5_dir / f"{class_name}_val.h5"]

    elif dataset_name == "MultiStain":
        train_fnames.append(aux_data_hdf5_dir / f"{class_name}_train.h5")
        val_fnames.append(aux_data_hdf5_dir / f"{class_name}_val.h5")

    list(map(check_file_existence, train_fnames + val_fnames))

    train_data = create_dataloader(train_fnames,
                                   transform=get_augmentation_transforms(),
                                   return_dataset=False)
    val_data = create_dataloader(val_fnames,
                                 batch_size=64,
                                 shuffle=False,
                                 return_dataset=False)

    # parse trainer and model settings
    hyper_params = {
        "patch_size": args.patch_size,
        "lr": args.lr,
        "epochs": args.epochs,
        "depth": args.depth,
        "wf": args.wf,
        "edge_weight": args.edge_weight
    }

    if args.dry_run:
        print("=" * 10 + "DRY RUN SUMMARY" + "=" * 10)
        print(f"Train data fnames: {train_fnames};")
        print(f"Val data fnames: {val_fnames};")
        print(f"Log directory: {log_dir};")
        print(f"Hyper params for model: {hyper_params}")

    else:
        ckpt_callback = get_checkpoint_callback(save_top_k=1)
        trainer = create_pl_trainer(use_gpu=True,
                                    root_dir=log_dir,
                                    epochs=hyper_params["epochs"],
                                    checkpoint_callback=ckpt_callback,
                                    callbacks=callbacks)

        unet = UNet(in_channels=3,
                    n_classes=2,
                    depth=hyper_params["depth"],
                    wf=hyper_params["wf"],
                    padding=True,
                    batch_norm=True,
                    up_mode="upconv",
                    optimizer=[partial(Adam, lr=hyper_params["lr"])],
                    edge_weight=hyper_params["edge_weight"])

        trainer.fit(unet, train_data, val_data)

        # save best model result and best model path in a yaml file
        best_model_yaml = Path(ckpt_callback.dirpath).parent / "best_model.yaml"
        ckpt_callback.to_yaml(best_model_yaml)

        return ckpt_callback


def parse_arguments():
    parser = argparse.ArgumentParser()

    # positional arguments
    parser.add_argument("data_root",
                        help="root directory containing data; relative to the script")
    parser.add_argument("output_dir",
                        help="root directory saving logs and model checkpoints;"
                             "relative to the script;"
                             "dataset_name and class_name will be appended to it.")
    parser.add_argument("class_name",
                        choices=CLASS_NAMES,
                        help="the name of the target output class.")

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

    # optional arguments: model hyper-parameters
    parser.add_argument("-v", "--dry_run",
                        action="store_true",
                        help="Run in 'dry_run' mode to check input arguments.")
    parser.add_argument("--lr",
                        type=float,
                        default=1e-3,
                        help="learning rate of Adam optimizer")
    parser.add_argument("--epochs",
                        type=int,
                        default=60,
                        help="training epochs")
    parser.add_argument("--depth",
                        type=int,
                        default=5,
                        help="depth of UNet model")
    parser.add_argument("--wf",
                        type=int,
                        default=3,
                        help="number of filters in the first layer is 2**wf")
    parser.add_argument("--edge_weight",
                        type=float,
                        default=1.2,
                        help="special weight for edges used in loss function")

    # optional arguments: optuna settings
    parser.add_argument("--use_optuna",
                        action="store_true",
                        help="use optuna for automatic tunning")
    parser.add_argument("-n",
                        "--n_trials",
                        type=int,
                        default=500,
                        help="number of optuna trials")

    args = parser.parse_args()
    use_multistain = args.use_multistain
    use_collage = args.use_collage

    args.dataset_name = "AAPI" if not use_collage and not use_multistain \
                            else ("MultiStain" if use_multistain else "Collage")

    args.data_root = (Path.cwd() / Path(args.data_root)).resolve()
    args.log_dir = (Path.cwd() / Path(args.output_dir) / Path(f"{args.dataset_name}/{args.class_name}/")).resolve()

    return args


def check_file_existence(p: Path):
    if not p.exists():
        raise FileNotFoundError(f"{p} doesn't exist.")


def start_optuna_tuning(args):
    if args.use_optuna:
        db_path = "sqlite:///" + str(args.log_dir) + "/study.db"
        study = optuna.create_study(study_name=f"UNet-Bin-{args.class_name}",
                                    storage=db_path,
                                    load_if_exists=True,
                                    direction="maximize")
        if args.dry_run:
            print("=" * 10 + "DRY RUN FOR OPTUNA" + "=" * 10)
            print(f"DB: {db_path};")
            train_binary_unet_model(args)
        else:
            study.optimize(lambda trial: objective_for_binary_unet(args, trial), n_trials=args.n_trials)
    else:
        train_binary_unet_model(args)


def objective_for_binary_unet(args, trial: optuna.trial.Trial):
    args.lr = trial.suggest_loguniform("lr", low=1e-5, high=1e-2)
    args.edge_weight = trial.suggest_uniform("edge_weight", low=1, high=5)
    args.wf = trial.suggest_int("wf", low=2, high=4)
    args.depth = trial.suggest_int("depth", low=4, high=6)

    pl_pruning_callback = PyTorchLightningPruningCallback(trial, "val/f1_score")
    ckpt_callback = train_binary_unet_model(args, callbacks=[pl_pruning_callback])

    best_f1_score = ckpt_callback.best_model_score.detach().cpu().numpy().item()
    trial.set_user_attr("best_val_f1", best_f1_score)
    trial.set_user_attr("best_model_path", ckpt_callback.best_model_path)

    return best_f1_score


if __name__ == "__main__":
    args = parse_arguments()
    start_optuna_tuning(args)
