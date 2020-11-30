import albumentations as A
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping


def get_augmentation_transforms(add_non_spatial=True):
    transforms = [
        A.VerticalFlip(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.OneOf([
            A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5),
            A.GridDistortion(p=0.5),
            A.OpticalDistortion(distort_limit=2, shift_limit=0.5, p=1)
        ], p=0.8),
    ]

    if add_non_spatial:
        transforms.extend([
        A.CLAHE(p=0.8),
        A.RandomBrightnessContrast(p=0.8),
        A.RandomGamma(p=0.8)
    ])

    aug_transform = A.Compose(transforms)

    return aug_transform


def get_early_stopping_callback(monitor="val/f1_score", mode="max", patience=10):
    return EarlyStopping(monitor=monitor, mode=mode, patience=patience, min_delta=0)


def get_checkpoint_callback(save_top_k, monitor="val/f1_score", mode="max"):
    return ModelCheckpoint(save_top_k=save_top_k, monitor=monitor, mode=mode)


def create_pl_trainer(use_gpu, root_dir, epochs, callbacks=None, checkpoint_callback=None):

    if callbacks is None:
        callbacks = [get_early_stopping_callback(patience=20)]

    if checkpoint_callback is None:
        checkpoint_callback = get_checkpoint_callback(1, )

    trainer = Trainer(gpus=1 if use_gpu else 0,
                      default_root_dir=root_dir,
                      max_epochs=epochs,
                      callbacks=callbacks,
                      checkpoint_callback=checkpoint_callback,
                      log_every_n_steps=1,
                      flush_logs_every_n_steps=50)

    return trainer