# Adapted from https://discuss.pytorch.org/t/unet-implementation/426

import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import f1_score, precision_recall
from functools import partial


class UNet(pl.LightningModule):
    def __init__(self,
                 in_channels,
                 n_classes,
                 depth,
                 wf,
                 padding,
                 batch_norm,
                 up_mode='upconv',
                 **train_kwargs
                 ):
        """
        Implementation of
        U-Net: Convolutional Networks for Biomedical Image Segmentation
        (Ronneberger et al., 2015)
        https://arxiv.org/abs/1505.04597

        Using the default arguments will yield the exact version used
        in the original paper

        Refactored using pytorch lightning, but is still compatible with pytorch

        Args:
            in_channels (int): number of input channels
            n_classes (int): number of output channels
            depth (int): depth of the network
            wf (int): number of filters in the first layer is 2**wf
            padding (bool): if True, apply padding such that the input shape
                            is the same as the output.
                            This may introduce artifacts
            batch_norm (bool): Use BatchNorm after layers with an
                               activation function
            up_mode (str): one of 'upconv' or 'upsample'.
                           'upconv' will use transposed convolutions for
                           learned upsampling.
                           'upsample' will use bilinear upsampling.
        """
        super().__init__()
        assert up_mode in ('upconv', 'upsample')
        self.save_hyperparameters()
        self.padding = padding
        self.depth = depth
        self.n_classes = n_classes
        prev_channels = in_channels
        self.down_path = nn.ModuleList()
        for i in range(depth):
            self.down_path.append(UNetConvBlock(prev_channels, 2**(wf+i),
                                                padding, batch_norm))
            prev_channels = 2**(wf+i)

        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path.append(UNetUpBlock(prev_channels, 2**(wf+i), up_mode,
                                            padding, batch_norm))
            prev_channels = 2**(wf+i)

        self.last = nn.Conv2d(prev_channels, n_classes, kernel_size=1)

        self.train_kwargs = train_kwargs

        self.optimizer = self.train_kwargs.get("optimizer",
                                               [partial(Adam, lr=1e-3)])
        self.optimizer = [opt(self.parameters()) for opt in self.optimizer]

        self.scheduler = self.train_kwargs.get("scheduler",
                                               [ReduceLROnPlateau(opt, factor=0.1, patience=25)
                                                for opt in self.optimizer])
        self.scheduler = [{
            'scheduler': lr_scheduler,
            'reduce_on_plateau': True,
            'monitor': 'val/loss'} for lr_scheduler in self.scheduler]
        self.criterion = self.train_kwargs.get("criterion", nn.CrossEntropyLoss(reduction="none"))
        self.edge_weight = self.train_kwargs.get("edge_weight", 1.)

    def forward(self, x):
        blocks = []
        for i, down in enumerate(self.down_path):
            x = down(x)
            if i != len(self.down_path)-1:
                blocks.append(x)
                x = F.avg_pool2d(x, 2)

        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i-1])

        return self.last(x)

    def configure_optimizers(self):
        return self.optimizer, self.scheduler

    def __shared_step_op(self, batch, batch_idx, phase, log=True):
        img, mask, edge_mask = batch
        output = self.forward(img)
        loss_matrix = self.criterion(output, mask)
        loss = (loss_matrix * (self.edge_weight ** edge_mask)).mean()

        output_labels = torch.argmax(output, dim=1).view(-1)
        ground_truths = mask.view(-1)
        f1 = f1_score(output_labels,
                      ground_truths,
                      num_classes=self.n_classes,
                      class_reduction='macro')

        precision, recall = precision_recall(output_labels,
                                             ground_truths,
                                             num_classes=self.n_classes,
                                             class_reduction="macro")

        if log:
            self.log(f"{phase}/loss", loss, prog_bar=True)
            self.log(f"{phase}/f1_score", f1, prog_bar=True)
            self.log(f"{phase}/precision", precision, prog_bar=False)
            self.log(f"{phase}/recall", recall, prog_bar=False)

        return {f"{phase}_loss": loss, f"{phase}_f1_score": f1}

    def training_step(self, batch, batch_idx):
        res = self.__shared_step_op(batch, batch_idx, "train")
        return res["train_loss"]

    def validation_step(self, batch, batch_idx):
        res = self.__shared_step_op(batch, batch_idx, "val")
        return res

    def test_step(self, batch, batch_idx):
        res = self.__shared_step_op(batch, batch_idx, "test")
        return res

    def __shared_epoch_op(self, outputs, phase):
        pass

    def validation_epoch_end(self, outputs):
        return self.__shared_epoch_op(outputs, "val")

    def test_epoch_end(self, outputs):
        return self.__shared_epoch_op(outputs, "test")


class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, padding, batch_norm):
        super(UNetConvBlock, self).__init__()
        block = []

        block.append(nn.Conv2d(in_size, out_size, kernel_size=3,
                               padding=int(padding)))
        block.append(nn.ReLU())
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))

        block.append(nn.Conv2d(out_size, out_size, kernel_size=3,
                               padding=int(padding)))
        block.append(nn.ReLU())
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out


class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, up_mode, padding, batch_norm):
        super(UNetUpBlock, self).__init__()
        if up_mode == 'upconv':
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2,
                                         stride=2)
        elif up_mode == 'upsample':
            self.up = nn.Sequential(nn.Upsample(mode='bilinear', scale_factor=2),
                                    nn.Conv2d(in_size, out_size, kernel_size=1))

        self.conv_block = UNetConvBlock(in_size, out_size, padding, batch_norm)

    def center_crop(self, layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[:, :, diff_y:(diff_y + target_size[0]), diff_x:(diff_x + target_size[1])]

    def forward(self, x, bridge):
        up = self.up(x)
        crop1 = self.center_crop(bridge, up.shape[2:])
        out = torch.cat([up, crop1], 1)
        out = self.conv_block(out)

        return out