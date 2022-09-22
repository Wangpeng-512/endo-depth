from typing import Dict
import torch
import torch.nn.functional as F
import torch.optim as optim
from time import time

import networks.layers as layers
import networks.losses as LOSSES
from kornia.losses.depth_smooth import inverse_depth_smoothness_loss
from kornia.geometry.depth import depth_to_normals, depth_to_3d

from networks.endodepth import ResnetAttentionEncoder, DepthDecoder
from datasets.blender_dataset import BlenderDataset

import pytorch_lightning as pl


class plEndoDepth(pl.LightningModule):

    def __init__(self, options, *args, **kwargs):
        super(plEndoDepth, self).__init__()

        ######################################################

        self.encoder = ResnetAttentionEncoder(options.num_layers, False)
        self.decoder = DepthDecoder(self.encoder.num_ch_enc, [0, 1, 2, 3])

        ######################################################

        if not options.test == True:
            # checking height and width are multiples of 32
            assert options.height % 32 == 0, "'height' must be a multiple of 32"
            assert options.width % 32 == 0, "'width' must be a multiple of 32"

            self.train_set, self.val_set = BlenderDataset(options.data_path), None

            dataset = self.train_set.src
            options.min_depth_units = dataset.log_near
            options.max_depth_units = dataset.log_far
            options.min_depth = dataset.near
            options.max_depth = dataset.far

            K, iK = dataset.get_intrinsic()
            self.register_buffer("K", (K[:3, :3]).view(1, 3, 3))
            self.register_buffer("iK", (iK[:3, :3]).view(1, 3, 3))
        else:
            self.eval()
            self.register_buffer("K", torch.eye(3).view(1, 3, 3).float())
            self.register_buffer("iK", torch.eye(3).view(1, 3, 3).float())

        self.options = options

    def configure_optimizers(self):
        param = []
        param += list(self.encoder.parameters())
        param += list(self.decoder.parameters())
        opm = optim.Adam(param, self.options.learning_rate,
                         betas=self.options.betas,
                         weight_decay=self.options.weight_decay)

        if isinstance(self.options.scheduler_step_size, int):
            milestone = [self.options.scheduler_step_size]
        else:
            milestone = self.options.scheduler_step_size

        lr = optim.lr_scheduler.MultiStepLR(
            opm, milestones=milestone,
            gamma=self.options.lr_decade_coeff)
        return [opm], [lr]

    def forward(self, images: torch.Tensor):
        features = self.encoder.forward(images)
        outputs = self.decoder.forward(features, scales=self.options.scales)
        return outputs

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):

        self.train()
        I = torch.squeeze(batch["color"], 1)
        D = torch.squeeze(batch["depth"], 1)

        out = self.forward(I)

        _, D_ = layers.disp_to_depth_log10(
            out[0], self.options.min_depth_units,
            self.options.max_depth_units, 1.0)

        losses = self.compute_loss(I, D, D_, prefix="train_")

        # Record
        self.log_dict(losses, True)
        return losses["train_loss"]

    def training_epoch_end(self, outputs):
        sch = self.lr_schedulers()
        sch.step()
        self.log(f'lr', sch.get_last_lr()[0])

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):

        self.eval()
        I = torch.squeeze(batch["color"], 1)
        D = torch.squeeze(batch["depth"], 1)

        out = self.forward(I)
        _, D_ = layers.disp_to_depth_log10(
            out, self.options.min_depth_units,
            self.options.max_depth_units, 1.0)

        losses = self.compute_loss(I, D, D_, prefix="val_")
        self.log_dict(losses)

    def compute_loss(self, images: torch.Tensor, depth_gt: torch.Tensor, depth: torch.Tensor, prefix: str = ""):

        losses = {}

        if self.options.use_smooth_loss:
            idepth = 1.0 / (depth + 1e-7)
            losses[f'{prefix}loss/smooth'] = inverse_depth_smoothness_loss(idepth, images)
        else:
            losses[f'{prefix}loss/smooth'] = 0.0

        if self.options.use_depth_loss:
            losses[f'{prefix}loss/depth'] = LOSSES.depth_loss(depth_gt, depth, use_depth=True, use_normal=False, use_gradient=False)
        else:
            losses[f'{prefix}loss/depth'] = 0.0

        if self.options.use_normal_loss:
            losses[f'{prefix}loss/pc'], losses[f'{prefix}loss/normal'] = self.compute_normal_loss(depth, depth_gt, self.K)
        else:
            losses[f'{prefix}loss/pc'] = 0.0
            losses[f'{prefix}loss/normal'] = 0.0

        losses[f'{prefix}loss'] = \
            self.options.weight_depth_loss * losses[f'{prefix}loss/depth'] + \
            self.options.weight_normal_pc_loss * losses[f'{prefix}loss/pc'] + \
            self.options.weight_normal_norm_loss * losses[f'{prefix}loss/normal'] + \
            self.options.weight_smooth_loss * losses[f'{prefix}loss/smooth']
        return losses

    def compute_normal_loss(self, depth: torch.Tensor, depth_gt: torch.Tensor, K0: torch.Tensor):
        B, C, H, W = depth_gt.shape

        K = K0.clone()
        K[..., 0, 0] *= W
        K[..., 1, 1] *= H
        K[..., 0, 2] *= (W - 1)
        K[..., 1, 2] *= (H - 1)

        pc0 = depth_to_3d(depth_gt, K)
        norm0 = depth_to_normals(depth_gt, K)
        pc1 = depth_to_3d(depth, K)
        norm1 = depth_to_normals(depth, K)

        loss_pc = F.l1_loss(pc0, pc1)
        simil_ = F.cosine_similarity(norm0, norm1, dim=-1)
        loss_normal = torch.mean(1.0 - simil_.abs())

        return loss_pc, loss_normal
