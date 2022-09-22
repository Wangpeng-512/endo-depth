import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from collections import OrderedDict
from kornia.losses.depth_smooth import inverse_depth_smoothness_loss
from kornia.geometry.depth import depth_to_normals, depth_to_3d

import networks.metrics as METRIC
import networks.layers as layers
import networks.losses as LOSSES
from networks.endodepth import ResnetAttentionEncoder, DepthDecoder

from datasets.dataloader import ConcatDataset
from tools.trainer import Trainer
from tools.trainers.make_dataset import get_depth_dataset_only
from geometry.sdf.pysdf import SDFOptions, mesh_to_tsdf
from geometry.projective import DepthBackproj
from geometry import lie


class EndoDepthTrainer(Trainer):

    @staticmethod
    def make_models(options, *args, **kwargs):
        param = []

        encoder = ResnetAttentionEncoder(options.num_layers, False)
        param += list(encoder.parameters())

        depth = DepthDecoder(encoder.num_ch_enc, [0, 1, 2, 3])
        param += list(depth.parameters())

        models = OrderedDict([
            ("encoder", encoder),
            ("depth", depth)
        ])
        return models, param

    def __init__(self, options, *args, **kwargs):

        # checking height and width are multiples of 32
        assert options.height % 32 == 0, "'height' must be a multiple of 32"
        assert options.width % 32 == 0, "'width' must be a multiple of 32"

        if isinstance(options.data_path, list):
            train_dataset = [get_depth_dataset_only(
                path, options.height, options.width, options.png,
                options.frame_ids, True) for path in options.data_path]
            train_dataset = ConcatDataset(train_dataset)
        else:
            train_dataset = get_depth_dataset_only(
                options.data_path, options.height, options.width, options.png,
                options.frame_ids, True)
        train_loader = DataLoader(
            train_dataset, options.batch_size, True,
            num_workers=options.num_workers, pin_memory=True, drop_last=False
        )

        if options.val_path is not None:
            val_dataset = get_depth_dataset_only(
                options.val_path, options.height, options.width, options.png,
                options.frame_ids, True)
            val_loader = DataLoader(
                val_dataset, options.batch_size, True,
                num_workers=options.num_workers, pin_memory=True, drop_last=False
            )
        else:
            val_loader = None
        ######################################################
        # val_dataset = Subset(val_dataset, range(0, len(val_dataset), 3))
        if isinstance(train_dataset, Subset):
            dataset = train_dataset.dataset
        if isinstance(train_dataset, ConcatDataset):
            dataset = train_dataset.datasets[0]
        else:
            dataset = train_dataset

        self.K = dataset.K
        self.IK = dataset.invK
        options.min_depth_units = dataset.log_near
        options.max_depth_units = dataset.log_far
        options.min_depth = dataset.near
        options.max_depth = dataset.far

        ######################################################
        # NETWORK #
        ######################################################
        models, param = self.make_models(options)

        opm = optim.Adam(param, options.learning_rate,
                         betas=options.betas,
                         weight_decay=options.weight_decay)

        if isinstance(options.scheduler_step_size, int):
            milestone = [options.scheduler_step_size]
        else:
            milestone = options.scheduler_step_size
        lr = optim.lr_scheduler.MultiStepLR(
            opm, milestones=milestone,
            gamma=options.lr_decade_coeff)

        # Replace options with dict for saving
        self.sdf_opts: SDFOptions = options.sdf_options
        options.sdf_options = self.sdf_opts.__dict__.copy()

        super(EndoDepthTrainer, self).__init__(
            models=models,
            optim=opm,
            sceduler=lr,
            train_loader=train_loader,
            val_loader=val_loader,
            options=options,
            *args, **kwargs
        )

        ######################################################
        self.K = self.K.float().to(self.device)
        self.IK = self.IK.float().to(self.device)

        self.cam_scale = self.sdf_opts.scale
        self.cam_origin = torch.FloatTensor(self.sdf_opts.origin).view(3).to(self.device)
        self.backproj = DepthBackproj(options.height, options.width).to(self.device)

    def on_model_input(self, inputs):
        I = torch.squeeze(inputs["color"], 1)
        D = torch.squeeze(inputs["depth"], 1)
        # print(inputs["index"][0])
        return I, D

    def run_batch(self, inputs, bid=None):
        images, depth_gt = self.on_model_input(inputs)

        encoder: ResnetAttentionEncoder = self.models['encoder']
        decoder: DepthDecoder = self.models['depth']

        features = encoder.forward(images)
        outputs = decoder.forward(features, scales=self.options.scales)

        # Convert sigmoid output to depth
        depths = {}
        for scale in self.options.scales:
            disp = outputs[scale]
            _, depth = layers.disp_to_depth_log10(
                disp, self.options.min_depth_units,
                self.options.max_depth_units, 1.0)
            depths[scale] = depth

        # Calculate losses
        losses = self.compute_loss(images, depth_gt, depths)

        # Calculate metrics
        if bid is None or bid % 20 == 0:  # validation
            with torch.no_grad():
                metrics = METRIC.compute_depth_metrics(depth_gt, depths[0])
                for k, v in metrics.items():
                    metrics[k] = v.detach().cpu().item()
                losses.update(metrics)
        return depths, losses

    def compute_loss(self, images, depth_gt, depths):

        losses = {}
        if self.options.use_sdf_loss:
            losses['loss/sdf'] = self.compute_sdf_loss(depths[0], depth_gt, self.IK)
        else:
            losses['loss/sdf'] = 0

        if self.options.use_smooth_loss:
            idepth = 1.0 / (depths[0] + 1e-7)
            losses['loss/smooth'] = inverse_depth_smoothness_loss(idepth, images)
        else:
            losses['loss/smooth'] = 0

        if self.options.use_depth_loss:
            losses['loss/depth'] = LOSSES.depth_loss(depth_gt, depths[0], use_depth=True, use_normal=False, use_gradient=False)
        else:
            losses['loss/depth'] = 0

        if self.options.use_normal_loss:
            losses['loss/pc'], losses['loss/normal'] = self.compute_normal_loss(depths[0], depth_gt, self.K)
        else:
            losses['loss/pc'] = 0
            losses['loss/normal'] = 0

        losses['loss'] = \
            self.options.weight_sdf_loss * losses['loss/sdf'] + \
            self.options.weight_depth_loss * losses['loss/depth'] + \
            self.options.weight_normal_pc_loss * losses['loss/pc'] + \
            self.options.weight_normal_norm_loss * losses['loss/normal'] + \
            self.options.weight_smooth_loss * losses['loss/smooth']
        return losses

    def compute_normal_loss(self, depth: torch.Tensor, depth_gt: torch.Tensor, K0: torch.Tensor):

        B, C, H, W = depth_gt.shape

        K = K0.clone()
        K[0, 0] *= W
        K[1, 1] *= H
        K[0, 2] *= (W - 1)
        K[1, 2] *= (H - 1)

        pc0 = depth_to_3d(depth_gt, K[:3, :3][None])
        norm0 = depth_to_normals(depth_gt, K[:3, :3][None])
        pc1 = depth_to_3d(depth, K[:3, :3][None])
        norm1 = depth_to_normals(depth, K[:3, :3][None])

        loss_pc = F.l1_loss(pc0, pc1)
        simil_ = F.cosine_similarity(norm0, norm1, dim=-1)
        loss_normal = torch.mean(1.0 - simil_.abs())

        return loss_pc, loss_normal

    def compute_sdf_loss(self, depth: torch.Tensor, depth_gt: torch.Tensor, IKn: torch.Tensor):

        B, _, H, W = depth_gt.shape

        # A. Reproject to point cloud
        P0 = self.backproj.forward(depth_gt, IKn, norm=True)
        P1 = self.backproj.forward(depth, IKn, norm=True)

        # B. calculate sdf for each point cloud
        P = torch.cat([P0, P1], dim=1).view(-1, 4, H, W)

        bb = P[:, :3].permute(1, 0, 2, 3).contiguous().view(3, -1)
        bb_min = bb.min(dim=-1).values.view(1, 3)
        bb_max = bb.max(dim=-1).values.view(1, 3)
        sdf = mesh_to_tsdf(P, resolution=self.sdf_opts.resolution,
                           bb_min=bb_min, bb_max=bb_max,
                           offset=self.cam_origin, scale=self.cam_scale, ret_xyz=True,
                           trunc=self.sdf_opts.trunc, trunc_fill=self.sdf_opts.trunc_fill)

        sdf = sdf.view(B, -1, *sdf.shape[1:])
        sdf_gt, sdf_pred = sdf.unbind(dim=1)

        # from tools.o3d_tsdf import run_tsdf_windows
        # from tools.visualize import visual_rgb
        # run_tsdf_windows(sdf_gt[0].detach().cpu().numpy())
        # run_tsdf_windows(sdf_pred[0].detach().cpu().numpy())

        l1 = sdf_gt[..., -1]
        l2 = sdf_pred[..., -1]
        mask = torch.bitwise_and(l1 > 0, l2 > 0)

        # C. calculate cross sdf loss
        return F.mse_loss(l1[mask], l2[mask])

    def on_save_model(self, state_dict):
        state_dict['height'] = self.options.height
        state_dict['width'] = self.options.width

    def on_load_model(self, state_dict):
        if 'height' in state_dict.keys():
            self.options.height = state_dict['height']
        if 'width' in state_dict.keys():
            self.options.width = state_dict['width']
