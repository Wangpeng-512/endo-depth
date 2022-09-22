import torch
import torch.nn.functional as F
from torch.utils.data.dataset import Subset

import networks.metrics as METRIC
import networks.layers as layers
from networks.endodepth import ResnetAttentionEncoder, DepthDecoder
from tools.trainers.endodepth import EndoDepthTrainer
from geometry.normal import depth_to_normal


class EndoDepthTrainer_N(EndoDepthTrainer):

    def __init__(self, options, *args, **kwargs):
        super().__init__(options, *args, **kwargs)

        dataset = self.train_loader.dataset
        if isinstance(dataset, Subset):
            dataset = dataset.dataset
        self.K = dataset.K.to(self.device)

        self.options.lambda_depth = 1.0
        self.options.lambda_normal = 1.0

    def on_model_input(self, inputs):
        return inputs["color"], inputs["depth"]

    def run_batch(self, depth_gt, bid=None):

        images, depth_gt = self.on_model_input(depth_gt)

        encoder: ResnetAttentionEncoder = self.models['encoder']
        decoder: DepthDecoder = self.models['depth']

        B, N, C, H, W = images.shape
        depth_gt = depth_gt.view(-1, 1, H, W)

        features = encoder.forward(images.view(-1, 3, H, W))
        outputs = decoder.forward(features)

        # Convert sigmoid output to depth
        depth_pred = {}
        for scale in self.options.scales:
            disp = outputs[scale]
            _, depth = layers.disp_to_depth_log10(
                disp, self.options.min_depth_units,
                self.options.max_depth_units, 1.0)
            depth_pred[scale] = depth

        losses = self.compute_loss(images, depth_gt, depth_pred)
        return depth_pred, losses

    def compute_loss(self, images, depth_gt, depth_pred):

        losses = {}

        gamma = 0.9
        losses['loss/pc'] = 0
        losses['loss/normal'] = 0
        for s in self.options.scales:
            w = gamma ** s
            Dx = depth_pred[s]
            B, _, H, W = Dx.shape
            Dt = F.interpolate(depth_gt, (H, W), mode='bilinear', align_corners=False)

            norm0, pc0 = depth_to_normal(Dt.view(B, H, W), self.K, pc=True)
            norm1, pc1 = depth_to_normal(Dx.view(B, H, W), self.K, pc=True)

            loss_depth = F.l1_loss(pc0, pc1) * w

            simil_ = F.cosine_similarity(norm0, norm1, dim=-1)
            loss_normal = torch.mean(1.0 - simil_.abs()) * w

            losses['loss/pc/{}'.format(s)] = loss_depth
            losses['loss/normal/{}'.format(s)] = loss_normal
            losses['loss/pc'] += loss_depth
            losses['loss/normal'] += loss_normal

        losses['loss'] = \
            self.options.lambda_depth * losses['loss/pc'] + \
            self.options.lambda_normal * losses['loss/normal']

        # Calculate metrics
        with torch.no_grad():
            metrics = METRIC.compute_depth_metrics(depth_gt, depth_pred[0])
            for k, v in metrics.items():
                metrics[k] = v.detach().cpu().item()
        losses.update(metrics)
        return losses
