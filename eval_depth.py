import os
import cv2
import glob
import numpy as np
import argparse
import torch
import torchvision.transforms as tfm
from pathlib import Path
from geometry.camera import CameraModel, DepthBackprojD
import torchvision.transforms as tfm
from torchvision.datasets.folder import pil_loader
from tools.trainers.endodepth import plEndoDepth
from tools.visualize import visual_depth
from tools.o3d_visual import np6d_o3d_color, o3d


def parse_args():
    parser = argparse.ArgumentParser(
        description='Function to estimate depth maps of single or multiple images using an Endo-Depth model.')

    parser.add_argument(
        '--log_path',
        type=str,
        help='path to lightning log foler.',
        required=True
    )
    parser.add_argument(
        '--image_path',
        type=str,
        help='path to a test image or folder of images',
        required=True
    )
    parser.add_argument(
        '--intrinsic_path',
        type=str,
        help='path to numpy txt files of 4*4 intrinsic matrix',
        default=None
    )
    parser.add_argument(
        '--ext',
        type=str,
        help='image extension to search for in folder',
        default="png"
    )
    parser.add_argument(
        '--output_path',
        type=str,
        default=None,
        help='path to the output directory where the results will be stored'
    )
    return parser.parse_args()


def predict(args):
    # Get log
    path = Path(args.log_path)
    ckpt = [f for f in (path / "checkpoints").glob("*.ckpt")]
    if len(ckpt) <= 0:
        raise RuntimeError("len(ckpt) <= 0")
    ckpt = str(ckpt[-1])
    hpam = path / "hparams.yaml"
    # Load model
    model = plEndoDepth.load_from_checkpoint(
        "test/epoch=15-step=1439.ckpt",
        test=True,
        hparams_file=str(hpam),
        map_location="cpu")

    # Load camera
    H, W = model.options.height, model.options.width
    cam = CameraModel(args.intrinsic_path)
    backproj = DepthBackprojD(H, W, cam.IK, cam.rd, cam.td)
    transform = tfm.Compose([
        tfm.ToTensor(),
        tfm.Resize((H, W)),
    ])

    with torch.no_grad():
        # PREPROCESS
        file = Path(args.image_path)
        color = pil_loader(file)
        color = transform(color)[None]
        depth = model.forward(color)

        # VISUALIZE
        pc = backproj.forward(depth)
        pc_col = torch.cat([pc[:, 0, :3], color], dim=1)
        pcd = np6d_o3d_color(pc_col.detach().cpu().view(6, -1).T.contiguous().numpy())

        if args.output_path is not None:
            outpath = Path(args.output_path).expanduser()
            if not outpath.exists():
                os.makedirs(outpath)

            f = file.stem
            img = visual_depth(depth, show=False)
            fname = '{}/{}.jpg'.format(outpath, f)
            cv2.imwrite(fname, img)
            fname = '{}/{}.npy'.format(outpath, f)
            np.save(fname, depth.cpu().numpy())
            fname = '{}/mesh_{}.ply'.format(outpath, f)
            o3d.io.write_point_cloud(fname, pcd, write_ascii=True)

        o3d.visualization.draw_geometries([pcd], width=640, height=480)


if __name__ == "__main__":
    args = parse_args()
    predict(args)
