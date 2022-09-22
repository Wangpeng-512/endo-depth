import math
import torch
from pathlib import Path
from torch.utils.data import Dataset
from geometry.camera import CameraModel


class BlenderData:

    def __init__(self, path: str):

        self.cam = CameraModel("assets/blender.yaml")
        self.K = self.cam.K.float()
        self.invK = self.cam.IK.float()

        self.near = 0.01  # ratio of focal length
        self.far = 100   # ratio of focal length
        self.focal = 20   # [mm]

        self.log_near = math.log10(self.near)
        self.log_far = math.log10(self.far)

        self.filenames = [p for p in Path(path).glob("*.pth")]
        self.length = len(self.filenames)
        self.data = {}

    def get_index(self, index):
        fname: Path = self.filenames[index]
        return int(fname.stem)

    def get_color(self, index, case: int):
        f = self.filenames[index]
        data = torch.load(f)
        return data[case]

    def get_depth(self, index):
        f = self.filenames[index]
        data = torch.load(f)
        return data[-1]

    def get_intrinsic(self):
        return self.K, self.invK


CASE_NORMAL = 0
CASE_LIGHT = 1
CASE_DARK = 2


class BlenderDataset(Dataset):
    """ Blender sysnthesis dataset.
    """

    def __init__(self, case: int, src: BlenderData = None,
                 *args, **kwargs):
        super(BlenderDataset, self).__init__(*args, **kwargs)
        assert(case in [0, 1, 2])
        self.src = src
        self.case = case

    def get_index(self, index):
        return self.src.get_index(index)

    def get_color(self, index):
        return self.src.get_color(index, self.case)

    def get_depth(self, index):
        return self.src.get_depth(index)

    def get_intrinsic(self, index=None):
        return self.src.get_intrinsic()

    def __getitem__(self, index):
        inputs = {
            "color": self.get_color(index),
            "depth": self.get_depth(index),
        }
        return inputs
    
    def __len__(self):
        return self.src.length
