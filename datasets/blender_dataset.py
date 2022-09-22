import math
import torch
from pathlib import Path
from torch.utils.data import Dataset
from geometry.camera import CameraModel


class BlenderData:

    def __init__(self, path: str, device = "cuda"):

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

    def get_data(self, index):
        f = self.filenames[index]
        return torch.load(f)


CASE_NORMAL = 0
CASE_LIGHT = 1
CASE_DARK = 2


class BlenderDataset(Dataset):
    """ Blender sysnthesis dataset.
    """

    def __init__(self, path: str, *args, **kwargs):
        super(BlenderDataset, self).__init__(*args, **kwargs)
        self.src = BlenderData(path)

    def __getitem__(self, index):
        c = index // self.src.length
        i = index % self.src.length
        data = self.src.get_data(i)
        inputs = {
            "color": data[c],
            "depth": data[-1],
        }
        return inputs

    def __len__(self):
        return self.src.length * 3
