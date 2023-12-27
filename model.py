import numpy as np
import torch
from torch import nn


class GaussianModel:

    def __init__(self, device="cpu"):
        self.param_point_cloud = None
        self.device = device
        pass

    def init_from_scene_info(self, scene_info):
        point_cloud_tensor = torch.tensor(scene_info.point_cloud.points).to(self.device)
        self.param_point_cloud = nn.Parameter(point_cloud_tensor.requires_grad_(True))
