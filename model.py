import numpy as np
import torch
from torch import nn
from simple_knn._C import distCUDA2
from utils.sh_utils import RGB2SH
from utils.utils import inverse_sigmoid


class GaussianModel:

    def __init__(self, model_params, device="cpu"):
        self.sh_degree = 0
        self.max_sh_degree = model_params.sh_degree
        self.param_point_cloud = None
        self.features_dc = None
        self.features_rest = None
        self.scaling = None
        self.rotation = None
        self.opacity = None
        self.device = device

    @property
    def features(self):
        return torch.cat((self.features_dc, self.features_rest), dim=1)

    def init_from_scene_info(self, scene_info):
        point_cloud_tensor = torch.tensor(scene_info.point_cloud.points).to(self.device)
        self.param_point_cloud = nn.Parameter(point_cloud_tensor.requires_grad_(True))

        point_colors_tensor = RGB2SH(torch.tensor(scene_info.point_cloud.colors, device=self.device))
        features_tensor = torch.zeros((point_colors_tensor.shape[0], 3, (self.max_sh_degree + 1) ** 2), dtype=torch.float, device=self.device)
        features_tensor[:, :3, 0 ] = point_colors_tensor
        features_tensor[:, 3:, 1:] = 0.0
        self.features_dc = nn.Parameter(features_tensor[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self.features_rest = nn.Parameter(features_tensor[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))

        squared_dists = torch.clamp_min(distCUDA2(torch.tensor(scene_info.point_cloud.points, device=self.device)), 0.0000001)
        scaling_tensor = torch.log(torch.sqrt(squared_dists))[...,None].repeat(1, 3)
        self.scaling = nn.Parameter(scaling_tensor.requires_grad_(True))

        rotation_tensor = torch.zeros((point_cloud_tensor.shape[0], 4), device=self.device)
        rotation_tensor[:, 0] = 1
        self.rotation = nn.Parameter(rotation_tensor.requires_grad_(True))

        opacity_tensor = inverse_sigmoid(0.1 * torch.ones((point_cloud_tensor.shape[0], 1), device=self.device))
        self.opacity = nn.Parameter(opacity_tensor.requires_grad_(True))
