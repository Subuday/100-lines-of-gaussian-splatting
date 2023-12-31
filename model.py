import numpy as np
import torch
from torch import nn
from simple_knn._C import distCUDA2
from utils.sh_utils import RGB2SH
from utils.utils import inverse_sigmoid, build_rotation


class GaussianModel:

    def __init__(self, model_params, device="cpu"):
        self.sh_degree = 0
        self.max_sh_degree = model_params.sh_degree
        self.spatial_lr_scale = 0
        self._point_cloud = None
        self._features_dc = None
        self._features_rest = None
        self._scaling = None
        self._rotation = None
        self._opacity = None
        self.gaussians_max_radiuses = None
        self.denom = None
        self.point_cloud_gradient_accum = None
        self.percent_dense = 0
        self.optimizer = None
        self.device = device

    @property
    def point_cloud(self):
        return self._point_cloud

    @property
    def features(self):
        return torch.cat((self._features_dc, self._features_rest), dim=1)

    @property
    def scaling(self):
        return torch.exp(self._scaling)

    def densify_and_prune_gaussians(self, grad_threshold):
        camera_extent = self.spatial_lr_scale

        grads = self.point_cloud_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.clone_gaussians(grads, grad_threshold, camera_extent)
        self.split_gaussians(grads, grad_threshold, camera_extent)

    def clone_gaussians(self, grads, grad_threshold, camera_extent):
        selected_points = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_points = torch.logical_and(
            selected_points,
            torch.max(self.scaling, dim=1).values <= self.percent_dense * camera_extent
        )

        new_point_cloud = self._point_cloud[selected_points]
        new_features_dc = self._features_dc[selected_points]
        new_features_rest = self._features_rest[selected_points]
        new_opacity = self._opacity[selected_points]
        new_scaling = self._scaling[selected_points]
        new_rotation = self._rotation[selected_points]

        self.update_gaussians_after_densification(
            new_point_cloud,
            new_features_dc,
            new_features_rest,
            new_opacity,
            new_scaling,
            new_rotation
        )

    def split_gaussians(self, grads, grad_threshold, camera_extent, N=2):
        n_points = self.point_cloud.shape[0]
        padded_grads = torch.zeros(n_points, device=self.device)
        padded_grads[:grads.shape[0]] = grads.squeeze()

        selected_points = torch.where(padded_grads >= grad_threshold, True, False)
        selected_points = torch.logical_and(
            selected_points,
            torch.max(self.scaling, dim=1).values > self.percent_dense * camera_extent
        )

        stds = self.scaling[selected_points].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device=self.device)
        samples = torch.normal(mean=means, std=stds)

        rot = build_rotation(self._rotation[selected_points]).repeat(N, 1, 1)
        new_point_cloud = torch.bmm(
            rot,
            samples.unsqueeze(-1)
        ).squeeze(-1) + self.point_cloud[selected_points].repeat(N, 1)
        new_scaling = torch.log(self.scaling[selected_points].repeat(N, 1) / (0.8 * N))
        new_rotation = self._rotation[selected_points].repeat(N, 1)
        new_features_dc = self._features_dc[selected_points].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_points].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_points].repeat(N, 1)

        self.update_gaussians_after_densification(
            new_point_cloud,
            new_features_dc,
            new_features_rest,
            new_opacity,
            new_scaling,
            new_rotation
        )



    def update_gaussians_after_densification(
            self,
            point_cloud,
            features_dc,
            features_rest,
            opacities,
            scaling,
            rotation
    ):
        params = {
            "point_cloud": point_cloud,
            "f_dc": features_dc,
            "f_rest": features_rest,
            "opacity": opacities,
            "scaling": scaling,
            "rotation": rotation
        }

        updated_params = self.update_optimizer(params)
        self._point_cloud = updated_params["point_cloud"]
        self._features_dc = updated_params["f_dc"]
        self._features_rest = updated_params["f_rest"]
        self._opacity = updated_params["opacity"]
        self._scaling = updated_params["scaling"]
        self._rotation = updated_params["rotation"]

        self.point_cloud_gradient_accum = torch.zeros((self.point_cloud.shape[0], 1), device=self.device)
        self.denom = torch.zeros((self.point_cloud.shape[0], 1), device=self.device)
        self.gaussians_max_radiuses = torch.zeros((self.point_cloud.shape[0]), device=self.device)

    def update_optimizer(self, params):
        updated_params = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            new_state = params[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat(
                    (stored_state["exp_avg"], torch.zeros_like(new_state)),
                    dim=0
                )
                stored_state["exp_avg_sq"] = torch.cat(
                    (stored_state["exp_avg_sq"], torch.zeros_like(new_state)),
                    dim=0
                )

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], new_state), dim=0).requires_grad_(True)
                )
                self.optimizer.state[group['params'][0]] = stored_state

                updated_params[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], new_state), dim=0).requires_grad_(True)
                )
                updated_params[group["name"]] = group["params"][0]
        return updated_params

    def prune_gaussians(self):
        pass

    def init_from_scene_info(self, scene_info):
        self.spatial_lr_scale = scene_info.nerf_normalization["radius"]

        point_cloud_tensor = torch.tensor(scene_info.point_cloud.points).to(self.device)
        self._point_cloud = nn.Parameter(point_cloud_tensor.requires_grad_(True))

        point_colors_tensor = RGB2SH(torch.tensor(scene_info.point_cloud.colors, device=self.device))
        features_tensor = torch.zeros((point_colors_tensor.shape[0], 3, (self.max_sh_degree + 1) ** 2),
                                      dtype=torch.float, device=self.device)
        features_tensor[:, :3, 0] = point_colors_tensor
        features_tensor[:, 3:, 1:] = 0.0
        self._features_dc = nn.Parameter(features_tensor[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features_tensor[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))

        squared_dists = torch.clamp_min(
            distCUDA2(torch.tensor(scene_info.point_cloud.points, device=self.device)),
            0.0000001
        )
        scaling_tensor = torch.log(torch.sqrt(squared_dists))[..., None].repeat(1, 3)
        self._scaling = nn.Parameter(scaling_tensor.requires_grad_(True))

        rotation_tensor = torch.zeros((point_cloud_tensor.shape[0], 4), device=self.device)
        rotation_tensor[:, 0] = 1
        self._rotation = nn.Parameter(rotation_tensor.requires_grad_(True))

        opacity_tensor = inverse_sigmoid(0.1 * torch.ones((point_cloud_tensor.shape[0], 1), device=self.device))
        self._opacity = nn.Parameter(opacity_tensor.requires_grad_(True))

        self.gaussians_max_radiuses = torch.zeros((self._point_cloud.shape[0]), device=self.device)

    def setup_training_params(self, params):
        self.point_cloud_gradient_accum = torch.zeros((self.point_cloud.shape[0], 1), device=self.device)
        self.denom = torch.zeros((self.point_cloud.shape[0], 1), device=self.device)
        self.percent_dense = params.percent_dense
        self.optimizer = torch.optim.Adam(
            params=[
                {'params': [self._point_cloud],
                 'lr': params.position_lr_init * self.spatial_lr_scale,
                 "name": "point_cloud"
                 },
                {'params': [self._features_dc], 'lr': params.feature_lr, "name": "f_dc"},
                {'params': [self._features_rest], 'lr': params.feature_lr / 20.0, "name": "f_rest"},
                {'params': [self._opacity], 'lr': params.opacity_lr, "name": "opacity"},
                {'params': [self._scaling], 'lr': params.scaling_lr, "name": "scaling"},
                {'params': [self._rotation], 'lr': params.rotation_lr, "name": "rotation"}
            ],
            lr=0.0,
            eps=1e-15
        )
