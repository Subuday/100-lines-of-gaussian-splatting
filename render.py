import math
import torch
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from model import GaussianModel
from camera import Camera


def render(camera: Camera, model: GaussianModel, scaling_modifier=1.0, override_color=None, device="cpu"):
    # settings = GaussianRasterizationSettings(
    #     image_width=camera.image_width,
    #     image_height=camera.image_height,
    #     tanfovx=math.tan(camera.FoVx * 0.5),
    #     tanfovy=math.tan(camera.FoVy * 0.5),
    #     bg=torch.tensor([0, 0, 0], dtype=torch.float32, device=device),
    #     scale_modifier=scaling_modifier,
    #     viewmatrix=camera.viewmatrix,
    # )
    pass