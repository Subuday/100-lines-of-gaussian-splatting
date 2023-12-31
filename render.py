import math
import torch
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from model import GaussianModel
from camera import Camera


def render(camera: Camera, model: GaussianModel, scaling_modifier=1.0, override_color=None, debug=False, device="cpu"):
    settings = GaussianRasterizationSettings(
        image_width=camera.image_width,
        image_height=camera.image_height,
        tanfovx=math.tan(camera.FoVx * 0.5),
        tanfovy=math.tan(camera.FoVy * 0.5),
        bg=torch.tensor([0, 0, 0], dtype=torch.float32, device=device),
        scale_modifier=scaling_modifier,
        viewmatrix=camera.world_2_camera_transform.to(device),
        projmatrix=camera.camera_2_clip_transform.to(device),
        sh_degree=model.sh_degree,
        campos=camera.camera_center.to(device),
        prefiltered=False,
        debug=debug
    )
    rasterizer = GaussianRasterizer(raster_settings=settings)

    clip_points = torch.zeros_like(model.point_cloud, dtype=model.point_cloud.dtype, requires_grad=True)

    rendered_image, radii = rasterizer(
        means3D = model.point_cloud,
        means2D = clip_points,
        shs = model.features,
        colors_precomp = None,
        opacities = model.opacity,
        scales = model.scaling,
        rotations = model.rotation,
        cov3D_precomp = None
    )

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {
            "image": rendered_image,
            "clip_points": clip_points,
            "visibility_filter" : radii > 0,
            "radii": radii
            }