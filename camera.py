from torch import nn
import torch
from utils.graphics_utils import getProjectionMatrix, getWorld2View2


class Camera(nn.Module):
    def __init__(self, uid, colmap_id, R, T, FoVx, FoVy, image, image_name):
        super(Camera, self).__init__()
        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.original_image = image.clamp(0.0, 1.0)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]
        self.z_far = 100.0
        self.z_near = 0.01
        self.world_2_camera_transform = torch.tensor(getWorld2View2(R, T)).transpose(0, 1)
        self.camera_2_clip_transform = getProjectionMatrix(znear=self.z_near, zfar=self.z_far, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1)
        self.world_2_clip_transform = self.camera_2_clip_transform @ self.world_2_camera_transform
        self.camera_center = self.world_2_camera_transform.inverse()[3, :3]
