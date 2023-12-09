import numpy as np

from camera import Camera
from utils.graphics_utils import fov2focal
from utils.utils import pil_to_torch


def create_camera_from_camera_info(id, camera_info, resolution=-1, resolution_scale=1.0):
    orig_w, orig_h = camera_info.image.size

    if resolution in [1, 2, 4, 8]:
        resolution = round(orig_w / (resolution_scale * resolution)), round(orig_h / (resolution_scale * resolution))
    else:
        if resolution == -1:
            if orig_w > 1600:
                print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                      "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    return Camera(colmap_id=camera_info.uid,
                  R=camera_info.R,
                  T=camera_info.T,
                  FoVx=camera_info.FovX,
                  FoVy=camera_info.FovY,
                  image=pil_to_torch(camera_info.image, resolution),
                  image_name=camera_info.image_name,
                  uid=id)


def camera_to_json(id, camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id': id,
        'img_name': camera.image_name,
        'width': camera.width,
        'height': camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy': fov2focal(camera.FovY, camera.height),
        'fx': fov2focal(camera.FovX, camera.width)
    }
    return camera_entry
