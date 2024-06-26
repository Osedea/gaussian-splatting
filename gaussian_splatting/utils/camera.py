#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import math

import numpy as np

from gaussian_splatting.dataset.cameras import Camera
from gaussian_splatting.utils.general import PILtoTorch
from gaussian_splatting.utils.graphics import fov2focal

WARNED = False


def load_camera(resolution, cam_id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.image.size

    if resolution in [1, 2, 4, 8]:
        resolution = round(orig_w / (resolution_scale * resolution)), round(
            orig_h / (resolution_scale * resolution)
        )
    else:  # should be a type that converts to float
        if resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print(
                        "[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1"
                    )
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    resized_image_rgb = PILtoTorch(cam_info.image, resolution)

    gt_image = resized_image_rgb[:3, ...]
    loaded_mask = None

    if resized_image_rgb.shape[1] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]

    return Camera(
        colmap_id=cam_info.uid,
        R=cam_info.R,
        T=cam_info.T,
        FoVx=cam_info.FovX,
        FoVy=cam_info.FovY,
        image=gt_image,
        gt_alpha_mask=loaded_mask,
        image_name=cam_info.image_name,
        uid=cam_id,
    )


def get_orthogonal_camera(image):
    camera = Camera(
        R=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        T=np.array([-0.5, -0.5, 1.0]),
        FoVx=2 * math.atan(0.5),
        FoVy=2 * math.atan(0.5),
        image=image,
        gt_alpha_mask=None,
        image_name="patate",
        colmap_id=0,
        uid=0,
    )

    return camera


def transform_camera(
    camera,
    rotation,
    translation,
    image,
    image_name: str = "",
    _id: int = 0,
):
    transformed_camera = Camera(
        R=np.matmul(camera.R, rotation),
        T=(camera.T + translation),
        FoVx=camera.FoVx,
        FoVy=camera.FoVy,
        image=image,
        gt_alpha_mask=None,
        image_name=image_name,
        colmap_id=_id,
        uid=_id,
    )
    return transformed_camera


def load_cameras(cameras_infos, resolution_scale, resolution):
    cameras = [
        load_camera(resolution, cam_id, cam_info, resolution_scale)
        for cam_id, cam_info in enumerate(cameras_infos)
    ]

    return cameras


def camera_to_json(_id, camera: Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]

    json_camera = {
        "id": _id,
        "img_name": camera.image_name,
        "width": camera.width,
        "height": camera.height,
        "position": pos.tolist(),
        "rotation": serializable_array_2d,
        "fy": fov2focal(camera.FovY, camera.height),
        "fx": fov2focal(camera.FovX, camera.width),
    }

    return json_camera
