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

import torch
from diff_gaussian_rasterization import (GaussianRasterizationSettings,
                                         GaussianRasterizer)

from gaussian_splatting.scene.gaussian_model import GaussianModel


def render(
    viewpoint_camera,
    gaussian_model: GaussianModel,
    bg_color: torch.Tensor = None,
    scaling_modifier=1.0,
):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """
    if bg_color is None:
        bg_color = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=gaussian_model.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=False,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = (
        torch.zeros_like(
            gaussian_model.get_xyz,
            dtype=gaussian_model.get_xyz.dtype,
            requires_grad=True,
            device="cuda",
        )
        + 0
    )
    screenspace_points.retain_grad()

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    rendered_image, radii = rasterizer(
        means3D=gaussian_model.get_xyz,
        means2D=screenspace_points,
        shs=gaussian_model.get_features,
        opacities=gaussian_model.get_opacity,
        scales=gaussian_model.get_scaling,
        rotations=gaussian_model.get_rotation,
        colors_precomp=None,
        cov3D_precomp=None,
    )

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return (
        rendered_image,
        screenspace_points,
        radii > 0,
        radii,
    )
