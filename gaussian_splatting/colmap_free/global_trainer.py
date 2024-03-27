import os
import uuid
from random import randint

import torch
from tqdm import tqdm

from gaussian_splatting.optimizer import Optimizer
from gaussian_splatting.render import render
from gaussian_splatting.trainer import Trainer
from gaussian_splatting.utils.general import safe_state
from gaussian_splatting.utils.loss import PhotometricLoss


class GlobalTrainer(Trainer):
    def __init__(self, gaussian_model):
        self._model_path = self._prepare_model_path()

        self.gaussian_model = gaussian_model
        self.cameras = []

        self.optimizer = Optimizer(self.gaussian_model)
        self._photometric_loss = PhotometricLoss(lambda_dssim=0.2)

        self._debug = False

        # Densification and pruning
        self._min_opacity = 0.005
        self._max_screen_size = 20
        self._percent_dense = 0.01
        self._densification_grad_threshold = 0.0002

        safe_state()

    def add_camera(self, camera):
        self.cameras.append(camera)

    def run(self, iterations: int = 1000):
        ema_loss_for_log = 0.0
        cameras = None
        first_iter = 1
        progress_bar = tqdm(range(first_iter, iterations), desc="Training progress")
        for iteration in range(first_iter, iterations + 1):
            self.optimizer.update_learning_rate(iteration)

            # Every 1000 its we increase the levels of SH up to a maximum degree
            if iteration % 1000 == 0:
                self.gaussian_model.oneupSHdegree()

            # Pick a random camera
            if not cameras:
                cameras = self.cameras.copy()
            camera = cameras.pop(randint(0, len(cameras) - 1))

            # Render image
            rendered_image, viewspace_point_tensor, visibility_filter, radii = render(
                camera, self.gaussian_model
            )

            # Loss
            gt_image = camera.original_image.cuda()
            loss = self._photometric_loss(rendered_image, gt_image)
            loss.backward()

            # Optimizer step
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
            progress_bar.update(1)

        progress_bar.close()

        point_cloud_path = os.path.join(
            self._model_path, "point_cloud/iteration_{}".format(iteration)
        )
        self.gaussian_model.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

        # Densification
        self.gaussian_model.update_stats(
            viewspace_point_tensor, visibility_filter, radii
        )
        self._densify_and_prune(True)
        self._reset_opacity()
