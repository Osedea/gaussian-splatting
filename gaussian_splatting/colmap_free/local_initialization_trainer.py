import math

import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt
from tqdm import tqdm
from transformers import pipeline
from pathlib import Path

from gaussian_splatting.dataset.cameras import Camera
from gaussian_splatting.model import GaussianModel
from gaussian_splatting.optimizer import Optimizer
from gaussian_splatting.render import render
from gaussian_splatting.trainer import Trainer
from gaussian_splatting.utils.general import PILtoTorch, safe_state
from gaussian_splatting.utils.graphics import BasicPointCloud
from gaussian_splatting.utils.loss import PhotometricLoss


class LocalInitializationTrainer(Trainer):
    def __init__(self, image, sh_degree: int = 3, iterations: int = 10000):
        DPT = self._load_DPT()
        depth_estimation = DPT(image)["predicted_depth"]

        image = PILtoTorch(image)
        initial_point_cloud = self._get_initial_point_cloud(
            image, depth_estimation, step=25
        )

        self.gaussian_model = GaussianModel(sh_degree)
        self.gaussian_model.initialize_from_point_cloud(initial_point_cloud)
        # TODO: set camera extent???

        self.optimizer = Optimizer(self.gaussian_model)
        self._photometric_loss = PhotometricLoss(lambda_dssim=0.2)

        self.camera = self._get_orthogonal_camera(image)

        # Densification and pruning
        self._opacity_reset_interval = 10001
        self._min_opacity = 0.005
        self._max_screen_size = 20
        self._percent_dense = 0.01
        self._densification_interval = 100
        self._densification_iteration_start = 500
        self._densification_iteration_stop = 15000
        self._densification_grad_threshold = 0.0002

        self._debug = True

        safe_state(seed=2234)

        self._output_path =Path(" artifacts/local/init/")
        self._output_path.mkdir(exist_ok=True, parents=True)


    def run(self, iterations: int = 3000):
        progress_bar = tqdm(range(iterations), desc="Initialization")

        best_loss, best_iteration, losses = None, 0, []
        for iteration in range(iterations):
            self.optimizer.update_learning_rate(iteration)
            rendered_image, viewspace_point_tensor, visibility_filter, radii = render(
                self.camera, self.gaussian_model
            )

            gt_image = self.camera.original_image.cuda()
            loss = self._photometric_loss(rendered_image, gt_image)
            loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

            if best_loss is None or best_loss > loss:
                best_loss = loss.cpu().item()
                best_iteration = iteration
            losses.append(loss.cpu().item())

            if iteration % 100 == 0:
                self._save_artifacts(self, losses, rendered_image, iteration)

            with torch.no_grad():
                # Densification
                if iteration < self._densification_iteration_stop:
                    self.gaussian_model.update_stats(
                        viewspace_point_tensor, visibility_filter, radii
                    )

                    if (
                        iteration >= self._densification_iteration_start
                        and iteration % self._densification_interval == 0
                    ):
                        self._densify_and_prune(
                            iteration > self._opacity_reset_interval
                        )

                # Reset opacity interval
                if iteration > 0 and iteration % self._opacity_reset_interval == 0:
                    print("Reset Opacity")
                    self._reset_opacity()

            progress_bar.set_postfix(
                {
                    "Loss": f"{loss:.{5}f}",
                    "Num_visible": f"{visibility_filter.int().sum().item()}/{len(visibility_filter)}",
                }
            )
            progress_bar.update(1)

        progress_bar.close()
        print(
            f"Training done. Best loss = {best_loss:.{5}f} at iteration {best_iteration}."
        )

        torchvision.utils.save_image(gt_image, self._output_path / "gt.png")

    def _get_orthogonal_camera(self, image):
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

    def _get_initial_point_cloud(self, frame, depth_estimation, step: int = 50):
        # Frame and depth_estimation width do not exactly match.
        _, w, h = depth_estimation.shape

        _min_depth = depth_estimation.min()
        _max_depth = depth_estimation.max()

        half_step = step // 2
        points, colors, normals = [], [], []
        for x in range(step, w - step, step):
            for y in range(step, h - step, step):
                _depth = depth_estimation[0, x, y].item()
                # Normalized points
                points.append(
                    [y / h, x / w, (_depth - _min_depth) / (_max_depth - _min_depth)]
                )
                # Average RGB color in the window color around selected pixel
                colors.append(
                    frame[
                        :, x - half_step : x + half_step, y - half_step : y + half_step
                    ]
                    .mean(axis=[1, 2])
                    .tolist()
                )
                normals.append(
                    [
                        0.0,
                        0.0,
                        0.0,
                    ]
                )

        point_cloud = BasicPointCloud(
            points=np.array(points),
            colors=np.array(colors),
            normals=np.array(normals),
        )

        return point_cloud

    def _load_DPT(self):
        checkpoint = "vinvino02/glpn-nyu"
        depth_estimator = pipeline("depth-estimation", model=checkpoint)

        return depth_estimator

    def _save_artifacts(self, losses, rendered_image, iteration):
        plt.cla()
        plt.plot(losses)
        plt.yscale("log")
        plt.savefig(self._output_path / "losses.png")

        torchvision.utils.save_image(
            rendered_image, self._output_path / f"rendered_{iteration}.png"
        )
