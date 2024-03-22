import numpy as np
from transformers import pipeline
from tqdm import tqdm

import torch
import torchvision

from gaussian_splatting.model import GaussianModel
from gaussian_splatting.optimizer import Optimizer
from gaussian_splatting.render import render
from gaussian_splatting.utils.graphics import BasicPointCloud
from gaussian_splatting.utils.general import PILtoTorch
from gaussian_splatting.dataset.cameras import Camera
from gaussian_splatting.utils.loss import l1_loss, ssim
from gaussian_splatting.trainer import Trainer


class LocalTrainer(Trainer):
    def __init__(self, image, sh_degree: int = 3):
        DPT = self._load_DPT()
        depth_estimation = DPT(image)["predicted_depth"]

        image = PILtoTorch(image)

        initial_point_cloud = self._get_initial_point_cloud(image, depth_estimation)

        self.gaussian_model = GaussianModel(sh_degree)
        self.gaussian_model.initialize_from_point_cloud(initial_point_cloud)
        # TODO: set camera extent???

        self.optimizer = Optimizer(self.gaussian_model)

        self._camera = self._get_orthogonal_camera(image)

        self._iterations = 10000
        self._lambda_dssim = 0.2

        self._opacity_reset_interval = 3000
        self._min_opacity = 0.005
        self._max_screen_size = 20
        self._percent_dense = 0.01
        self._densification_interval = 100
        self._densification_iteration_start = 500
        self._densification_iteration_stop = 15000
        self._densification_grad_threshold = 0.0002

        self._debug = True

    def run(self):
        progress_bar = tqdm(
            range(self._iterations), desc="Training progress"
        )
        for iteration in range(self._iterations):
            self.optimizer.update_learning_rate(iteration)
            rendered_image, viewspace_point_tensor, visibility_filter, radii = render(
                self._camera, self.gaussian_model
            )

            if iteration == 0:
                torchvision.utils.save_image(rendered_image, f"rendered_{iteration}.png")

            gt_image = self._camera.original_image.cuda()
            Ll1 = l1_loss(rendered_image, gt_image)
            loss = (1.0 - self._lambda_dssim) * Ll1 + self._lambda_dssim * (
                1.0 - ssim(rendered_image, gt_image)
            )

            loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

            with torch.no_grad():
                # Densification
                if iteration < self._densification_iteration_stop:
                    self.gaussian_model.update_stats(
                        viewspace_point_tensor, visibility_filter, radii
                    )

            #        if (
            #            iteration >= self._densification_iteration_start
            #            and iteration % self._densification_interval == 0
            #        ):
            #            self._densify_and_prune(
            #                iteration > self._opacity_reset_interval
            #            )

                # Reset opacity interval
            #    if iteration % self._opacity_reset_interval == 0:
            #        self._reset_opacity()

            progress_bar.set_postfix({"Loss": f"{loss:.{5}f}"})
            progress_bar.update(1)

        torchvision.utils.save_image(rendered_image, f"rendered_{iteration}.png")
        torchvision.utils.save_image(gt_image, f"gt.png")

    def _get_orthogonal_camera(self, image):
        camera = Camera(
            R=np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]),
            T=np.array([0.5, 0.5, -1]),
            FoVx=1.,
            FoVy=1.,
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

        half_step = step // 2
        points, colors, normals = [], [], []
        for x in range(step, w - step, step):
            for y in range(step, h - step, step):
                # Normalized h, w
                points.append([
                    x / w,
                    y / h,
                    depth_estimation[0, x, y].item()
                ])
                # Average RGB color in the window color around selected pixel
                colors.append(
                    frame[
                        :,
                        x - half_step: x + half_step,
                        y - half_step: y + half_step
                    ].mean(axis=[1, 2]).tolist()
                )
                normals.append([0., 0., 0.,])

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


