from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt
from torchvision.utils import save_image
from transformers import pipeline

from gaussian_splatting.model import GaussianModel
from gaussian_splatting.optimizer import Optimizer
from gaussian_splatting.pose_free.transformation_model import \
    AffineTransformationModel
from gaussian_splatting.render import render
from gaussian_splatting.utils.early_stopper import EarlyStopper
from gaussian_splatting.utils.general import TorchToPIL, safe_state
from gaussian_splatting.utils.graphics import BasicPointCloud
from gaussian_splatting.utils.loss import PhotometricLoss


class LocalTrainer:
    def __init__(
        self,
        sh_degree: int = 3,
        init_iterations: int = 250,
        transfo_iterations: int = 250,
    ):
        self._depth_estimator = self._load_depth_estimator()
        self._point_cloud_step = 25
        self._sh_degree = sh_degree

        self._photometric_loss = PhotometricLoss(lambda_dssim=0.2)

        self._init_iterations = init_iterations
        self._init_early_stopper = EarlyStopper(patience=10)
        self._init_save_artifacts_iterations = 50

        self._transfo_lr = 0.0001
        self._transfo_iterations = transfo_iterations
        self._transfo_early_stopper = EarlyStopper(patience=10)
        self._transfo_save_artifacts_iterations = 100

        self._debug = True

        self._output_path = Path("artifacts/local/")
        self._output_path.mkdir(exist_ok=True, parents=True)

        safe_state(seed=2234)

    def run_init(self, image, camera, run_id: int = 0):
        output_path = self._output_path / "init"
        output_path.mkdir(exist_ok=True, parents=True)

        gaussian_model = self._get_initial_gaussian_model(image)
        optimizer = Optimizer(gaussian_model)

        image = image.cuda()
        losses = []
        for iteration in range(self._init_iterations):
            optimizer.update_learning_rate(iteration)

            rendered_image, _, _, _ = render(camera, gaussian_model)

            loss = self._photometric_loss(rendered_image, image)
            loss.backward()
            loss_value = loss.cpu().item()
            losses.append(loss_value)

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            if self._init_early_stopper.step(loss_value):
                self._init_early_stopper.print_early_stop()
                break

            if (
                self._debug
                or iteration % self._init_save_artifacts_iterations == 0
                or iteration == self._init_iterations - 1
            ):
                self._save_artifacts(
                    losses, rendered_image, output_path / str(run_id), iteration
                )

        if self._debug:
            save_image(image, output_path / f"{run_id}_ground_truth.png")

        return gaussian_model

    def run_transfo(self, image, camera, gaussian_model, run_id: int = 0):
        output_path = self._output_path / "transfo"
        output_path.mkdir(exist_ok=True, parents=True)

        transformation_model = AffineTransformationModel()
        optimizer = torch.optim.Adam(
            transformation_model.parameters(), lr=self._transfo_lr
        )

        image = image.cuda()
        transformation_model = transformation_model.cuda()

        losses = []
        initial_xyz = gaussian_model.get_xyz.detach()
        for iteration in range(self._transfo_iterations):
            xyz = transformation_model(initial_xyz)
            gaussian_model.set_optimizable_tensors({"xyz": xyz})

            rendered_image, _, _, _ = render(camera, gaussian_model)

            loss = self._photometric_loss(rendered_image, image)
            loss.backward()
            loss_value = loss.cpu().item()
            losses.append(loss_value)

            optimizer.step()

            if self._transfo_early_stopper.step(loss_value):
                self._transfo_early_stopper.print_early_stop()
                transformation = self._transfo_early_stopper.get_best_params(
                    transformation
                )
                break
            else:
                transformation = transformation_model.transformation
                self._init_early_stopper.set_best_params(transformation)

            if (
                self._debug
                or iteration % self._transfo_save_artifacts_iterations == 0
                or iteration == self._transfo_iterations - 1
            ):
                self._save_artifacts(
                    losses,
                    rendered_image,
                    output_path / str(run_id),
                    iteration,
                )

        if self._debug:
            save_image(image, output_path / f"{run_id}_ground_truth.png")

        return transformation

    def _get_initial_gaussian_model(self, image):
        PIL_image = TorchToPIL(image)

        depth_estimation = self._depth_estimator(PIL_image)["predicted_depth"]
        point_cloud = self._get_initial_point_cloud_from_depth_estimation(
            image, depth_estimation, step=self._point_cloud_step
        )

        gaussian_model = GaussianModel(self._sh_degree)
        gaussian_model.initialize_from_point_cloud(point_cloud)

        return gaussian_model

    def _get_initial_point_cloud_from_depth_estimation(
        self, frame, depth_estimation, step: int = 50
    ):
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

    def _load_depth_estimator(self):
        checkpoint = "vinvino02/glpn-nyu"
        depth_estimator = pipeline("depth-estimation", model=checkpoint)

        return depth_estimator

    def _save_artifacts(self, losses, rendered_image, output_path, iteration):
        output_path.mkdir(exist_ok=True, parents=True)
        plt.cla()
        plt.plot(losses)
        plt.yscale("log")
        plt.savefig(output_path / "losses.png")

        save_image(rendered_image, self._output_path / f"rendered_{iteration}.png")
