from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt
from torchvision.utils import save_image

from gaussian_splatting.model import GaussianModel
from gaussian_splatting.optimizer import Optimizer
from gaussian_splatting.pose_free.depth_estimator import DepthEstimator
from gaussian_splatting.pose_free.transformation_model import \
    AffineTransformationModel
from gaussian_splatting.render import render
from gaussian_splatting.utils.early_stopper import EarlyStopper
from gaussian_splatting.utils.general import safe_state
from gaussian_splatting.utils.graphics import BasicPointCloud
from gaussian_splatting.utils.loss import PhotometricLoss


class LocalTrainer:
    def __init__(
        self,
        sh_degree: int = 3,
        init_iterations: int = 1000,
        transfo_iterations: int = 1000,
        debug: bool = False,
    ):
        self._depth_estimator = DepthEstimator()
        self._point_cloud_step = 25
        self._sh_degree = sh_degree

        self._photometric_loss = PhotometricLoss(lambda_dssim=0.2)

        self._init_iterations = init_iterations
        self._init_early_stopper = EarlyStopper(patience=10)
        self._init_save_artifacts_iterations = 100

        self._transfo_lr = 0.00001
        self._transfo_iterations = transfo_iterations
        self._transfo_early_stopper = EarlyStopper(
            patience=100,
        )
        self._transfo_save_artifacts_iterations = 10

        self._debug = debug

        self._output_path = Path("artifacts/local/")

        safe_state(seed=2234)

    def run_init(self, image, camera, progress_bar=None, run_id: int = 0):
        output_path = self._output_path / "init" / str(run_id)
        output_path.mkdir(exist_ok=True, parents=True)

        gaussian_model = self.get_initial_gaussian_model(image, output_path)
        optimizer = Optimizer(gaussian_model)
        self._init_early_stopper.reset()

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
                break

            if self._debug and (iteration % self._init_save_artifacts_iterations == 0):
                self._save_artifacts(losses, rendered_image, output_path, iteration)

            if progress_bar is not None:
                progress_bar.set_postfix(
                    {
                        "stage": "init",
                        "iteration": f"{iteration}/{self._init_iterations}",
                        "loss": f"{loss_value:.5f}",
                    }
                )

        if self._debug:
            self._save_artifacts(losses, rendered_image, output_path, "best")
            save_image(image, output_path / "ground_truth.png")

        return gaussian_model

    def run_transfo(
        self, image, camera, gaussian_model, progress_bar=None, run_id: int = 0
    ):
        output_path = self._output_path / "transfo" / str(run_id)
        output_path.mkdir(exist_ok=True, parents=True)

        transformation_model = AffineTransformationModel()
        optimizer = torch.optim.Adam(
            transformation_model.parameters(), lr=self._transfo_lr
        )
        self._transfo_early_stopper.reset()

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
                transformation = self._transfo_early_stopper.get_best_params()
                break
            else:
                transformation = transformation_model.transformation
                self._transfo_early_stopper.set_best_params(transformation)

            if self._debug and (
                iteration % self._transfo_save_artifacts_iterations == 0
            ):
                self._save_artifacts(
                    losses,
                    rendered_image,
                    output_path,
                    iteration,
                )

            if progress_bar is not None:
                progress_bar.set_postfix(
                    {
                        "stage": "transfo",
                        "iteration": f"{iteration}/{self._transfo_iterations}",
                        "loss": f"{loss_value:.5f}",
                    }
                )

        if self._debug:
            self._save_artifacts(losses, rendered_image, output_path, "best")
            save_image(image, output_path / f"ground_truth.png")

        return transformation

    def get_initial_gaussian_model(self, image, output_folder: Path = None):
        depth_estimation = self._depth_estimator.run(image)
        if self._debug and output_folder is not None:
            save_image(
                depth_estimation,
                output_folder / f"depth_estimation.png",
            )

        point_cloud = self._get_initial_point_cloud_from_depth_estimation(
            image, depth_estimation, step=self._point_cloud_step
        )

        gaussian_model = GaussianModel(self._sh_degree)
        gaussian_model.initialize_from_point_cloud(point_cloud)

        return gaussian_model

    def _get_initial_point_cloud_from_depth_estimation(
        self, frame, depth_estimation, step: int = 50
    ):
        w, h = depth_estimation.shape
        half_step = step // 2
        points, colors, normals = [], [], []
        for x in range(step, w - step, step):
            for y in range(step, h - step, step):
                _depth = depth_estimation[x, y].item()
                # Normalized points
                points.append([y / h, x / w, _depth])
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

    def _save_artifacts(self, losses, rendered_image, output_path, iteration):
        plt.cla()
        plt.plot(losses)
        plt.yscale("log")
        plt.savefig(output_path / "losses.png")

        save_image(rendered_image, output_path / f"rendered_{iteration}.png")
