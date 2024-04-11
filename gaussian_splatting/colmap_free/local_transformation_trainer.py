from pathlib import Path

import torch
import torchvision
from matplotlib import pyplot as plt
from tqdm import tqdm

from gaussian_splatting.colmap_free.transformation_model import \
    AffineTransformationModel
from gaussian_splatting.render import render
from gaussian_splatting.trainer import Trainer
from gaussian_splatting.utils.general import safe_state
from gaussian_splatting.utils.loss import PhotometricLoss


class LocalTransformationTrainer(Trainer):
    def __init__(self, gaussian_model):
        self.gaussian_model = gaussian_model

        self.transformation_model = AffineTransformationModel()
        self.transformation_model.to(gaussian_model.get_xyz.device)

        self.optimizer = torch.optim.Adam(
            self.transformation_model.parameters(), lr=0.0001
        )
        self._photometric_loss = PhotometricLoss(lambda_dssim=0.2)

        self._output_path = Path("artifacts/local/transfo/")
        self._output_path.mkdir(exist_ok=True, parents=True)

        safe_state(seed=2234)

    def run(self, current_camera, gt_image, iterations: int = 1000, run: int = 0):
        gt_image = gt_image.to(self.gaussian_model.get_xyz.device)

        progress_bar = tqdm(range(iterations), desc="Transformation")

        losses = []
        best_loss, best_iteration, best_xyz = None, 0, None
        best_rotation, best_translation = None, None
        patience = 0
        initial_xyz = self.gaussian_model.get_xyz.detach()
        for iteration in range(iterations):
            xyz = self.transformation_model(initial_xyz)
            self.gaussian_model.set_optimizable_tensors({"xyz": xyz})

            rendered_image, viewspace_point_tensor, visibility_filter, radii = render(
                current_camera, self.gaussian_model
            )

            loss = self._photometric_loss(rendered_image, gt_image)
            loss.backward()
            self.optimizer.step()

            losses.append(loss.cpu().item())

            progress_bar.set_postfix({"Loss": f"{loss:.{5}f}"})
            progress_bar.update(1)

            if iteration % 10 == 0 or iteration == iterations - 1:
                self._save_artifacts(losses, rendered_image, iteration, run)

            if best_loss is None or best_loss > loss:
                best_loss = loss.cpu().item()
                best_iteration = iteration
                best_xyz = xyz.detach()

                best_rotation = self.transformation_model.rotation.numpy()
                best_translation = self.transformation_model.translation.numpy()

            # elif best_loss < loss and patience > 10:
            #    self._save_artifacts(losses, rendered_image, iteration, run)
            #    break
            # else:
            #    patience += 1

        progress_bar.close()

        torchvision.utils.save_image(gt_image, self._output_path / "gt.png")

        print(f"Best loss = {best_loss:.{5}f} at iteration {best_iteration}.")
        self.gaussian_model.set_optimizable_tensors({"xyz": best_xyz})

        if best_rotation is None or best_translation is None:
            best_rotation = self.transformation_model.rotation.numpy()
            best_translation = self.transformation_model.translation.numpy()

        return best_rotation, best_translation

    def _save_artifacts(self, losses, rendered_image, iteration, run):
        plt.cla()
        plt.plot(losses)
        plt.yscale("log")
        plt.savefig(self._output_path / "losses.png")

        torchvision.utils.save_image(
            rendered_image, self._output_path / f"{run}_rendered_{iteration}.png"
        )
