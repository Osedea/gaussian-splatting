import torch
import torch.nn as nn
import torchvision
from matplotlib import pyplot as plt
from tqdm import tqdm

from gaussian_splatting.render import render
from gaussian_splatting.trainer import Trainer
from gaussian_splatting.utils.general import safe_state
from gaussian_splatting.utils.loss import PhotometricLoss


class QuaternionRotation(nn.Module):
    def __init__(self):
        super().__init__()
        self.quaternion = nn.Parameter(torch.Tensor([[0, 0, 0, 1]]))

    def forward(self, input_tensor):
        rotation_matrix = self.get_rotation_matrix()
        rotated_tensor = torch.matmul(input_tensor, rotation_matrix)

        return rotated_tensor

    def get_rotation_matrix(self):
        # Normalize quaternion to ensure unit magnitude
        quaternion_norm = torch.norm(self.quaternion, p=2, dim=1, keepdim=True)
        normalized_quaternion = self.quaternion / quaternion_norm

        x, y, z, w = normalized_quaternion[0]
        rotation_matrix = torch.zeros(
            3, 3, dtype=torch.float32, device=self.quaternion.device
        )
        rotation_matrix[0, 0] = 1 - 2 * (y**2 + z**2)
        rotation_matrix[0, 1] = 2 * (x * y - w * z)
        rotation_matrix[0, 2] = 2 * (x * z + w * y)
        rotation_matrix[1, 0] = 2 * (x * y + w * z)
        rotation_matrix[1, 1] = 1 - 2 * (x**2 + z**2)
        rotation_matrix[1, 2] = 2 * (y * z - w * x)
        rotation_matrix[2, 0] = 2 * (x * z - w * y)
        rotation_matrix[2, 1] = 2 * (y * z + w * x)
        rotation_matrix[2, 2] = 1 - 2 * (x**2 + y**2)

        return rotation_matrix


class Translation(nn.Module):
    def __init__(self):
        super().__init__()
        self.translation = nn.Parameter(torch.Tensor(1, 3))
        nn.init.zeros_(self.translation)

    def forward(self, input_tensor):
        translated_tensor = torch.add(self.translation, input_tensor)

        return translated_tensor


class TransformationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self._rotation = QuaternionRotation()
        self._translation = Translation()

    def forward(self, xyz):
        transformed_xyz = self._rotation(xyz)
        transformed_xyz = self._translation(transformed_xyz)

        return transformed_xyz

    @property
    def rotation(self):
        rotation = self._rotation.get_rotation_matrix().detach().cpu()

        return rotation

    @property
    def translation(self):
        translation = self._translation.translation.detach().cpu()

        return translation


class LocalTransformationTrainer(Trainer):
    def __init__(self, gaussian_model):
        self.gaussian_model = gaussian_model

        self.transformation_model = TransformationModel()
        self.transformation_model.to(gaussian_model.get_xyz.device)

        self.optimizer = torch.optim.Adam(
            self.transformation_model.parameters(), lr=0.0001
        )
        self._photometric_loss = PhotometricLoss(lambda_dssim=0.2)

        safe_state(seed=2234)

    def run(self, current_camera, gt_image, iterations: int = 1000):
        gt_image = gt_image.to(self.gaussian_model.get_xyz.device)

        progress_bar = tqdm(range(iterations), desc="Transformation")

        losses = []
        best_loss, best_iteration, best_xyz = None, 0, None
        patience = 0
        for iteration in range(iterations):
            xyz = self.transformation_model(self.gaussian_model.get_xyz.detach())
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

            if iteration % 10 == 0:
                self._save_artifacts(losses, rendered_image, iteration)

            if best_loss is None or best_loss > loss:
                best_loss = loss.cpu().item()
                best_iteration = iteration
                best_xyz = xyz.detach()
            elif best_loss < loss and patience > 10:
                self._save_artifacts(losses, rendered_image, iteration)
                break
            else:
                patience += 1

        progress_bar.close()

        print(f"Best loss = {best_loss:.{5}f} at iteration {best_iteration}.")
        self.gaussian_model.set_optimizable_tensors({"xyz": best_xyz})

        rotation = self.transformation_model.rotation.numpy()
        translation = self.transformation_model.translation.numpy()

        return rotation, translation

    def _save_artifacts(self, losses, rendered_image, iteration):
        plt.cla()
        plt.plot(losses)
        plt.yscale("log")
        plt.savefig("artifacts/local/transfo/losses.png")

        torchvision.utils.save_image(
            rendered_image, f"artifacts/local/transfo/rendered_{iteration}.png"
        )
