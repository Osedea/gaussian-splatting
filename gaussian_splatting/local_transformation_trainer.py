import torch
import torch.nn as nn
import torchvision
from matplotlib import pyplot as plt
from tqdm import tqdm

from gaussian_splatting.render import render
from gaussian_splatting.trainer import Trainer
from gaussian_splatting.utils.general import PILtoTorch, safe_state
from gaussian_splatting.utils.loss import l1_loss, ssim


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
    def __init__(self, image, camera, gaussian_model, iterations: int = 100):
        self.camera = camera
        self.gaussian_model = gaussian_model

        self.xyz = gaussian_model.get_xyz.detach()

        self.transformation_model = TransformationModel()
        self.transformation_model.to(self.xyz.device)

        self.image = PILtoTorch(image).to(self.xyz.device)

        self.optimizer = torch.optim.Adam(
            self.transformation_model.parameters(), lr=0.0001
        )

        self._iterations = iterations
        self._lambda_dssim = 0.2

        safe_state(seed=2234)

    def run(self):
        progress_bar = tqdm(range(self._iterations), desc="Transformation")

        best_loss, best_iteration, losses = None, 0, []
        best_xyz = None
        for iteration in range(self._iterations):
            xyz = self.transformation_model(self.xyz)
            self.gaussian_model.set_optimizable_tensors({"xyz": xyz})

            rendered_image, viewspace_point_tensor, visibility_filter, radii = render(
                self.camera, self.gaussian_model
            )

            if iteration % 10 == 0:
                plt.cla()
                plt.plot(losses)
                plt.yscale("log")
                plt.savefig("artifacts/local/transfo/losses.png")

                torchvision.utils.save_image(
                    rendered_image, f"artifacts/local/transfo/rendered_{iteration}.png"
                )

            gt_image = self.image
            Ll1 = l1_loss(rendered_image, gt_image)
            loss = (1.0 - self._lambda_dssim) * Ll1 + self._lambda_dssim * (
                1.0 - ssim(rendered_image, gt_image)
            )
            if best_loss is None or best_loss > loss:
                best_loss = loss.cpu().item()
                best_iteration = iteration
                best_xyz = xyz
            losses.append(loss.cpu().item())

            loss.backward()

            self.optimizer.step()

            progress_bar.set_postfix(
                {
                    "Loss": f"{loss:.{5}f}",
                }
            )
            progress_bar.update(1)

        progress_bar.close()
        print(
            f"Training done. Best loss = {best_loss:.{5}f} at iteration {best_iteration}."
        )
        self.gaussian_model.set_optimizable_tensors({"xyz": best_xyz})

        torchvision.utils.save_image(
            rendered_image, f"artifacts/local/transfo/rendered_best.png"
        )
        torchvision.utils.save_image(gt_image, f"artifacts/local/transfo/gt.png")

    def get_affine_transformation(self):
        rotation = self.transformation_model.rotation.numpy()
        translation = self.transformation_model.translation.numpy()

        return rotation, translation
