import torch
import torchvision
from matplotlib import pyplot as plt
from tqdm import tqdm

from gaussian_splatting.render import render
from gaussian_splatting.trainer import Trainer
from gaussian_splatting.utils.general import PILtoTorch, safe_state
from gaussian_splatting.utils.loss import l1_loss, ssim


class TransformationModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(in_features=3, out_features=3)

        torch.nn.init.eye_(self.linear.weight.data)
        torch.nn.init.zeros_(self.linear.bias.data)

    def forward(self, xyz):
        transformed_xyz = self.linear(xyz)

        return transformed_xyz


class LocalTransformationTrainer(Trainer):
    def __init__(self, image, camera, gaussian_model):
        self.camera = camera
        self.gaussian_model = gaussian_model

        self.xyz = gaussian_model.get_xyz.detach()

        self.transformation_model = TransformationModel()
        self.transformation_model.to(self.xyz.device)

        self.image = PILtoTorch(image).to(self.xyz.device)

        self.optimizer = torch.optim.Adam(
            self.transformation_model.parameters(), lr=0.0001
        )

        self._iterations = 101
        self._lambda_dssim = 0.2

        safe_state(seed=2234)

    def run(self):
        progress_bar = tqdm(range(self._iterations), desc="Transformation")

        best_loss, best_iteration, losses = None, 0, []
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
            losses.append(loss.cpu().item())

            loss.backward()

            self.optimizer.step()

            progress_bar.set_postfix(
                {
                    "Loss": f"{loss:.{5}f}",
                }
            )
            progress_bar.update(1)

        print(f"Training done. Best loss = {best_loss} at iteration {best_iteration}.")

        torchvision.utils.save_image(
            rendered_image, f"artifacts/local/transfo/rendered_{iteration}.png"
        )
        torchvision.utils.save_image(gt_image, f"artifacts/local/transfo/gt.png")
