import copy
from pathlib import Path

import numpy as np
import torch
import torchvision
from tqdm import tqdm

from gaussian_splatting.colmap_free.global_trainer import GlobalTrainer
from gaussian_splatting.colmap_free.local_initialization_trainer import \
    LocalInitializationTrainer
from gaussian_splatting.colmap_free.local_transformation_trainer import \
    LocalTransformationTrainer
from gaussian_splatting.dataset.cameras import Camera
from gaussian_splatting.dataset.image_dataset import ImageDataset
from gaussian_splatting.render import render
from gaussian_splatting.utils.general import PILtoTorch
from gaussian_splatting.utils.loss import PhotometricLoss


def main():
    debug = True
    iteration_step_size = 50
    initialization_iterations = 250
    transformation_iterations = 250
    global_iterations = 5

    photometric_loss = PhotometricLoss(lambda_dssim=0.2)
    dataset = ImageDataset(images_path=Path("data/phil/1/input/"))

    # Initialize Local3DGS gaussians
    current_image = dataset.get_frame(0)
    local_initialization_trainer = LocalInitializationTrainer(current_image)
    local_initialization_trainer.run(iterations=initialization_iterations)

    # We set a copy of the initialized model to both the local transformation and the
    # global models.
    next_gaussian_model = local_initialization_trainer.gaussian_model
    local_transformation_trainer = LocalTransformationTrainer(next_gaussian_model)
    # global_trainer = GlobalTrainer(copy.deepcopy(next_gaussian_model))

    current_camera = local_initialization_trainer.camera
    for iteration in tqdm(
        range(iteration_step_size, len(dataset), iteration_step_size)
    ):
        # Keep a copy of current gaussians to compare
        with torch.no_grad():
            current_gaussian_model = copy.deepcopy(next_gaussian_model)

        # Find transformation from current to next camera poses
        next_image = PILtoTorch(dataset.get_frame(iteration))
        rotation, translation = local_transformation_trainer.run(
            current_camera,
            next_image,
            iterations=transformation_iterations,
            run=iteration,
        )

        # Add new camera to Global3DGS training cameras
        next_camera = Camera(
            R=np.matmul(current_camera.R, rotation),
            T=current_camera.T + translation,
            FoVx=current_camera.FoVx,
            FoVy=current_camera.FoVy,
            image=next_image,
            gt_alpha_mask=None,
            image_name="patate",
            colmap_id=iteration,
            uid=iteration,
        )

        if debug:
            # Save artifact
            next_camera_image, _, _, _ = render(next_camera, current_gaussian_model)
            next_gaussian_image, _, _, _ = render(current_camera, next_gaussian_model)
            loss = photometric_loss(next_camera_image, next_gaussian_image)

            print(loss)
            torchvision.utils.save_image(
                next_camera_image, f"artifacts/global/next_camera_{iteration}.png"
            )
            torchvision.utils.save_image(
                next_gaussian_image, f"artifacts/global/next_gaussian_{iteration}.png"
            )

        break
        # global_trainer.add_camera(next_camera)
        # global_trainer.run(global_iterations)

        current_camera = next_camera


if __name__ == "__main__":
    main()
