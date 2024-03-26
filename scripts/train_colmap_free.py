import copy
from pathlib import Path

import numpy as np
import torchvision

from gaussian_splatting.dataset.cameras import Camera
from gaussian_splatting.dataset.image_dataset import ImageDataset
from gaussian_splatting.global_trainer import GlobalTrainer
from gaussian_splatting.local_initialization_trainer import \
    LocalInitializationTrainer
from gaussian_splatting.local_transformation_trainer import \
    LocalTransformationTrainer
from gaussian_splatting.render import render
from gaussian_splatting.utils.general import PILtoTorch


def main():
    dataset = ImageDataset(images_path=Path("data/phil/1/input/"))
    first_image = dataset.get_frame(0)

    local_initialization_trainer = LocalInitializationTrainer(
        first_image, iterations=500
    )
    local_initialization_trainer.run()

    step = 5
    gaussian_model = local_initialization_trainer.gaussian_model
    camera = local_initialization_trainer.camera
    global_trainer = GlobalTrainer(
        gaussian_model=gaussian_model, cameras=[camera], iterations=100
    )
    for iteration in range(step, len(dataset), step):
        next_image = dataset.get_frame(iteration)
        local_transformation_trainer = LocalTransformationTrainer(
            next_image,
            camera=camera,
            gaussian_model=copy.deepcopy(gaussian_model),
            iterations=250,
        )
        local_transformation_trainer.run()
        rotation, translation = local_transformation_trainer.get_affine_transformation()

        next_image = PILtoTorch(next_image)
        next_camera = Camera(
            R=np.matmul(camera.R, rotation),
            T=camera.T + translation,
            FoVx=camera.FoVx,
            FoVy=camera.FoVy,
            image=next_image,
            gt_alpha_mask=None,
            image_name="patate",
            colmap_id=iteration,
            uid=iteration,
        )
        global_trainer.add_camera(next_camera)
        global_trainer.run()

        rendered_image, viewspace_point_tensor, visibility_filter, radii = render(
            next_camera,
            local_initialization_trainer.gaussian_model,
        )
        torchvision.utils.save_image(
            rendered_image, f"artifacts/global/rendered_{iteration}.png"
        )
        torchvision.utils.save_image(next_image, f"artifacts/global/gt_{iteration}.png")

        print(f">>> Iteration {iteration} / {len(dataset) // step}")


if __name__ == "__main__":
    main()
