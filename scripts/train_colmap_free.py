import copy
import math
from pathlib import Path

import numpy as np
import torchvision

from gaussian_splatting.colmap_free.global_trainer import GlobalTrainer
from gaussian_splatting.colmap_free.local_initialization_trainer import \
    LocalInitializationTrainer
from gaussian_splatting.colmap_free.local_transformation_trainer import \
    LocalTransformationTrainer
from gaussian_splatting.dataset.cameras import Camera
from gaussian_splatting.dataset.image_dataset import ImageDataset
from gaussian_splatting.render import render
from gaussian_splatting.utils.loss import PhotometricLoss


def main():
    debug = True
    step_size = 10
    initialization_iterations = 1000
    transformation_iterations = 250
    global_iterations = 50

    photometric_loss = PhotometricLoss(lambda_dssim=0.2)
    dataset = ImageDataset(images_path=Path("data/phil/1/input/"))

    global_trainer = None
    for i in range(len(dataset) // step_size):
        print(f">>> Current: {i * step_size} / Next: {(i + 1) * step_size}")

        current_image = dataset.get_frame(i * step_size)
        next_image = dataset.get_frame((i + 1) * step_size)

        if i == 0:
            current_camera = _get_orthogonal_camera(current_image)
        else:
            current_camera = next_camera

        current_camera, current_gaussian_model = _initialize_local_gaussian_model(
            current_image, current_camera, iterations=initialization_iterations, run=i
        )

        if i == 0:
            global_gaussian_model = copy.deepcopy(current_gaussian_model)
            global_trainer = GlobalTrainer(global_gaussian_model)

        if debug:
            current_gaussian_model_copy = copy.deepcopy(current_gaussian_model)

        next_camera, next_gaussian_model = _transform_local_gaussian_model(
            next_image, current_camera, current_gaussian_model,
            iterations=transformation_iterations, run=i
        )

        global_trainer.add_camera(next_camera)

        if debug:
            save_artifacts(
                current_camera,
                current_gaussian_model_copy,
                current_image,
                next_camera,
                next_gaussian_model,
                next_image,
                i,
            )

        global_trainer.run((i + 1) * global_iterations)

    #global_trainer.run(global_iterations)

def _initialize_local_gaussian_model(
    image, camera, iterations: int = 250, run: int = 0
):
    local_initialization_trainer = LocalInitializationTrainer(image, camera)
    local_initialization_trainer.run(iterations=iterations)

    current_camera = local_initialization_trainer.camera
    gaussian_model = local_initialization_trainer.gaussian_model

    return camera, gaussian_model


def _transform_local_gaussian_model(
    image, camera, gaussian_model, iterations: int = 250, run: int = 0
):
    local_transformation_trainer = LocalTransformationTrainer(gaussian_model)
    rotation, translation = local_transformation_trainer.run(
        camera,
        image,
        iterations=iterations,
        run=run,
    )

    transformed_camera = _transform_camera(camera, rotation, translation, image, run)
    transformed_gaussian_model = local_transformation_trainer.gaussian_model

    return transformed_camera, transformed_gaussian_model


def _get_orthogonal_camera(image):
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


def _transform_camera(camera, rotation, translation, image, _id, image_name=""):
    transformed_camera = Camera(
        R=np.matmul(camera.R, rotation),
        T=(camera.T + translation),
        FoVx=camera.FoVx,
        FoVy=camera.FoVy,
        image=image,
        gt_alpha_mask=None,
        image_name=image_name,
        colmap_id=_id,
        uid=_id,
    )
    return transformed_camera


def save_artifacts(
    current_camera,
    current_gaussian_model,
    current_image,
    next_camera,
    next_gaussian_model,
    next_image,
    iteration,
):
    # Save artifact
    current_camera_image, _, _, _ = render(current_camera, current_gaussian_model)
    next_camera_image, _, _, _ = render(next_camera, current_gaussian_model)
    next_gaussian_image, _, _, _ = render(current_camera, next_gaussian_model)

    output_path = Path("artifacts/global")
    output_path.mkdir(exist_ok=True, parents=True)
    torchvision.utils.save_image(
        current_camera_image, output_path / f"{iteration}_current_camera.png"
    )
    torchvision.utils.save_image(
        next_camera_image, output_path / f"{iteration}_next_camera.png"
    )
    torchvision.utils.save_image(
        next_gaussian_image, output_path / f"{iteration}_next_gaussian.png"
    )
    torchvision.utils.save_image(
        current_image, output_path / f"{iteration}_current_image.png"
    )
    torchvision.utils.save_image(
        next_image, output_path / f"{iteration}_next_image.png"
    )


if __name__ == "__main__":
    main()
