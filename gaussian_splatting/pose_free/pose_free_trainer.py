import copy
from pathlib import Path

import torchvision
from tqdm import tqdm

from gaussian_splatting.dataset.image_dataset import ImageDataset
from gaussian_splatting.pose_free.global_trainer import GlobalTrainer
from gaussian_splatting.pose_free.local_trainer import LocalTrainer
from gaussian_splatting.render import render
from gaussian_splatting.utils.camera import (get_orthogonal_camera,
                                             transform_camera)
from gaussian_splatting.utils.loss import PhotometricLoss


class PoseFreeTrainer:
    def __init__(self, source_path: Path):
        self._debug = True

        self._initialization_iterations = 1000
        self._transformation_iterations = 250
        self._global_iterations = 50

        self._photometric_loss = PhotometricLoss(lambda_dssim=0.2)
        self._dataset = ImageDataset(images_path=source_path, step_size=5)

        self._output_path = Path("artifacts/global")
        self._output_path.mkdir(exist_ok=True, parents=True)

        self._local_trainer = LocalTrainer(init_iterations=25, transfo_iterations=25)

    def run(self):
        current_image = self._dataset.get_frame(0)
        current_camera = get_orthogonal_camera(current_image)
        global_trainer = self._initialize_global_trainer(current_image, current_camera)

        for i in tqdm(range(1, len(self._dataset))):
            next_image = self._dataset.get_frame(i)

            gaussian_model = self._local_trainer.run_init(
                current_image, current_camera, run_id=i
            )
            rotation, translation = self._local_trainer.run_transfo(
                next_image,
                current_camera,
                gaussian_model,
                run_id=i,
            )
            next_camera = transform_camera(
                current_camera, rotation, translation, next_image, _id=i
            )
            global_trainer.add_camera(next_camera)

            current_image = next_image
            current_camera = next_camera

        global_trainer.run(self._global_iterations)

    def _initialize_global_trainer(self, initial_image, initial_camera):
        initial_gaussian_model = self._local_trainer.run_init(
            initial_image, initial_camera, run_id=0
        )

        global_gaussian_model = copy.deepcopy(initial_gaussian_model)
        global_trainer = GlobalTrainer(global_gaussian_model)

        return global_trainer

    def _save_artifacts(
        self,
        current_camera,
        current_gaussian_model,
        current_image,
        next_camera,
        next_gaussian_model,
        next_image,
        iteration,
    ):
        current_camera_image, _, _, _ = render(current_camera, current_gaussian_model)
        next_camera_image, _, _, _ = render(next_camera, current_gaussian_model)
        next_gaussian_image, _, _, _ = render(current_camera, next_gaussian_model)

        for image, filename in [
            (current_camera_image, f"{iteration}_current_camera.png"),
            (next_camera_image, f"{iteration}_next_camera.png"),
            (next_gaussian_image, f"{iteration}_next_gaussian.png"),
            (current_image, f"{iteration}_current_image.png"),
            (next_image, f"{iteration}_next_image.png"),
        ]:
            torchvision.utils.save_image(image, self._output_path / filename)
