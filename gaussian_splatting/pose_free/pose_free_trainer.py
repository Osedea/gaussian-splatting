from pathlib import Path

from torchvision.utils import save_image
from tqdm import tqdm

from gaussian_splatting.dataset.image_dataset import ImageDataset
from gaussian_splatting.pose_free.global_trainer import GlobalTrainer
from gaussian_splatting.pose_free.local_trainer import LocalTrainer
from gaussian_splatting.render import render
from gaussian_splatting.utils.camera import (get_orthogonal_camera,
                                             transform_camera)


class PoseFreeTrainer:
    def __init__(self, source_path: Path):
        self._debug = True

        self._dataset = ImageDataset(
            images_path=source_path, step_size=5, downscale_factor=1
        )

        self._local_trainer = LocalTrainer(
            init_iterations=1000, transfo_iterations=1000, debug=True
        )

        self._output_path = Path("artifacts/global")
        self._output_path.mkdir(exist_ok=True, parents=True)

    def run(self):
        current_image = self._dataset.get_frame(0)
        initial_gaussian_model = self._local_trainer.get_initial_gaussian_model(
            current_image, self._output_path
        )
        global_trainer = GlobalTrainer(initial_gaussian_model, iterations=1000)

        current_camera = get_orthogonal_camera(current_image)

        progress_bar = tqdm(range(1, len(self._dataset)))
        for i in progress_bar:
            next_image = self._dataset.get_frame(i)

            gaussian_model = self._local_trainer.run_init(
                current_image, current_camera, progress_bar, run_id=i
            )
            rotation, translation = self._local_trainer.run_transfo(
                next_image,
                current_camera,
                gaussian_model,
                progress_bar,
                run_id=i,
            )
            next_camera = transform_camera(
                current_camera, rotation, translation, next_image, _id=i
            )
            global_trainer.run(current_camera, next_camera, progress_bar, run_id=i)

            if self._debug:
                rendered_image, _, _, _ = render(next_camera, gaussian_model)
                save_image(
                    rendered_image, self._output_path / f"{i}_rendered_image.png"
                )
                save_image(next_image, self._output_path / f"{i}_image.png")

            current_image = next_image
            current_camera = next_camera
