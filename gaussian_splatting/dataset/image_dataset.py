from pathlib import Path

from PIL import Image

from gaussian_splatting.utils.general import PILtoTorch


class ImageDataset:
    def __init__(
        self, images_path: Path, step_size: int = 1, downscale_factor: int = 1
    ):
        self._images_paths = [
            f for i, f in enumerate(images_path.iterdir()) if i % step_size == 0
        ]
        self._images_paths.sort(key=lambda f: int(f.stem))

        self._downscale_factor = downscale_factor

    def __len__(self):
        return len(self._images_paths)

    def get_frame(self, i: int):
        image_path = self._images_paths[i]
        image = Image.open(image_path)

        if self._downscale_factor > 1:
            image = self._downscale(image)

        image = PILtoTorch(image)

        return image

    def _downscale(self, image):
        h, w = image.size
        image = image.resize(
            (h // self._downscale_factor, w // self._downscale_factor), Image.LANCZOS
        )

        return image
