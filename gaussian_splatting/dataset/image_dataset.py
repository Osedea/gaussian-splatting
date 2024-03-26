from pathlib import Path

from PIL import Image


class ImageDataset:
    def __init__(self, images_path: Path):
        self._images_paths = [f for f in images_path.iterdir()]
        self._images_paths.sort(key=lambda f: int(f.stem))

    def get_frame(self, i: int):
        image_path = self._images_paths[i]
        image = Image.open(image_path)

        return image

    def __len__(self):
        return len(self._images_paths)
