from pathlib import Path

from gaussian_splatting.local_trainer import LocalTrainer
from gaussian_splatting.dataset.image_dataset import ImageDataset


def main():
    dataset = ImageDataset(images_path=Path("data/phil/1/input/"))
    image = dataset.get_frame(0)

    local_trainer = LocalTrainer(image)
    local_trainer.run()


if __name__ == "__main__":
    main()
