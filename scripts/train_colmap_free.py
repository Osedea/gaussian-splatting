from pathlib import Path

from gaussian_splatting.dataset.image_dataset import ImageDataset
from gaussian_splatting.local_initialization_trainer import \
    LocalInitializationTrainer
from gaussian_splatting.local_transformation_trainer import \
    LocalTransformationTrainer


def main():
    dataset = ImageDataset(images_path=Path("data/phil/1/input/"))
    image_0 = dataset.get_frame(0)
    image_1 = dataset.get_frame(10)

    local_initialization_trainer = LocalInitializationTrainer(image_0, iterations=100)
    local_initialization_trainer.run()

    local_transformation_trainer = LocalTransformationTrainer(
        image_1,
        camera=local_initialization_trainer.camera,
        gaussian_model=local_initialization_trainer.gaussian_model,
    )
    local_transformation_trainer.run()


if __name__ == "__main__":
    main()
