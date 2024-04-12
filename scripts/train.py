from argparse import ArgumentParser
from pathlib import Path

from gaussian_splatting.pose_free.pose_free_trainer import PoseFreeTrainer
from gaussian_splatting.trainer import Trainer

if __name__ == "__main__":
    parser = ArgumentParser(description="Training script.")
    parser.add_argument("-s", "--source-path", type=Path, required=True)
    parser.add_argument("--pose-free", action="store_true")
    args = parser.parse_args()

    if args.pose_free:
        trainer = PoseFreeTrainer(source_path=args.source_path)
    else:
        trainer = Trainer(source_path=args.source_path)

    trainer.run()
