#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import sys
from argparse import ArgumentParser

from gaussian_splatting.training import Trainer

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument("-s", "--source-path", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--resolution", default=-1, type=int)
    args = parser.parse_args(sys.argv[1:])

    trainer = Trainer(
        source_path=args.source_path,
        resolution=args.resolution,
        checkpoint_path=args.checkpoint_path,
    )
    trainer.run()
