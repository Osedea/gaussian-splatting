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
from gaussian_splatting.arguments import ModelParams, PipelineParams, OptimizationParams
from gaussian_splatting.training import Trainer


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    trainer = Trainer(
        testing_iterations=args.test_iterations,
        saving_iterations=args.save_iterations,
        checkpoint_iterations=args.checkpoint_iterations,
        checkpoint=args.start_checkpoint,
        debug_from=args.debug_from,
        quiet=args.quiet,
        detect_anomaly=args.detect_anomaly
    )
    trainer.run(
        dataset=lp.extract(args),
        opt=op.extract(args),
        pipe=pp.extract(args),
    )
