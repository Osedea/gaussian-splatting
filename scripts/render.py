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

import os
from argparse import ArgumentParser
from os import makedirs

import torch
import torchvision
from tqdm import tqdm

from gaussian_splatting.gaussian_renderer import GaussianModel, render
from gaussian_splatting.scene import Dataset
from gaussian_splatting.utils.general import safe_state
from gaussian_splatting.utils.system import searchForMaxIteration


def render_set(model_path, name, iteration, views, gaussian_model):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render(view, gaussian_model)["render"]
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(
            rendering, os.path.join(render_path, "{0:05d}".format(idx) + ".png")
        )
        torchvision.utils.save_image(
            gt, os.path.join(gts_path, "{0:05d}".format(idx) + ".png")
        )


def render_sets(
    source_path,
    model_path,
    iteration: int,
    skip_train: bool,
    skip_test: bool,
    resolution: int = -1,
):
    with torch.no_grad():
        dataset = Dataset(
            source_path,
            shuffle=False,
            resolution=resolution,
        )
        gaussian_model = GaussianModel(dataset.sh_degree)

        if iteration == -1:
            iteration = searchForMaxIteration(
                os.path.join(self.model_path, "point_cloud")
            )
        print(f"Loading trained model at iteration {iteration}.")

        gaussian_model.load_ply(
            os.path.join(
                model_path,
                "point_cloud",
                f"iteration_{iteration}",
                "point_cloud.ply",
            )
        )

        if not skip_train:
            render_set(
                model_path,
                "train",
                dataset.getTrainCameras(),
                gaussian_model,
            )

        if not skip_test:
            render_set(
                model_path,
                "test",
                dataset.getTestCameras(),
                gaussian_model,
            )


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    parser.add_argument("-s", "--source-path", type=str, required=True)
    parser.add_argument("-m", "--model-path", type=str, required=True)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    safe_state()

    render_sets(
        source_path, model_path, args.iteration, args.skip_train, args.skip_test
    )
