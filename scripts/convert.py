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

import logging
import os
import shutil
from argparse import ArgumentParser
from pathlib import Path


def _run_command(command, step):
    exit_code = os.system(command)
    if exit_code != 0:
        logging.error(f"{step} failed with code {exit_code}. Exiting.")
        exit(exit_code)


def _resize(source_path, resizing_factor=2):
    destination_folder = source_path / f"images_{resizing_factor}"
    destination_folder.mkdir(exist_ok=True)
    for _file in (source_path + "/images").iterdir():
        shutil.copy2(_file, destination_folder / _file.name)
        _run_command(
            step=f"resize {resizing_factor}",
            command=f"magick mogrify \
            -resize {100/resizing_factor:.1f}% \
            {destination_folder / _file.name}",
        )


def main(source_path, resize, use_gpu):
    (source_path / "distorted" / "sparse").mkdir(parents=True, exist_ok=True)
    _run_command(
        step="Feature extraction",
        command=f"colmap feature_extractor \
        --database_path {source_path / 'distorted' / 'database.db'} \
        --image_path {source_path / 'input'} \
        --ImageReader.single_camera 1 \
        --ImageReader.camera_model OPENCV \
        --SiftExtraction.use_gpu {use_gpu}",
    )

    _run_command(
        step=" Feature matching",
        command=f"colmap sequential_matcher \
        --database_path {source_path / 'distorted' / 'database.db'} \
        --SiftMatching.use_gpu {use_gpu}",
    )

    _run_command(
        step="Mapper",
        command=f"colmap mapper \
        --database_path {source_path / 'distorted' / 'database.db'} \
        --image_path {source_path / 'input'} \
        --output_path {source_path / 'distorted' / 'sparse'}",
    )

    _run_command(
        step="Image undistortion",
        command=f"colmap image_undistorter \
        --image_path {source_path / 'input'} \
        --input_path {source_path / 'distorted' / 'sparse' / '0'} \
        --output_path {source_path} \
        --output_type COLMAP",
    )

    (source_path / "sparse" / "0").mkdir(exist_ok=True)
    for _file in (source_path / "sparse").iterdir():
        if _file.is_dir():
            continue
        shutil.move(_file, source_path / "sparse" / "0" / _file.name)

    if resize:
        [
            _resize(source_path, resizing_factor=resizing_factor)
            for resizing_factor in [2, 4, 8]
        ]


if __name__ == "__main__":
    # This Python script is based on the shell converter script provided in the MipNerF 360 repository.
    parser = ArgumentParser("Colmap converter")
    parser.add_argument("--source_path", "-s", required=True, type=Path)
    parser.add_argument("--no_gpu", action="store_true")
    parser.add_argument("--resize", action="store_true")
    args = parser.parse_args()
    main(
        source_path=args.source_path,
        resize=args.resize,
        use_gpu=(1 if not args.no_gpu else 0),
    )
