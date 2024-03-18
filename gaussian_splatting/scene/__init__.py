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

import json
import os
import random

from gaussian_splatting.scene.dataset_readers import readColmapSceneInfo
from gaussian_splatting.utils.camera import (camera_to_JSON,
                                             cameraList_from_camInfos)


class Dataset:

    def __init__(
        self,
        source_path,
        keep_eval=False,
        shuffle=True,
        resolution=-1,
        resolution_scales=[1.0],
    ):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(source_path, "sparse")):
            scene_info = readColmapSceneInfo(source_path, keep_eval)
        else:
            assert False, "Could not recognize scene type!"

        if shuffle:
            random.shuffle(scene_info.train_cameras)
            random.shuffle(scene_info.test_cameras)

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(
                scene_info.train_cameras, resolution_scale, resolution
            )
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(
                scene_info.test_cameras, resolution_scale, resolution
            )

        self.scene_info = scene_info

    def save_scene_info(self, model_path):
        with open(self.scene_info.ply_path, "rb") as src_file, open(
            os.path.join(model_path, "input.ply"), "wb"
        ) as dest_file:
            dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if self.scene_info.test_cameras:
                camlist.extend(self.scene_info.test_cameras)
            if self.scene_info.train_cameras:
                camlist.extend(self.scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(model_path, "cameras.json"), "w") as file:
                json.dump(json_cams, file)

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
