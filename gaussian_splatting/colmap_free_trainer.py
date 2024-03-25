import os
import uuid

import torch
from tqdm import tqdm

from gaussian_splatting.model import GaussianModel
from gaussian_splatting.optimizer import Optimizer
from gaussian_splatting.render import render
from gaussian_splatting.utils.general import safe_state
from gaussian_splatting.utils.image import psnr
from gaussian_splatting.utils.loss import l1_loss, ssim


class ColmapFreeTrainer:
    def __init__(
        self,
        source_path,
        keep_eval=False,
        resolution=-1,
        sh_degree=3,
        checkpoint_path=None,
    ):
        self._model_path = self._prepare_model_path()

        self.dataset = ImageDataset(images_path=source_path)

        self.global_3DGS = GaussianModel(sh_degree)
        self.global_3DGS.initialize(self.dataset)
        self.global_3DGS_optimizer = Optimizer(self.global_3DGS)

        safe_state()

    def run(self):
        progress_bar = tqdm(range(len(self.dataset)), desc="Training progress")
        for iteration in range(len(dataset)):
            I_t = self.dataset[i]
            I_t_plus_1 = self.dataset[i + 1]

            local_3DGS_trainer = LocalTrainer()

            # self.optimizer.update_learning_rate(iteration)

            # Every 1000 its we increase the levels of SH up to a maximum degree
            # if iteration % 1000 == 0:
            #    self.gaussian_model.oneupSHdegree()

            # Pick a random camera
            # if not cameras:
            #    cameras = self.dataset.get_train_cameras().copy()
            # camera = cameras.pop(randint(0, len(cameras) - 1))

            # Render image
            rendered_image, viewspace_point_tensor, visibility_filter, radii = render(
                camera, self.gaussian_model
            )

            # Loss
            gt_image = camera.original_image.cuda()
            Ll1 = l1_loss(rendered_image, gt_image)
            loss = (1.0 - self._lambda_dssim) * Ll1 + self._lambda_dssim * (
                1.0 - ssim(rendered_image, gt_image)
            )

            # try:
            loss.backward()
            # except Exception:
            #    import pdb; pdb.set_trace()

            with torch.no_grad():
                # Progress bar
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                if iteration % 10 == 0:
                    progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                    progress_bar.update(10)
                if iteration == self._iterations:
                    progress_bar.close()

                # Log and save
                if iteration in self._testing_iterations:
                    self._report(iteration)

                if iteration in self._saving_iterations:
                    print("\n[ITER {}] Saving Gaussians".format(iteration))
                    point_cloud_path = os.path.join(
                        self.model_path, "point_cloud/iteration_{}".format(iteration)
                    )
                    self.gaussian_model.save_ply(
                        os.path.join(point_cloud_path, "point_cloud.ply")
                    )

                # Densification
                if iteration < self._densification_iteration_stop:
                    self.gaussian_model.update_stats(
                        viewspace_point_tensor, visibility_filter, radii
                    )

                    if (
                        iteration >= self._densification_iteration_start
                        and iteration % self._densification_interval == 0
                    ):
                        self._densify_and_prune(
                            iteration > self._opacity_reset_interval
                        )

                # Reset opacity interval
                if iteration % self._opacity_reset_interval == 0:
                    self._reset_opacity()

                # Optimizer step
                if iteration < self._iterations:
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)

                # Save checkpoint
                if iteration in self._checkpoint_iterations:
                    print("\n[ITER {}] Saving Checkpoint".format(iteration))
                    torch.save(
                        (
                            self.gaussian_model.state_dict(),
                            self.optimizer.state_dict(),
                            iteration,
                        ),
                        self.model_path + "/chkpnt" + str(iteration) + ".pth",
                    )

    def _prepare_model_path(self):
        unique_str = str(uuid.uuid4())
        model_path = os.path.join("./output/", unique_str[0:10])

        # Set up output folder
        print("Output folder: {}".format(model_path))
        os.makedirs(model_path, exist_ok=True)

        return model_path

    def _report(self, iteration):
        # Report test and samples of training set
        torch.cuda.empty_cache()
        validation_configs = {
            "test": self.dataset.get_test_cameras(),
            "train": [
                self.dataset.get_train_cameras()[
                    idx % len(self.dataset.get_train_cameras())
                ]
                for idx in range(5, 30, 5)
            ],
        }

        for config_name, cameras in validation_configs:
            if not cameras or len(cameras) == 0:
                continue

            l1_test, psnr_test = 0.0, 0.0
            for idx, camera in enumerate(cameras):
                rendered_image, _, _, _ = render(camera, self.gaussian_model)
                gt_image = camera.original_image.to("cuda")

                rendered_image = torch.clamp(rendered_image, 0.0, 1.0)
                gt_image = torch.clamp(gt, 0.0, 1.0)

                l1_test += l1_loss(image, gt_image).mean().double()
                psnr_test += psnr(image, gt_image).mean().double()

            psnr_test /= len(cameras)
            l1_test /= len(cameras)

            print(
                f"\n[ITER {iteration}] Evaluating {config_name}: L1 {l1_test} PSNR {psnr_test}"
            )

        torch.cuda.empty_cache()

    def _densify_and_prune(self, prune_big_points):
        # Clone large gaussian in over-reconstruction areas
        self._clone_points()
        # Split small gaussians in under-construction areas.
        self._split_points()

        # Prune transparent and large gaussians.
        prune_mask = (self.gaussian_model.get_opacity < self._min_opacity).squeeze()
        if prune_big_points:
            big_points_vs = self.gaussian_model.max_radii2D > self._max_screen_size
            big_points_ws = (
                self.gaussian_model.get_scaling.max(dim=1).values
                > 0.1 * self.gaussian_model.camera_extent
            )
            prune_mask = torch.logical_or(
                torch.logical_or(prune_mask, big_points_vs), big_points_ws
            )
        if self._debug:
            print(f"Pruning: {prune_mask.sum().item()} points.")
        self._prune_points(valid_mask=~prune_mask)

        torch.cuda.empty_cache()

    def _split_points(self):
        new_points, split_mask = self.gaussian_model.split_points(
            self._densification_grad_threshold, self._percent_dense
        )
        self._concatenate_points(new_points)

        prune_mask = torch.cat(
            (
                split_mask,
                torch.zeros(2 * split_mask.sum(), device="cuda", dtype=bool),
            )
        )
        if self._debug:
            print(f"Densification: split {split_mask.sum().item()} points.")
        self._prune_points(valid_mask=~prune_mask)

    def _clone_points(self):
        new_points, clone_mask = self.gaussian_model.clone_points(
            self._densification_grad_threshold, self._percent_dense
        )
        if self._debug:
            print(f"Densification: clone {clone_mask.sum().item()} points.")
        self._concatenate_points(new_points)

    def _reset_opacity(self):
        new_opacity = self.gaussian_model.reset_opacity()
        optimizable_tensors = self.optimizer.replace_points(new_opacity, "opacity")
        self.gaussian_model.set_optimizable_tensors(optimizable_tensors)

    def _prune_points(self, valid_mask):
        optimizable_tensors = self.optimizer.prune_points(valid_mask)
        self.gaussian_model.set_optimizable_tensors(optimizable_tensors)
        self.gaussian_model.mask_stats(valid_mask)

    def _concatenate_points(self, new_tensors):
        optimizable_tensors = self.optimizer.concatenate_points(new_tensors)
        self.gaussian_model.set_optimizable_tensors(optimizable_tensors)
        self.gaussian_model.reset_stats()
