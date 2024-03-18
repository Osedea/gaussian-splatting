import os
import uuid
from argparse import Namespace
from random import randint

import torch
from tqdm import tqdm

from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.optimizer import Optimizer
from gaussian_splatting.scene import Dataset
from gaussian_splatting.scene.gaussian_model import GaussianModel
from gaussian_splatting.utils.general import safe_state
from gaussian_splatting.utils.image import psnr
from gaussian_splatting.utils.loss import l1_loss, ssim


class Trainer:
    def __init__(
        self,
        source_path,
        model_path="",
        keep_eval=False,
        sh_degree=3,
        resolution=-1,
        testing_iterations=None,
        saving_iterations=None,
        checkpoint_iterations=None,
        checkpoint_path=None,
    ):
        self._resolution = resolution

        if testing_iterations is None:
            testing_iterations = [7000, 30000]
        self._testing_iterations = testing_iterations

        if saving_iterations is None:
            saving_iterations = [7000, 30000]
        self._saving_iterations = saving_iterations

        if checkpoint_iterations is None:
            checkpoint_iterations = []
        self._checkpoint_iterations = checkpoint_iterations

        self._checkpoint_path = checkpoint_path

        self.iterations = 30000
        self.lambda_dssim = 0.2
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15000
        self.densify_grad_threshold = 0.0002

        self.dataset = Dataset(source_path, keep_eval=keep_eval, resolution=resolution)
        self.dataset.save_scene_info(model_path)

        self.gaussian_model = GaussianModel(sh_degree)
        self.gaussian_model.initialize(self.dataset)

        self.optimizer = Optimizer(self.gaussian_model)

        safe_state()

    def run(self):
        first_iter = 0

        if self._checkpoint_path:
            gaussian_model_state_dict, self.optimizer_state_dict, first_iter = (
                torch.load(checkpoint_path)
            )
            self.gaussian_model.load_state_dict(gaussian_model_state_dict)
            self.optimizer.load_state_dict(optmizer_state_dict)

        iter_start = torch.cuda.Event(enable_timing=True)
        iter_end = torch.cuda.Event(enable_timing=True)

        viewpoint_stack = None
        ema_loss_for_log = 0.0
        progress_bar = tqdm(
            range(first_iter, self.iterations), desc="Training progress"
        )
        first_iter += 1
        for iteration in range(first_iter, self.iterations + 1):
            iter_start.record()

            self.optimizer.update_learning_rate(iteration)

            # Every 1000 its we increase the levels of SH up to a maximum degree
            if iteration % 1000 == 0:
                self.gaussian_model.oneupSHdegree()

            # Pick a random Camera
            if not viewpoint_stack:
                viewpoint_stack = self.dataset.getTrainCameras().copy()
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

            render_pkg = render(viewpoint_cam, self.gaussian_model)
            image, viewspace_point_tensor, visibility_filter, radii = (
                render_pkg["render"],
                render_pkg["viewspace_points"],
                render_pkg["visibility_filter"],
                render_pkg["radii"],
            )

            # Loss
            gt_image = viewpoint_cam.original_image.cuda()
            Ll1 = l1_loss(image, gt_image)
            loss = (1.0 - self.lambda_dssim) * Ll1 + self.lambda_dssim * (
                1.0 - ssim(image, gt_image)
            )
            loss.backward()

            iter_end.record()

            with torch.no_grad():
                # Progress bar
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                if iteration % 10 == 0:
                    progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                    progress_bar.update(10)
                if iteration == self.iterations:
                    progress_bar.close()

                # Log and save
                training_report(
                    iteration,
                    Ll1,
                    loss,
                    l1_loss,
                    iter_start.elapsed_time(iter_end),
                    self._testing_iterations,
                    self.dataset,
                    self.gaussian_model,
                    render,
                )
                if iteration in self._saving_iterations:
                    print("\n[ITER {}] Saving Gaussians".format(iteration))
                    point_cloud_path = os.path.join(
                        self.model_path, "point_cloud/iteration_{}".format(iteration)
                    )
                    self.gaussian_model.save_ply(
                        os.path.join(point_cloud_path, "point_cloud.ply")
                    )

                # Densification
                if iteration < self.densify_until_iter:
                    # Keep track of max radii in image-space for pruning
                    self.gaussian_model.max_radii2D[visibility_filter] = torch.max(
                        self.gaussian_model.max_radii2D[visibility_filter],
                        radii[visibility_filter],
                    )
                    self.gaussian_model.add_densification_stats(
                        viewspace_point_tensor, visibility_filter
                    )

                    # Densify
                    if (
                        iteration > self.densify_from_iter
                        and iteration % self.densification_interval == 0
                    ):
                        size_threshold = (
                            20 if iteration > self.opacity_reset_interval else None
                        )
                        self.gaussian_model.densify_and_prune(
                            self.densify_grad_threshold,
                            0.005,
                            self.dataset.cameras_extent,
                            size_threshold,
                        )

                    # Reset interval
                    if (
                        iteration % self.opacity_reset_interval == 0
                        or iteration == self.densify_from_iter
                    ):
                        self.reset_opacity()

                # Optimizer step
                if iteration < self.iterations:
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)

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

    def reset_opacity(self):
        new_opacity = inverse_sigmoid(
            torch.min(
                self.gaussian_model.get_opacity,
                torch.ones_like(self.gaussian_model.get_opacity) * 0.01,
            )
        )
        optimizable_tensors = self.optimizer.replace_tensor(new_opacity, "opacity")
        self.gaussian_model.set_opacity = optimizable_tensors["opacity"]

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self.optimizer.prune_mask(valid_points_mask)
        self.gaussian_model.set_optimizable_tensors(optimizable_tensors)
        # TODO
        self.gaussian_model.xyz_gradient_accum = self.gaussian_model.xyz_gradient_accum[
            valid_points_mask
        ]
        self.gaussian_model.denom = self.gaussian_model.denom[valid_points_mask]
        self.gaussian_model.max_radii2D = self.gaussian_model.max_radii2D[
            valid_points_mask
        ]

    def densification_postfix(
        self,
        new_xyz,
        new_features_dc,
        new_features_rest,
        new_opacities,
        new_scaling,
        new_rotation,
    ):
        d = {
            "xyz": new_xyz,
            "f_dc": new_features_dc,
            "f_rest": new_features_rest,
            "opacity": new_opacities,
            "scaling": new_scaling,
            "rotation": new_rotation,
        }

        optimizable_tensors = self.optimizer.cat_tensors(d)
        self.gaussian_model.set_optimizable_tensors(optimizable_tensors)

        # TODO
        self.gaussian_model.xyz_gradient_accum = torch.zeros(
            (self.gaussian_model.get_xyz.shape[0], 1), deviiblece="cuda"
        )
        self.gaussian_model.denom = torch.zeros(
            (self.gaussian_model.get_xyz.shape[0], 1), device="cuda"
        )
        self.gaussian_model.max_radii2D = torch.zeros(
            (self.gaussian_model.get_xyz.shape[0]), device="cuda"
        )

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[: grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values
            > self.percent_dense * scene_extent,
        )

        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[
            selected_pts_mask
        ].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(
            self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N)
        )
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)

        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacity,
            new_scaling,
            new_rotation,
        )

        prune_filter = torch.cat(
            (
                selected_pts_mask,
                torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool),
            )
        )
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(
            torch.norm(grads, dim=-1) >= grad_threshold, True, False
        )
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values
            <= self.percent_dense * scene_extent,
        )

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacities,
            new_scaling,
            new_rotation,
        )

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(
                torch.logical_or(prune_mask, big_points_vs), big_points_ws
            )
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv("OAR_JOB_ID"):
            unique_str = os.getenv("OAR_JOB_ID")
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), "w") as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))


def training_report(
    iteration,
    Ll1,
    loss,
    l1_loss,
    elapsed,
    testing_iterations,
    dataset: Dataset,
    gaussian_model: GaussianModel,
    renderFunc,
):
    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = (
            {"name": "test", "cameras": dataset.getTestCameras()},
            {
                "name": "train",
                "cameras": [
                    dataset.getTrainCameras()[idx % len(dataset.getTrainCameras())]
                    for idx in range(5, 30, 5)
                ],
            },
        )

        for config in validation_configs:
            if config["cameras"] and len(config["cameras"]) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config["cameras"]):
                    image = torch.clamp(
                        renderFunc(viewpoint, gaussian_model)["render"],
                        0.0,
                        1.0,
                    )
                    gt_image = torch.clamp(
                        viewpoint.original_image.to("cuda"), 0.0, 1.0
                    )
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config["cameras"])
                l1_test /= len(config["cameras"])
                print(
                    "\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(
                        iteration, config["name"], l1_test, psnr_test
                    )
                )

        torch.cuda.empty_cache()

        print("\nTraining complete.")
