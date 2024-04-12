from pathlib import Path

from gaussian_splatting.optimizer import Optimizer
from gaussian_splatting.render import render
from gaussian_splatting.trainer import Trainer
from gaussian_splatting.utils.general import safe_state
from gaussian_splatting.utils.loss import PhotometricLoss


class GlobalTrainer(Trainer):
    def __init__(self, gaussian_model, iterations: int = 100, output_path=None):
        self._model_path = self._prepare_model_path(output_path)

        self.gaussian_model = gaussian_model

        self.optimizer = Optimizer(self.gaussian_model)
        self._photometric_loss = PhotometricLoss(lambda_dssim=0.2)

        self._iterations = iterations

        self._debug = False

        # Densification and pruning
        self._min_opacity = 0.005
        self._max_screen_size = 20
        self._percent_dense = 0.01
        self._densification_grad_threshold = 0.0002

        safe_state()

    def run(self, current_camera, next_camera, progress_bar=None, run_id: int = 0):
        cameras = (current_camera, next_camera)
        for iteration in range(self._iterations):
            self.optimizer.update_learning_rate(iteration)

            # Every 1000 its we increase the levels of SH up to a maximum degree
            if iteration % 1000 == 0:
                self.gaussian_model.oneupSHdegree()

            camera = cameras[iteration % 2]

            # Render image
            rendered_image, viewspace_point_tensor, visibility_filter, radii = render(
                camera, self.gaussian_model
            )

            # Loss
            gt_image = camera.original_image.cuda()
            loss = self._photometric_loss(rendered_image, gt_image)
            loss.backward()
            loss_value = loss.cpu().item()

            # Optimizer step
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

            if progress_bar is not None:
                progress_bar.set_postfix(
                    {
                        "stage": "global",
                        "iteration": f"{iteration}/{self._iterations}",
                        "loss": f"{loss_value:.5f}",
                    }
                )

        self.gaussian_model.save_ply(
            Path(self._model_path) / "point_cloud" / str(run_id) / "point_cloud.ply"
        )

        # Densification
        self.gaussian_model.update_stats(
            viewspace_point_tensor, visibility_filter, radii
        )
        self._densify_and_prune(True)
        # self._reset_opacity()
