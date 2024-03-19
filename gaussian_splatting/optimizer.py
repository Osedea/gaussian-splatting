import torch
from torch import nn

from gaussian_splatting.utils.general import get_expon_lr_func


class Optimizer:
    def __init__(
        self,
        gaussian_model,
        position_lr_init=0.00016,
        position_lr_final=0.0000016,
        position_lr_delay_mult=0.01,
        position_lr_max_steps=30000,
        feature_lr=0.0025,
        opacity_lr=0.05,
        scaling_lr=0.005,
        rotation_lr=0.001,
    ):
        params = [
            {
                "params": [gaussian_model._xyz],
                "lr": position_lr_init * gaussian_model.camera_extent,
                "name": "xyz",
            },
            {
                "params": [gaussian_model._features_dc],
                "lr": feature_lr,
                "name": "f_dc",
            },
            {
                "params": [gaussian_model._features_rest],
                "lr": feature_lr / 20.0,
                "name": "f_rest",
            },
            {
                "params": [gaussian_model._opacity],
                "lr": opacity_lr,
                "name": "opacity",
            },
            {
                "params": [gaussian_model._scaling],
                "lr": scaling_lr,
                "name": "scaling",
            },
            {
                "params": [gaussian_model._rotation],
                "lr": rotation_lr,
                "name": "rotation",
            },
        ]

        self._optimizer = torch.optim.Adam(params, lr=0.0, eps=1e-15)
        self._xyz_scheduler_args = get_expon_lr_func(
            lr_init=position_lr_init * gaussian_model.camera_extent,
            lr_final=position_lr_final * gaussian_model.camera_extent,
            lr_delay_mult=position_lr_delay_mult,
            max_steps=position_lr_max_steps,
        )

    def update_learning_rate(self, iteration):
        """Learning rate scheduling per step"""
        for param_group in self._optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self._xyz_scheduler_args(iteration)
                param_group["lr"] = lr

                return lr

    def step(self):
        self._optimizer.step()

    def zero_grad(self, set_to_none=True):
        self._optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        self._optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self._optimizer.load_state_dict(state_dict)

    def replace_points(self, tensor, name):
        optimizable_tensors = {}
        for group in self._optimizer.param_groups:
            if group["name"] == name:
                stored_state = self._optimizer.state.get(group["params"][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self._optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self._optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def prune_points(self, mask):
        optimizable_tensors = {}
        for group in self._optimizer.param_groups:
            stored_state = self._optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self._optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    (group["params"][0][mask].requires_grad_(True))
                )
                self._optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    group["params"][0][mask].requires_grad_(True)
                )
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def concatenate_points(self, tensors_dict):
        optimizable_tensors = {}
        for group in self._optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self._optimizer.state.get(group["params"][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat(
                    (stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0
                )
                stored_state["exp_avg_sq"] = torch.cat(
                    (stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)),
                    dim=0,
                )

                del self._optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat(
                        (group["params"][0], extension_tensor), dim=0
                    ).requires_grad_(True)
                )
                self._optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat(
                        (group["params"][0], extension_tensor), dim=0
                    ).requires_grad_(True)
                )
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors
