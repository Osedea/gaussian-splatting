import torch
from transformers import pipeline

from gaussian_splatting.utils.general import TorchToPIL


class DepthEstimator:
    def __init__(self, model: str = "Intel/dpt-large"):
        self._model = pipeline("depth-estimation", model=model)

    def run(self, image):
        PIL_image = TorchToPIL(image)
        depth_estimation = self._model(PIL_image)["predicted_depth"]

        depth_estimation = torch.nn.functional.interpolate(
            depth_estimation.unsqueeze(1),
            size=PIL_image.size[::-1],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

        _min = depth_estimation.min()
        _max = depth_estimation.max()
        depth_estimation = (depth_estimation - _min) / (_max - _min)

        depth_estimation = -1 * (depth_estimation - 1)

        return depth_estimation
