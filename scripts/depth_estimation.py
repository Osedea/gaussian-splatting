import time
from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt
from torchvision.utils import save_image
from tqdm import tqdm
from transformers import DPTForDepthEstimation, pipeline

from gaussian_splatting.dataset.image_dataset import ImageDataset
from gaussian_splatting.utils.general import TorchToPIL

_AGG = {"mean": np.mean, "min": np.min, "max": np.max}


def main(source_path, output_path):
    dataset = ImageDataset(images_path=source_path, step_size=1, downscale_factor=1)
    depth_estimator = pipeline("depth-estimation", model="Intel/dpt-large")

    stats = []
    for i in tqdm(range(len(dataset))):
        image = dataset.get_frame(i)
        time0 = time.time()
        normalized_depth_estimation, _min, _max = _get_depth_estimation(
            depth_estimator, image
        )
        time1 = time.time()
        latency = time1 - time0

        stats.append((latency, _min, _max))
        save_image(normalized_depth_estimation, output_path / f"{i}_depth.png")
        save_image(image, output_path / f"{i}_image.png")

    x = [i for i in range(len(dataset))]
    latencies, _mins, _maxs = zip(*stats)

    _print(x, latencies, "latency")
    _print(x, _mins, "min")
    _print(x, _maxs, "max")


def _print(x, y, name, aggregators=_AGG):
    print(name)
    for agg_name, agg in aggregators.items():
        print(f"> {agg_name}: {agg(y)}")

    plt.cla()
    plt.plot(x, y)
    plt.savefig(output_path / f"{name}.png")


def _get_depth_estimation(depth_estimator, image):
    PIL_image = TorchToPIL(image)
    depth_estimation = depth_estimator(PIL_image)["predicted_depth"]

    depth_estimation = torch.nn.functional.interpolate(
        depth_estimation.unsqueeze(1),
        size=PIL_image.size[::-1],
        mode="bicubic",
        align_corners=False,
    )

    _min, _max = depth_estimation.min().item(), depth_estimation.max().item()
    normalized_depth_estimation = (depth_estimation - _min) / (_max - _min)

    return normalized_depth_estimation, _min, _max


if __name__ == "__main__":
    output_path = Path("output_depth")
    output_path.mkdir(exist_ok=True, parents=True)

    main(source_path=Path("data/phil/1/input/"), output_path=output_path)
