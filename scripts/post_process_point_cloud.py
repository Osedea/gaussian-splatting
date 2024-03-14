from pathlib import Path
from argparse import ArgumentParser
from plyfile import PlyData, PlyElement
import numpy as np


def _is_not_outlier(value, _range, c=1.5):
    lower_range, upper_range = _range

    is_not_outlier = value > (lower_range - c * (upper_range - lower_range)) and \
        value < (upper_range + c * (upper_range - lower_range))

    return is_not_outlier

def _get_range(data):
    lower_range = np.percentile(data, 25)
    upper_range = np.percentile(data, 75)

    return lower_range, upper_range


def main(point_cloud_path, c=1.5):
    plydata = PlyData.read(point_cloud_path)

    vertices = plydata["vertex"]

    axis_ranges = {
        axis: _get_range(vertices[axis])
        for axis in ["x", "y", "z"]
    }

    keep_indexes = [
        i for i, (x, y, z) in enumerate(zip(
            vertices["x"],
            vertices["y"],
            vertices["z"],
        ))
        if (
            _is_not_outlier(x, axis_ranges["x"], c) and
            _is_not_outlier(y, axis_ranges["y"], c) and
            _is_not_outlier(z, axis_ranges["z"], c)
        )
    ]

    print(f"Removing {len(vertices) - len(keep_indexes)} data points.")
    new_vertices = vertices[keep_indexes]

    print(f"New cloud point has {len(new_vertices)} data points.")
    new_vertex_element = PlyElement.describe(new_vertices, 'vertex')

    new_point_cloud_path = point_cloud_path.parent / f"point_cloud_clean_iqr_c={c}.ply"
    print(f"Saving result to file: {new_point_cloud_path}")
    PlyData([new_vertex_element]).write(new_point_cloud_path)



if __name__ == "__main__":
    parser = ArgumentParser("Point Cloud Post-Processing")
    parser.add_argument("--point_cloud_path", "-p", required=True, type=Path)
    parser.add_argument("-c", default=1.5, type=float)
    args = parser.parse_args()

    main(args.point_cloud_path, args.c)
