import numpy as np


def apply_affine_transformation(xyz, rotation, translation):
    transformed_xyz = np.matmul(xyz, rotation.T) + translation

    return transformed_xyz
