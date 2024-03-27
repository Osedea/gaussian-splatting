import torch
from torch import nn


class QuaternionRotationLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.quaternion = nn.Parameter(torch.Tensor([[0, 0, 0, 1]]))

    def forward(self, input_tensor):
        rotation_matrix = self.get_rotation_matrix()
        rotated_tensor = torch.matmul(input_tensor, rotation_matrix)

        return rotated_tensor

    def get_rotation_matrix(self):
        # Normalize quaternion to ensure unit magnitude
        quaternion_norm = torch.norm(self.quaternion, p=2, dim=1, keepdim=True)
        normalized_quaternion = self.quaternion / quaternion_norm

        x, y, z, w = normalized_quaternion[0]
        rotation_matrix = torch.zeros(
            3, 3, dtype=torch.float32, device=self.quaternion.device
        )
        rotation_matrix[0, 0] = 1 - 2 * (y**2 + z**2)
        rotation_matrix[0, 1] = 2 * (x * y - w * z)
        rotation_matrix[0, 2] = 2 * (x * z + w * y)
        rotation_matrix[1, 0] = 2 * (x * y + w * z)
        rotation_matrix[1, 1] = 1 - 2 * (x**2 + z**2)
        rotation_matrix[1, 2] = 2 * (y * z - w * x)
        rotation_matrix[2, 0] = 2 * (x * z - w * y)
        rotation_matrix[2, 1] = 2 * (y * z + w * x)
        rotation_matrix[2, 2] = 1 - 2 * (x**2 + y**2)

        return rotation_matrix


class TranslationLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.translation = nn.Parameter(torch.Tensor(1, 3))
        nn.init.zeros_(self.translation)

    def forward(self, input_tensor):
        translated_tensor = torch.add(self.translation, input_tensor)

        return translated_tensor


class AffineTransformationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self._rotation = QuaternionRotationLayer()
        self._translation = TranslationLayer()

    def forward(self, xyz):
        transformed_xyz = self._rotation(xyz)
        transformed_xyz = self._translation(transformed_xyz)

        return transformed_xyz

    @property
    def rotation(self):
        rotation = self._rotation.get_rotation_matrix().detach().cpu()

        return rotation

    @property
    def translation(self):
        translation = self._translation.translation.detach().cpu()

        return translation
