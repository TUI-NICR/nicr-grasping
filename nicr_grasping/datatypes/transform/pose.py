import numpy as np
from scipy.spatial.transform import Rotation as R

from ...utils.io import mira_json_to_pose


class Pose:
    def __init__(self, position: np.ndarray = np.zeros((3,)), rotation: np.ndarray = np.eye(3)):
        self._position = position
        self._rotation = rotation

    @classmethod
    def from_transformation_matrix(cls, transformation_matrix: np.ndarray):
        rotation = transformation_matrix[:3, :3]
        position = transformation_matrix[:3, 3]
        return cls(position, rotation)

    @property
    def position(self):
        return self._position

    @property
    def rotation(self):
        return self._rotation

    @property
    def transformation_matrix(self):
        res = np.eye(4)
        res[:3, :3] = self._rotation
        res[:3, 3] = self._position

        return res

    def normalize_rotation(self):
        # https://github.com/wenbowen123/BundleTrack/blob/master/scripts/benchmark.py#L59
        for i in range(3):
            norm = np.linalg.norm(self.rotation[:,i])
            self.rotation[:,i] /= norm

    def transform(self, transform):
        mat = np.eye(4)
        if isinstance(transform, Pose):
            mat = transform.transformation_matrix
        elif isinstance(transform, np.ndarray):
            mat = transform
        else:
            raise TypeError("transform must be either a Pose or a transformation matrix")

        return Pose.from_transformation_matrix(mat @ self.transformation_matrix)

    @classmethod
    def from_mira_json(cls, json_object: dict):
        position, orientation = mira_json_to_pose(json_object)
        return cls(position, orientation)

    def inverse(self):
        return Pose.from_transformation_matrix(np.linalg.inv(self.transformation_matrix))
