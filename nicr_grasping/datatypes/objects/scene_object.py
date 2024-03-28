import numpy as np

from ..transform import Pose


class SceneObject:
    def __init__(self, pose: Pose) -> None:
        self.pose = pose

    def sample_points(self, num_points: int = 500) -> np.ndarray:
        raise NotImplementedError
