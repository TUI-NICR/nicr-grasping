import numpy as np

from . import SceneObject


class CollisionObject(SceneObject):

    def __init__(self, point_cloud: np.ndarray) -> None:
        if not isinstance(point_cloud, np.ndarray):
            try:
                point_cloud = np.array(point_cloud, dtype=np.float64)
            except AttributeError:
                raise TypeError("point_cloud must be a numpy array with dtype float64 or must by convertible to one")
        self.point_cloud = point_cloud

    def sample_points(self, num_points: int = -1) -> np.ndarray:
        if num_points > 0:
            vertex_indices = np.random.choice(self.point_cloud.shape[0], num_points)
            return self.point_cloud[vertex_indices]
        return self.point_cloud
