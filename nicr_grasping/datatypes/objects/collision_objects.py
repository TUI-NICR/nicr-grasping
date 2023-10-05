import numpy as np

from . import SceneObject

class CollisionObject(SceneObject):
    def __init__(self, point_cloud: np.ndarray) -> None:
        self.point_cloud = point_cloud

    def sample_points(self, num_points: int = 500):
        vertex_indices = np.random.choice(self.point_cloud.shape[0], num_points)
        return self.point_cloud[vertex_indices]
