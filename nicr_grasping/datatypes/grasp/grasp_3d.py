import numpy as np

from nicr_grasping.datatypes.grasp.grasp_2d import Grasp2D

from typing import Optional

from .grasp_base import Grasp
from ..intrinsics import PinholeCameraIntrinsic


class Grasp3D(Grasp):
    def __init__(self,
                 quality     : float = 0,
                 position    : np.ndarray = np.zeros((1, 3)),
                 orientation : np.ndarray = np.eye(3),
                 points      : Optional[np.ndarray] = None):
        super().__init__(quality=quality)
        # TODO: merge position and orientation/rotation to pose. Might make transformations easier.
        assert len(position.shape) == 2

        self.position = position
        self.orientation = orientation

        if points is None:
            self.points = self.position
        else:
            self.points = points

    def __eq__(self, __o: object) -> bool:

        if isinstance(__o, Grasp3D):
            return (self.position == __o.position).all() and (self.orientation == __o.orientation).all() and super(Grasp3D, self).__eq__(__o)

        return False

    def to_2d(self,
              camera_intrinsic: PinholeCameraIntrinsic):
        g = Grasp2D.from_points(camera_intrinsic.point_to_pixel(self.points))
        g.quality = self.quality
        return g


class PrallelGripperGrasp3D(Grasp3D):
    def __init__(self,
                 quality: float = 0,
                 position: np.ndarray = np.zeros((1, 3)),
                 orientation: np.ndarray = np.eye(3),
                 points: np.ndarray = np.zeros((4, 3))):
        super().__init__(quality=quality, position=position, orientation=orientation, points=points)

        self._width = 0
        self._height = 0

    @classmethod
    def from_grasp3d(cls, grasp_3d: Grasp3D):
        obj = cls()
        obj.position = grasp_3d.position
        obj.orientation = grasp_3d.orientation
        obj.quality = grasp_3d.quality

        return cls
