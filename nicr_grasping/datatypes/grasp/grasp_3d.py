import numpy as np

import xml.etree.ElementTree as ET

from typing import Optional

import matplotlib.pyplot as plt

from .grasp_base import Grasp
from .grasp_2d import Grasp2D
from ..intrinsics import PinholeCameraIntrinsic

from ..transform.pose import Pose

from ...utils.io import mira_json_to_pose


class Grasp3D(Grasp):
    def __init__(self,
                 quality: float = 0,
                 position: np.ndarray = np.zeros((1, 3)),
                 orientation: np.ndarray = np.eye(3),
                 points: Optional[np.ndarray] = None):
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

    @property
    def transformation_matrix(self):
        res = np.eye(4)
        res[:3, :3] = self.orientation
        res[:3, 3] = self.position

        return res

    def transform(self, transform):
        transformation_matrix = np.eye(4)
        if isinstance(transform, Pose):
            transformation_matrix = transform.transformation_matrix
        elif isinstance(transform, np.ndarray):
            transformation_matrix = transform
        else:
            raise TypeError("transform must be either a Pose or a transformation matrix")

        new_transform = transformation_matrix @ self.transformation_matrix
        self.orientation = new_transform[:3, :3]
        self.position = new_transform[:3, 3]

    def open3d_geometry(self, **kwargs):
        gripper_mesh = o3d.geometry.TriangleMesh.create_box(width=0.01, height=0.01, depth=0.01)

        return gripper_mesh


class ParallelGripperGrasp3D(Grasp3D):
    def __init__(self,
                 width: float = 0,
                 quality: float = 0,
                 position: np.ndarray = np.zeros((1, 3)),
                 orientation: np.ndarray = np.eye(3),
                 points: Optional[np.ndarray] = None):
        super().__init__(quality=quality, position=position,
                         orientation=orientation, points=points)

        self._width = width
        self._height = 0

    @classmethod
    def from_grasp3d(cls, grasp_3d: Grasp3D):
        obj = cls()
        obj.position = grasp_3d.position
        obj.orientation = grasp_3d.orientation
        obj.quality = grasp_3d.quality

        return cls

    @classmethod
    def from_mira_json(cls, json_object: dict):
        obj = cls()

        position, orientation = mira_json_to_pose(json_object['Pose'])
        obj.position = position
        obj.orientation = orientation
        obj.quality = json_object['Quality']
        obj._width = json_object['GripperWidth']

        return obj

    @property
    def width(self):
        return self._width

    @width.setter
    def width(self, value):
        self._width = value

    @property
    def height(self):
        if self._height != 0:
            return self._height
        else:
            return self._width / 3

    @height.setter
    def height(self, value):
        self._height = value

    def open3d_geometry(self, simple_grasps=False, **kwargs):

        # import needs to be here to avoid problems when importing in MIRA unit
        import open3d as o3d

        if simple_grasps:
            width = self.width
            height = 0.0025
            depth = 0.04
            finger_width = 0.0025
        else:
            width = self.width
            height = self.height
            finger_width = 0.01

        gripper_mesh = o3d.geometry.TriangleMesh.create_box(width=width, height=height, depth=finger_width)

        gripper_transform = np.eye(4)
        gripper_transform[:3, 3] = np.array([-width / 2, - height / 2, -finger_width / 2])

        gripper_mesh = gripper_mesh.transform(gripper_transform)

        left_finger_mesh = o3d.geometry.TriangleMesh.create_box(width=finger_width, height=height, depth=0.04)

        left_finger_transform = np.eye(4)
        left_finger_transform[:3, 3] = np.array([width, 0, 0])

        right_finger_mesh = o3d.geometry.TriangleMesh.create_box(width=finger_width, height=height, depth=0.04)

        right_finger_transform = np.eye(4)
        right_finger_transform[:3, 3] = np.array([0, 0, 0])

        gripper_mesh += left_finger_mesh.transform(left_finger_transform @ gripper_transform)
        gripper_mesh += right_finger_mesh.transform(right_finger_transform @ gripper_transform)

        gripper_mesh.compute_vertex_normals()

        # set color based on confidence
        cmap = plt.get_cmap('jet')
        color = cmap(self.quality)[:3]
        gripper_mesh.paint_uniform_color(color)

        return gripper_mesh
