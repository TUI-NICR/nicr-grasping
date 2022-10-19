import numpy as np
import cv2
import abc
import enum
from pathlib import Path
from copy import deepcopy
import rtree.index

from shapely.geometry import Polygon
from shapely.affinity import translate, rotate

from skimage.draw import polygon, polygon_perimeter, line
from ...utils.draw import draw_gauss

from typing import Any, List, Optional, Union

from .grasp_base import Grasp
from ..rotation import Rotation2D

from ..intrinsics import PinholeCameraIntrinsic

class RectangleGraspDrawingMode(enum.Enum):
    INNER_RECTANGLE = 0
    INNER_RECTANGLE_WITH_MARGIN = 1
    CENTER_POINT = 2
    INNER_TENTH_RECTANGLE = 3
    INNER_TENTH_RECTANGLE_WITH_MARGIN = 4
    GAUSS = 5
    GAUSS_WITH_MARGIN = 6
    INNER_RECTANGLE_WITH_FULL_MARGIN = 7
    INNER_TENTH_RECTANGLE_WITH_FULL_MARGIN = 8
    GAUSS_WITH_FULL_MARGIN = 9


class DepthLookupMethod(enum.Enum):
    CENTER_POINT = 0
    CENTER_AREA = 1


class Grasp2D(Grasp):
    def __init__(self,
                 quality: float      = 0,
                 center : np.ndarray = np.zeros((1, 2))):
        """Grasprepresentation for 2D or rectangle grasps.
        In general a 2D grasp should define a center usually where the end effector frame will be located.

        Parameters
        ----------
        quality : float, optional
            The quality of the grasp, by default 0
        center : np.ndarray, optional
            The center point of the grasp, by default np.zeros((1, 2))
        """
        super().__init__(quality=quality)

        assert len(center.shape) == 2

        self._center  = center
        self._points = center

    @classmethod
    def from_points(cls, points : np.ndarray):
        obj = cls()

        obj._points = points
        obj._center = points.mean(axis=0, keepdims=True)

        return obj

    def __eq__(self, __o: object) -> bool:

        if isinstance(__o, Grasp2D):
            return np.isclose(self.points, __o.points).all() and super(Grasp2D, self).__eq__(__o)

        return False

    @property
    def points(self) -> np.ndarray:
        return self._points

    @property
    def center(self):
        return self._points.mean(axis=0, keepdims=True)

    def rotate(self, rotation: Union[Rotation2D, float]):
        if not isinstance(rotation, Rotation2D):
            rotation = Rotation2D(angle_rad=rotation)

        # use inverted matrix because rotation in image plane is clockwise
        # whereas rotation in normal coordinate frame is counterclockwise
        R = rotation.to_mat().T

        c = self._center
        self._points = ((np.dot(R, (self.points - c).T)).T + c)

    def scale(self, scale_factor: float):
        self._points *= scale_factor

    def to_3d(self,
              depth_image: np.ndarray,
              intrinsic: PinholeCameraIntrinsic,
              depth_lookup_method: DepthLookupMethod = DepthLookupMethod.CENTER_POINT):
        from .grasp_3d import Grasp3D

        # get depth value through specified method
        if depth_lookup_method == DepthLookupMethod.CENTER_POINT:
            depth = self._depth_lookup_center(depth_image)
        elif depth_lookup_method == DepthLookupMethod.CENTER_AREA:
            raise NotImplementedError()

        # project pixels into 3d space
        points_3d = intrinsic.pixel_to_point(self.points, depth)

        # compute rotation matrix based on points
        rotation = self._compute_rotation_matrix(points_3d)

        return Grasp3D(self.quality, points_3d.mean(axis=0, keepdims=True), rotation, points_3d)

    def _compute_rotation_matrix(self, points_3d):
        return np.eye(3)

    def _depth_lookup_center(self, depth_image):
        return depth_image[self.center.astype(int)[:, 0], self.center.astype(int)[:, 1]]

class RectangleGrasp(Grasp2D):
    def __init__(self,
                 quality: float = 0,
                 center: np.ndarray = np.zeros((1, 2)),
                 width: float = 1,
                 angle: float = 0,
                 length: Optional[float] = None,
                 additional_params = None):
        """Rectangle representation of a 2d grasp for parallel grippers.

        Parameters
        ----------
        quality : float, optional
            The quality of the grasp, by default 0
        center : np.ndarray, optional
            The center point of the grasp, by default np.zeros((1,2))
        length : float, optional
            The length of the grasp. Equals the width of the fingers or gripper plates, by default width / 2
        width : float, optional
            The width of the grasp. Equals the distance between the fingers/plates when needed for executing this grasp, by default 0
        """
        super().__init__(quality=quality, center=center)
        assert length != 0 and width != 0

        # if length is not defined choose default ratio
        length: float = length if length is not None else width / 2

        self._compute_points(angle, width, length)

        self._additional_params = additional_params

    @property
    def width(self):
        return np.linalg.norm(self._points[1] - self._points[2])

    @width.setter
    def width(self, value):
        assert value != 0, "Width may not be 0!"
        rotation = Rotation2D(angle_rad=self.angle)
        rotmat = rotation.to_mat()
        # first rotate with inverse (same as in rotate() method)
        center = self._center
        self._points -= self._center
        self._points = np.dot(self._points, rotmat.T)

        # scale width
        self._points[:, 0] *= value / self.width

        #rotate back
        self._points = np.dot(self._points, rotmat)

        self._points += center

    @property
    def angle(self):
        """Compute angle based on rectangle edges.
        It is important to note that angle is given for image coordinate system.
        This is why the arctan is take from (-y, x) as the arctan works in normal coordinate system.

        Returns
        -------
        float
            The angle of the grasp in image coordinate system.
        """
        diff = self._points[1] - self._points[2]
        return np.arctan2(-diff[1], diff[0])

    @property
    def length(self) -> float:
        return np.linalg.norm(self._points[0] - self._points[1])

    @length.setter
    def length(self, value : float) -> None:
        assert value != 0, "Length may not be 0!"
        rotation = Rotation2D(angle_rad=self.angle)
        rotmat = rotation.to_mat()

        center = self._center
        self._points -= center

        # first rotate with inverse (same as in rotate() method)
        self._points = np.dot(self._points, rotmat.T)

        # scale length
        self._points[:, 1] *= value / self.length

        # rotate back
        self._points = np.dot(self._points, rotmat)

        self._points += center

    def to_polygon(self) -> Polygon:
        return Polygon(self.points)

    def _compute_points(self,
                        angle,
                        width,
                        length) -> np.ndarray:
        """Computes keypoints of rectengular representation (edges).

        Returns
        -------
        np.ndarray
            Numpy array of four points. Dimensions: [4,2]
        """
        rot = Rotation2D(angle_rad=angle)

        # define grasp points for centered grasp
        base_grasp = np.zeros((4, 2))
        base_grasp[1:3, 1] = length / 2
        base_grasp[0, 1] = -length / 2
        base_grasp[3, 1] = -length / 2

        base_grasp[0:2, 0] = width / 2
        base_grasp[2:4, 0] = -width / 2

        # rotate by angle
        self._points = base_grasp
        self._points += self._center
        self.rotate(angle)

    def save(self, file_path: Union[Path, str]):
        np.save(file_path, self.points)

    @classmethod
    def load_from_file(cls, file_path: Union[Path, str]):
        points = np.load(file_path)
        return cls.from_points(points)

    def draw_label(self,
                   images : List[np.ndarray],
                   mode : RectangleGraspDrawingMode = RectangleGraspDrawingMode.INNER_RECTANGLE) -> List[np.ndarray]:

        if mode == RectangleGraspDrawingMode.INNER_RECTANGLE:
            return self._draw_label_inner_rectangle(images, 3)
        elif mode == RectangleGraspDrawingMode.INNER_RECTANGLE_WITH_MARGIN:
            return self._draw_label_inner_rectangle(images, 3, True)
        elif mode == RectangleGraspDrawingMode.CENTER_POINT:
            return self._draw_label_center_point(images)
        elif mode == RectangleGraspDrawingMode.INNER_TENTH_RECTANGLE:
            return self._draw_label_inner_rectangle(images, 10)
        elif mode == RectangleGraspDrawingMode.GAUSS:
            return self._draw_label_gauss(images, draw_margin=False)
        elif mode == RectangleGraspDrawingMode.GAUSS_WITH_MARGIN:
            return self._draw_label_gauss(images, draw_margin=True)
        elif mode == RectangleGraspDrawingMode.INNER_TENTH_RECTANGLE_WITH_MARGIN:
            return self._draw_label_inner_rectangle(images, 10, True)
        elif mode == RectangleGraspDrawingMode.INNER_RECTANGLE_WITH_FULL_MARGIN:
            return self._draw_label_inner_rectangle(images, 3, True, True)
        elif mode == RectangleGraspDrawingMode.INNER_TENTH_RECTANGLE_WITH_FULL_MARGIN:
            return self._draw_label_inner_rectangle(images, 10, True, True)
        elif mode == RectangleGraspDrawingMode.GAUSS_WITH_FULL_MARGIN:
            return self._draw_label_gauss(images, draw_margin=True, full_margin=True)
        else:
            raise NotImplementedError(f'Label for mode {mode} not implemented')

    def _draw_label_center_point(self, images: List[np.ndarray]) -> List[np.ndarray]:
        # invert x, y order for array indexing
        # original order is for image coordinate system
        center = self.center[0].astype(int)[::-1]
        try:
            images[0][tuple(center)] = self.quality
            images[1][tuple(center)] = self.angle
            images[2][tuple(center)] = self.width
        except:
            pass

        return images

    def _draw_label_inner_rectangle(self, images: List[np.ndarray],
                                    fraction: int,
                                    draw_margin: bool = False,
                                    full_margin: bool = False) -> List[np.ndarray]:
        g_copy = self.copy()
        g_copy.width /= fraction
        g_copy.length = self.length

        shape = images[0].shape

        rr, cc = polygon(g_copy.points[:, 1], g_copy.points[:, 0], shape)
        images[0][rr, cc] = self.quality
        images[1][rr, cc] = self.angle
        images[2][rr, cc] = self.width

        if draw_margin:
            if full_margin:
                g_copy = self.copy()
                g_copy.length = self.length

                shape = images[0].shape

                rr, cc = polygon(g_copy.points[:, 1], g_copy.points[:, 0], shape)
                mask = images[0][rr, cc]
                mask[mask == 0] = -1
                images[0][rr, cc] = mask
            else:
                g_copy = self.copy()
                g_copy.width *= 3 / fraction
                g_copy.length = self.length

                shape = images[0].shape

                rr, cc = polygon(g_copy.points[:, 1], g_copy.points[:, 0], shape)
                mask = images[0][rr, cc]
                mask[mask == 0] = -1
                images[0][rr, cc] = mask

        return images

    def _draw_label_gauss(self,
                          images: List[np.ndarray],
                          draw_margin: bool = False,
                          full_margin: bool = False) -> List[np.ndarray]:
        g_copy = self.copy()

        shape = images[0].shape

        gauss = draw_gauss(
            -g_copy.angle, g_copy.width, g_copy.length, g_copy.center, shape, scaling=1.0)

        # draw angle and width if quality is more than 0.5
        mask_gauss_to_draw = gauss > 0.5

        gauss_scaled = gauss * self.quality

        mask2 = gauss_scaled > np.squeeze(images[0])
        mask = np.logical_and(mask_gauss_to_draw, mask2)
        images[0] = np.where(mask, gauss_scaled, np.squeeze(images[0]))
        images[1] = np.where(mask, g_copy.angle, np.squeeze(images[1]))
        images[2] = np.where(mask, g_copy.width, np.squeeze(images[2]))

        # draw margin (-1) where 0.1 < gauss <= 0.5 and no other label already exists
        if draw_margin:
            if full_margin:
                g_copy = self.copy()
                g_copy.length = self.length

                shape = images[0].shape

                rr, cc = polygon(g_copy.points[:, 1], g_copy.points[:, 0], shape)
                mask = images[0][rr, cc]
                mask[mask == 0] = -1
                images[0][rr, cc] = mask
            else:
                mask_margin = np.logical_and(gauss <= 0.5, gauss > 0.2)
                mask_no_label = images[0] == 0
                mask_draw_margin = np.logical_and(mask_margin, mask_no_label)
                images[0] = np.where(mask_draw_margin, -1, images[0].squeeze())

        return images

    def plot(self, image : np.ndarray, **kwargs) -> np.ndarray:
        p1, p2, p3, p4 = self.points

        thickness = kwargs.get('thickness', 1)

        cv2.line(image, (int(p2[0]),int(p2[1])), (int(p3[0]),int(p3[1])), (0,0,255), thickness)
        cv2.line(image, (int(p4[0]),int(p4[1])), (int(p1[0]),int(p1[1])), (0,0,255), thickness)
        cv2.line(image, (int(p1[0]),int(p1[1])), (int(p2[0]),int(p2[1])), (255,0,0), thickness * 3)
        cv2.line(image, (int(p3[0]),int(p3[1])), (int(p4[0]),int(p4[1])), (255,0,0), thickness * 3)

        return image

    def to_3d(self,
              depth_image: np.ndarray,
              intrinsic: PinholeCameraIntrinsic,
              depth_lookup_method: DepthLookupMethod = DepthLookupMethod.CENTER_POINT):

        from .grasp_3d import PrallelGripperGrasp3D

        # get base 3d grasp
        # this is missing width and heigt of gripper
        grasp_3d = super().to_3d(depth_image, intrinsic, depth_lookup_method)
        parallel_grasp = PrallelGripperGrasp3D.from_grasp3d(grasp_3d)

        return parallel_grasp

    def _compute_rotation_matrix(self, points_3d):
        upper_point = points_3d[:2].mean(axis=0)
        open_point = points_3d[1:2].mean(axis=0)
        open_points_vector = open_point - points_3d.mean(axis=0)
        upper_points_vector = upper_point - points_3d.mean(axis=0)
        open_point_norm = np.linalg.norm(open_points_vector, axis = 1).reshape(-1, 1)
        upper_point_norm = np.linalg.norm(upper_points_vector, axis = 1).reshape(-1, 1)

        unit_open_points_vector = open_points_vector / open_point_norm
        unit_upper_points_vector = upper_points_vector / upper_point_norm

        x_axis = np.array([0, 0, 1])
        rotation = np.dstack((x_axis, unit_open_points_vector, unit_upper_points_vector))

        return rotation
