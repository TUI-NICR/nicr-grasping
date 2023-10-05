import pickle
import bz2

from collections import UserList
from typing import Tuple, List, Union

from pathlib import Path

import numpy as np
import rtree.index
from copy import deepcopy

from nicr_grasping.datatypes.grasp.grasp_3d import Grasp3D

from .grasp_base import Grasp
from .grasp_2d import Grasp2D, RectangleGrasp, RectangleGraspDrawingMode
from .grasp_3d import Grasp3D, ParallelGripperGrasp3D

from ..transform.pose import Pose
from ..transform.metrics import DifferenceType, difference

class GraspList(UserList):
    def __init__(self, grasps: List[Grasp] = []) -> None:
        super().__init__(grasps)
        # self.grasps: List[Grasp] = grasps

    def add(self, grasp : Grasp) -> None:
        self.append(grasp)
        # self.grasps.append(grasp)

    def __eq__(self, __o: object) -> bool:
        if isinstance(__o, GraspList):
            same_len = len(self) == len(__o)
            if not same_len:
                return False
            same_els = [g1 == g2 for g1, g2 in zip(self, __o)]
            return same_len and all(same_els)

    def plot(self, image : np.ndarray, **kwargs) -> np.ndarray:
        raise NotImplementedError()

    @property
    def grasps(self):
        return self.data
    # def to_type(self, to_type):
    #     return CONVERTER_REGISTRY.convert(self, to_type)

    def sort(self, reverse : bool = False) -> None:
        self.data = sorted(self.data, key=lambda x : x.quality, reverse=not reverse)

    def transform(self, transform):
        transformation_matrix = np.eye(4)
        if isinstance(transform, Pose):
            transformation_matrix = transform.transformation_matrix
        elif isinstance(transform, np.ndarray):
            transformation_matrix = transform
        else:
            raise TypeError("transform must be either a Pose or a transformation matrix")

        for grasp in self:
            grasp.transform(transformation_matrix)

    def copy(self) -> "GraspList":
        return deepcopy(self)


class Grasp2DList(GraspList):
    def __init__(self, grasps: List[Grasp2D] = []) -> None:
        super().__init__(grasps=grasps)

    def plot(self, image : np.ndarray, **kwargs) -> np.ndarray:
        for grasp in self.data:
            image = grasp.plot(image, **kwargs)

        return image

    def scale(self, scale_factor: float):
        [g.scale(scale_factor) for g in self]


class Grasp3DList(GraspList):
    def __init__(self, grasps: List[Grasp3D] = []) -> None:
        super().__init__(grasps=grasps)

    def nms(self, translation_threshold, rotation_threshold):
        # sort grasps
        self.sort()

        suppressed = np.zeros(len(self), dtype=bool)
        for gi, grasp in enumerate(self):

            if suppressed[gi]:
                continue

            for gj in range(gi+1, len(self)):
                # compute difference
                gi_pose = Pose.from_transformation_matrix(grasp.transformation_matrix)
                gj_pose = Pose.from_transformation_matrix(self[gj].transformation_matrix)

                diff_trans = difference(gi_pose, gj_pose, difference_type=DifferenceType.EUCLIDEAN)
                diff_rot = difference(gi_pose, gj_pose, difference_type=DifferenceType.ROTATION)

                if diff_trans['error'] < translation_threshold and diff_rot['error'] < rotation_threshold:
                    suppressed[gj] = True

        return suppressed


class RectangleGraspList(Grasp2DList):
    def __init__(self, grasps: List[RectangleGrasp] = []) -> None:
        super().__init__(grasps=grasps)

    def save(self,
             file_path: Union[Path, str],
             compressed: bool = False):
        assert Path(file_path).suffix == '.pkl'
        if compressed:
            with bz2.BZ2File(str(file_path) + '.bz', 'wb') as f:
                pickle.dump(self.grasps, f)
        else:
            with open(str(file_path), 'wb') as f:
                pickle.dump(self.grasps, f)

    @classmethod
    def from_points(cls, points, qualities):
        grasps = []
        for i in range(len(points)):
            g = RectangleGrasp.from_points(points[i].T)
            g.quality = qualities[i]
            grasps.append(g)
        return cls(grasps)

    @classmethod
    def load_from_file(cls, file_path: Union[Path, str]):
        suffixes = Path(file_path).suffixes
        if '.npy' in suffixes:
            # for backwards compatibility
            with bz2.BZ2File(str(file_path), 'rb') as f:
                grasps = pickle.load(f)
        elif '.bz' in suffixes:
            with bz2.BZ2File(str(file_path), 'rb') as f:
                grasps = pickle.load(f)
        else:
            with open(str(file_path), 'rb') as f:
                grasps = pickle.loads(f.read().replace(b'grasp_benchmark',b'nicr_grasping'))
        return cls(grasps)

    @classmethod
    def load_from_mira_json(cls, json_object: dict):
        grasps = [
            RectangleGrasp.from_mira_json(grasp_json) for grasp_json in json_object
        ]
        obj = cls(grasps)
        return obj

    def iou(self, grasp: RectangleGrasp) -> np.ndarray:
        """Function for compupting IoU of grasp against this list of grasps.
        Code taken from https://codereview.stackexchange.com/questions/204017/intersection-over-union-for-rotated-rectangles

        Parameters
        ----------
        grasp : RectangleGrasp
            The grasp to compare to the other grasps.

        Returns
        -------
        np.ndarray
            Array with iou for every grasped contained in this grasp list.
        """
        grasp_poligons = [g.to_polygon() for g in self]

        ious = np.zeros((len(self),))

        index = rtree.index.Index()
        for i, poly in enumerate(grasp_poligons):
            index.insert(i, poly.bounds)

        grasp_polygon_to_compare = grasp.to_polygon()
        for i in index.intersection(grasp_polygon_to_compare.bounds):
            poly = grasp_poligons[i]
            intersection = poly.intersection(grasp_polygon_to_compare).area
            if intersection:
                ious[i] = intersection / (poly.area + grasp_polygon_to_compare.area - intersection)

        return ious

    def create_sample_images(self,
                             shape: Tuple[int],
                             position: bool=True,
                             angle: bool=True,
                             width: bool=True,
                             mode: RectangleGraspDrawingMode=RectangleGraspDrawingMode.INNER_RECTANGLE) -> List[Union[np.ndarray, None]]:
        """Plot all GraspRectangles as solid rectangles in a numpy array, e.g. as network training data.

        Parameters
        ----------
        shape : Tuple[int]
            Output shape. Tuple of ints
        position : bool, optional
            Wether to generate position/quality map, by default True
        angle : bool, optional
            Wether to generate angle map, by default True
        width : bool, optional
            Wether to generate width map, by default True

        Returns
        -------
        List[np.ndarray]
            List of all generated maps as ndarrays or None if map was not generated.
        """
        if position:
            pos_out = np.zeros(shape)
        else:
            pos_out = None
        if angle:
            ang_out = np.zeros(shape)
        else:
            ang_out = None
        if width:
            width_out = np.zeros(shape)
        else:
            width_out = None

        labels = [pos_out, ang_out, width_out]

        self.sort(reverse=True)

        for gr in self.grasps:
            labels = gr.draw_label(labels, mode=mode)

        return labels


class ParallelGripperGrasp3DList(Grasp3DList):
    def __init__(self, grasps: List[ParallelGripperGrasp3D] = []) -> None:
        super().__init__(grasps)

    @classmethod
    def from_mira_json(cls, json_object: dict):
        grasps = [
            ParallelGripperGrasp3D.from_mira_json(grasp_json) for grasp_json in json_object
        ]
        obj = cls(grasps)
        return obj
