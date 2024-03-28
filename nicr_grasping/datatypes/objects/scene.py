from typing import Any, Dict, Union, List

import numpy as np
import open3d.visualization.gui as gui

from . import ObjectModel, CollisionObject
from ..grasp import ParallelGripperGrasp3DList
from ...evaluation import EvalResults

from ...collision.pointcloud_checker import PointCloudChecker

from ...visualization.visualizer import GraspEvalVisualizer

from . import logger as baselogger
logger = baselogger.getChild('scene')  # type: ignore


class Scene:

    STATIC_COLLISION_OBJECT_ID_OFFSET = 1000

    def __init__(self,
                 collision_checker: PointCloudChecker) -> None:
        self._objects: List[ObjectModel] = []
        self._static_collision_objects: List[CollisionObject] = []
        self._collision_scene_needs_update: bool = True
        self._collision_scene: Union[np.ndarray, None] = None

        logger.info('Using collision checker:', collision_checker.__class__.__name__)
        self._collision_checker: PointCloudChecker = collision_checker

        self.frames: Dict[str, np.ndarray] = {}

    def add_object(self, obj: ObjectModel) -> None:
        self._objects.append(obj)
        self._collision_scene_needs_update = True

    def add_static_collision_object(self, obj: CollisionObject) -> None:
        self._static_collision_objects.append(obj)
        self._collision_scene_needs_update = True

    @property
    def collision_scene(self) -> Union[np.ndarray, None]:
        if not self._collision_scene_needs_update:
            return self._collision_scene
        else:
            self._compute_collision_scene()
            return self._collision_scene

    @property
    def objects(self) -> List[ObjectModel]:
        return self._objects

    def check_collision(self, grasps: ParallelGripperGrasp3DList,
                        **kwargs: Any) -> tuple[np.ndarray, List[dict]]:
        if self._collision_scene_needs_update:
            self._compute_collision_scene()

        is_in_collision = np.zeros(len(grasps), dtype=bool)
        collision_infos = []

        for gi, grasp in enumerate(grasps):
            collision = self._collision_checker.check_collision(grasp, self.frames, **kwargs)
            is_in_collision[gi] = collision

            collision_infos.append(self._collision_checker.collision_info)

        return is_in_collision, collision_infos

    def _compute_collision_scene(self) -> None:
        point_clouds = []
        point_labels = []

        # collect samples pointcloud from all objects
        for oi, gr_obj in enumerate(self._objects):
            obj_points = gr_obj.sample_points()
            obj_points = np.concatenate([obj_points, np.ones((len(obj_points), 1))], axis=-1)
            obj_points = gr_obj.pose.transformation_matrix @ obj_points.T
            obj_points = obj_points.T[:, :3]
            point_clouds.append(obj_points)
            point_labels.append(np.ones((len(obj_points), 1)) * oi)

        for oi, col_obj in enumerate(self._static_collision_objects):
            object_points = col_obj.sample_points()
            point_clouds.append(object_points)
            point_labels.append(np.ones((len(object_points), 1)) * oi + self.STATIC_COLLISION_OBJECT_ID_OFFSET)

        collision_scene = np.concatenate(point_clouds, axis=0)
        self._collision_scene = collision_scene
        point_labels = np.concatenate(point_labels, axis=0)

        # set point cloud in collision checker
        self._collision_checker.set_point_cloud(collision_scene,
                                                labels=point_labels)  # type: ignore
        self._collision_scene_needs_update = False

    def assign_grasps_to_objects(self, grasps: ParallelGripperGrasp3DList) -> None:
        for grasp in grasps:
            # find object that is closest to grasp
            closest_obj = None
            closest_dist = np.inf
            for oi, obj in enumerate(self._objects):
                grasp_copy = grasp.copy()
                grasp_copy.transform(obj.pose.inverse())
                dist = np.linalg.norm(
                    obj.sample_points() - grasp_copy.position,
                    axis=1
                ).min()
                if dist < closest_dist:
                    closest_obj = oi
                    closest_dist = dist
            grasp.object_id = closest_obj

    def show(self,
             grasps: Union[ParallelGripperGrasp3DList, None] = None,
             eval_results: Union[EvalResults, None] = None) -> None:

        gui.Application.instance.initialize()
        vis = GraspEvalVisualizer(self)

        if grasps is not None:
            vis.add_grasps(grasps)
        if eval_results is not None:
            vis.add_evaluation_results(eval_results)

        gui.Application.instance.run()
