import os.path as osp
from typing import Union, Any, List
from pathlib import Path

import json
import functools

import numpy as np

from graspnetAPI import GraspGroup

from . import EvalParameters, EvalResults
from ..datatypes.objects import Scene, CollisionObject
from ..datatypes.transform import Pose
from ..datatypes.grasp import ParallelGripperGrasp3DList
from ..datatypes.grasp_conversion import CONVERTER_REGISTRY
from ..collision import PointCloudChecker, GraspNetChecker
from .graspnet.graspnet_funcs import get_graspnet_eval_instance

from .evaluator_base import Evaluator, Sample

from .graspnet.utils import create_table_points, _get_graspnet_objects, _get_object_ids

from . import logger as baselogger
logger = baselogger.getChild('evaluator')

GRASPNET = get_graspnet_eval_instance(camera='kinect', split='all')


class GraspNetEvaluator(Evaluator):

    RESULTS_FILE_NAME = 'results.csv'

    def __init__(self,
                 split: str,
                 root_dir: Union[Path, str],
                 params: EvalParameters,
                 save_dir: Union[Path, str],
                 max_width: float = 0.1,
                 **kwargs: Any):

        assert split in ['test_seen', 'test_similar', 'test_novel']
        super().__init__(root_dir, params, save_dir=save_dir, split=split, **kwargs)

        # self._graspnet = GraspNetEval(graspnet_dataset_path(), camera='kinect', split=split)
        self._use_graspnet_collision = kwargs.get('use_graspnet_collision', False)
        self._max_width = max_width

    def _get_save_path(self, sample: Sample) -> Path:
        if self._save_dir is None:
            raise ValueError('No save dir set!')

        scene_id, ann_id = sample
        return self._save_dir / self._split / f'{scene_id}' / 'step' / f'{ann_id:06d}'

    def save_evaluation(self, sample: Sample, evaluation_result: EvalResults, grasps: ParallelGripperGrasp3DList,
                        **kwargs: Any) -> None:
        scene_id, ann_id = sample
        save_dir = self._get_save_path(sample)

        save_dir.mkdir(parents=True, exist_ok=True)

        evaluation_result.save(str(save_dir / self.RESULTS_FILE_NAME))
        grasps.save(str(save_dir / 'grasps.pkl'))

        with (save_dir / 'sample.json').open('w') as f:
            json.dump(sample, f)

    def load_evaluation(self, sample: Sample, **kwarg: Any) -> tuple[EvalResults, ParallelGripperGrasp3DList]:
        scene_id, ann_id = sample
        scene_id = int(scene_id)
        ann_id = int(ann_id)

        save_dir = self._get_save_path(sample)

        evaluation_result = EvalResults.from_csv(str(save_dir / self.RESULTS_FILE_NAME))

        grasps = ParallelGripperGrasp3DList.load(str(save_dir / 'grasps.pkl'))

        return evaluation_result, grasps

    @functools.cache
    def create_scene(self, sample: Sample) -> Scene:
        logger.debug(f'Creating scene for sample {sample}')
        scene_id, ann_id = sample
        scene_id = int(scene_id)
        ann_id = int(ann_id)

        objects = _get_graspnet_objects(scene_id)

        checker = GraspNetChecker() if self._use_graspnet_collision else PointCloudChecker()
        scene = Scene(collision_checker=checker)

        # collect object poses in world
        obj_list, pose_list, camera_pose, align_mat = GRASPNET.get_model_poses(scene_id, ann_id=ann_id)
        cam_in_world = np.matmul(align_mat, camera_pose)

        object_poses = {
            obj_id: cam_in_world @ pose for obj_id, pose in zip(obj_list, pose_list)
        }

        global_object_ids = _get_object_ids(scene_id)

        logger.info(f'Loading {len(global_object_ids)} objects. This might take a while.')

        for object_id in global_object_ids:
            object_model = objects[object_id]
            object_model.pose = Pose.from_transformation_matrix(object_poses[object_id])

            scene.add_object(object_model)

        # create table as collision object
        table_points = create_table_points(1, 1, 0.05, -0.5, -0.5, -0.05, 0.008)
        table_object = CollisionObject(table_points)
        scene.add_static_collision_object(table_object)

        scene.frames['camera'] = cam_in_world

        return scene

    def compute_samples(self, **kwargs: Any) -> List[Sample]:
        sample_params = []

        sample_filter_func = kwargs.get('sample_filter_func', None)
        if sample_filter_func is None:
            sample_filter_func = lambda x: True

        if self._split == 'test_seen':
            scenes = range(100, 130)
        elif self._split == 'test_similar':
            scenes = range(130, 160)
        elif self._split == 'test_novel':
            scenes = range(160, 190)
        else:
            raise ValueError(f'Unknown split {self._split}')

        for scene_id in scenes:
            # to be consistent with GraspTrack evaluation we skip the first sample
            # as no tracking is done with GraspTrack
            for step in range(1, 256):
                sample = Sample(scene_id, step)
                if sample_filter_func(sample):
                    sample_params.append(sample)

        return sample_params

    def can_skip_sample(self, sample: Sample) -> bool:
        if self._save_dir is None:
            return False

        root = self._get_save_path(sample)
        result_file = root / self.RESULTS_FILE_NAME

        return result_file.exists()

    def load_grasps(self, sample: Sample, **kwargs: Any) -> ParallelGripperGrasp3DList:
        scene_id, ann_id = sample

        grasp_group = GraspGroup().from_npy(
            osp.join(self._root_dir, f'scene_{scene_id:04d}', 'kinect', f'{ann_id:04d}.npy')
        )

        # remove grasps with failed depth lookup
        grasp_group.remove(np.argwhere(grasp_group.translations[:, 2] <= 0))
        # grasp_group.translations[:, 2] += 0.02

        # TODO: use np.clip?
        gg_array = grasp_group.grasp_group_array
        min_width_mask = (gg_array[:, 1] < 0)
        max_width_mask = (gg_array[:, 1] > self._max_width)
        gg_array[min_width_mask, 1] = 0
        gg_array[max_width_mask, 1] = self._max_width
        grasp_group.grasp_group_array = gg_array

        # because graspnetAPI evaluation pipeline expects grasps given in camera frame
        # but we expect them in world frame, we transform them here
        _, _, camera_pose, align_mat = GRASPNET.get_model_poses(scene_id, ann_id=ann_id)
        cam_in_world = np.matmul(align_mat, camera_pose)

        grasp_group = grasp_group.transform(cam_in_world)

        grasps = CONVERTER_REGISTRY.convert(grasp_group, ParallelGripperGrasp3DList)

        for grasp in grasps:
            grasp.height = 0.02

        return grasps
