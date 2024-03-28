import os.path as osp
from typing import Union, Any, List
from pathlib import Path

import json
import functools

import numpy as np

from . import EvalParameters, EvalResults
from ..datatypes.objects import Scene, CollisionObject
from ..datatypes.transform import Pose
from ..datatypes.grasp import ParallelGripperGrasp3DList
from ..collision import PointCloudChecker, GraspNetChecker
from .graspnet.graspnet_funcs import get_graspnet_eval_instance

from .evaluator_base import Evaluator, Sample

from .graspnet import name_to_id
from .graspnet.utils import create_table_points, _get_graspnet_objects, _get_object_ids

from . import logger as baselogger
logger = baselogger.getChild('evaluator')

GRASPNET = get_graspnet_eval_instance(camera='kinect', split='all')


class GraspTrackEvaluator(Evaluator):

    RESULTS_FILE_NAME = 'results.csv'

    def __init__(self,
                 split: str,
                 root_dir: Union[Path, str],
                 params: EvalParameters,
                 save_dir: Union[Path, str, None] = None,
                 **kwargs: Any):

        assert split in ['test_seen', 'test_similar', 'test_novel']
        super().__init__(root_dir, params, save_dir=save_dir, split=split, **kwargs)

        # self._graspnet = GraspNetEval(graspnet_dataset_path(), camera='kinect', split=split)
        self._use_graspnet_collision = kwargs.get('use_graspnet_collision', False)

        self._grasp_file = kwargs.get('grasp_file', 'tracking_GraspPoseBelief.json')

    def _get_save_path(self, sample: Sample) -> Path:
        if self._save_dir is None:
            raise ValueError('No save dir set!')

        scene_id, ann_id = sample
        return self._save_dir / self._split / f'{scene_id}' / 'step' / f'{ann_id:06d}'

    def save_evaluation(self, sample: Sample, evaluation_result: EvalResults, grasps: ParallelGripperGrasp3DList, **kwargs: Any) -> None:
        # TODO: this function can maybe moved to base class as _get_save_path is the only thing that changes
        save_dir = self._get_save_path(sample)

        save_dir.mkdir(parents=True, exist_ok=True)

        evaluation_result.save(str(save_dir / self.RESULTS_FILE_NAME))
        grasps.save(str(save_dir / 'grasps.pkl'))

        with (save_dir / 'sample.json').open('w') as f:
            json.dump(sample, f)

    def load_evaluation(self, sample: Sample, **kwargs: Any) -> tuple[EvalResults, ParallelGripperGrasp3DList]:
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

        grasps = ParallelGripperGrasp3DList()

        obj_dirs = self._root_dir.glob(f'{scene_id}__*')
        for obj_dir in obj_dirs:
            root = obj_dir / 'step' / f'{ann_id:06d}'
            root = obj_dir / f'{ann_id}'

            object_name = obj_dir.name.split('__')[1]

            global_object_id = name_to_id(object_name)
            object_ids = _get_object_ids(scene_id)
            local_object_id = object_ids.index(global_object_id)

            logger.info(f'Loading grasps from {root}')
            file_name = osp.join(root, self._grasp_file)
            with open(file_name, 'r') as f:
                json_object = json.load(f)

            if len(json_object) == 0:
                # len 0 means we had no instance tracked by the tracker
                logger.info('No instance was tracked. JSON is empty.')
                continue
            elif len(json_object) > 1:
                raise ValueError('More than one instance was tracked.')

            has_grasps = False
            for gs in json_object:
                if 'Second' in gs:
                    if len(gs['Second']) == 0:
                        logger.info('no grasps found')
                        continue
                    else:
                        grasp_list = ParallelGripperGrasp3DList.from_mira_json(gs['Second'])
                        has_grasps = True
                elif 'DetId' in gs:
                    # we have the raw detections which are untracked
                    grasp_list = ParallelGripperGrasp3DList.from_mira_json(gs['Value'])
                    has_grasps = True

            if not has_grasps:
                continue

            # set height of grasps to 0.02 as is standard for graspnet
            # otherwise it would be 1/3 * width
            for grasp in grasp_list:
                grasp.height = 0.02
                grasp.object_id = local_object_id

            file_name = osp.join(root, 'tracking_ObjectPositionsBelief.json')

            with open(file_name, 'r') as f:
                object_pose = Pose.from_mira_json(json.load(f)[0]['Second'])

            # transform into global frame
            grasp_list.transform(object_pose.transformation_matrix)

            grasps.extend(grasp_list)

        return grasps
