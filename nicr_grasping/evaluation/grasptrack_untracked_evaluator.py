import os.path as osp
from typing import Union, Any
from pathlib import Path

import json

import numpy as np

from . import EvalParameters
from ..datatypes.grasp import ParallelGripperGrasp3DList

from .grasptrack_evaluator import GraspTrackEvaluator, GRASPNET, Sample, _get_object_ids
from .graspnet import name_to_id

from . import logger as baselogger
logger = baselogger.getChild('evaluator')


class UntrackedGRConvNetEvaluator(GraspTrackEvaluator):

    RESULTS_FILE_NAME: str = 'results.csv'

    def __init__(self,
                 split: str,
                 root_dir: Union[Path, str],
                 params: EvalParameters,
                 save_dir: Union[Path, str, None] = None,
                 **kwargs: Any):

        super().__init__(split, root_dir, params, save_dir=save_dir, **kwargs)

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

            _, _, camera_pose, align_mat = GRASPNET.get_model_poses(scene_id, ann_id=ann_id)
            cam_in_world = np.matmul(align_mat, camera_pose)

            # transform into global frame
            grasp_list.transform(cam_in_world)

            grasps.extend(grasp_list)

        return grasps
