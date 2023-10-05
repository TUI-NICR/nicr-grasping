import os
import os.path as osp

import time
import json

import argparse

import numpy as np
import tqdm

from copy import deepcopy

from graspnetAPI import GraspNetEval, GraspGroup

from nicr_grasping.evaluation.evaluation import Scene, ObjectModel, CollisionObject, eval_grasps_on_model, EvalParameters, EvalResults
from nicr_grasping.datatypes.transform.pose import Pose
from nicr_grasping.datatypes.grasp import ParallelGripperGrasp3DList
from nicr_grasping.datatypes.grasp_conversion import CONVERTER_REGISTRY
from nicr_grasping.collision import PointCloudChecker
from nicr_grasping.collision.graspnet_checker import GraspNetChecker

MODEL_CACHE_DIR = '/tmp/model_cache'

GRASPNET_ROOT_PATH = '/datasets_nas/grasping/graspnet'
GRASPNET = GraspNetEval(GRASPNET_ROOT_PATH, 'kinect', 'test')

ID_TO_NAME_MAPPING_PATH = '/home/best3125/wrk/py-projects/released-code/nicr-grasping/scripts/graspnet/id_to_name_mapping.json'
with open(ID_TO_NAME_MAPPING_PATH, 'r') as f:
    id_to_name_mapping = json.load(f)

name_to_id_mapping = {v: int(k) for k, v in id_to_name_mapping.items()}


def _parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument('src_dir', type=str)
    ap.add_argument('scene_id', type=int)

    ap.add_argument('--output-path', type=str, default=None)

    ap.add_argument('--viz', action='store_true')
    ap.add_argument('--use-graspnet-collision', action='store_true')

    ap.add_argument('--verbose', action='store_true', help='If set print more information like timings.')

    return ap.parse_args()


class Timer:
    def __init__(self, name, level=0, quiet=False):
        self.name = name
        self.quiet = quiet
        self.level = level

    def __enter__(self):
        self.start = time.perf_counter()

    def __exit__(self, *args):
        self.end = time.perf_counter()
        if not self.quiet:
            print('\t' * self.level + f'{self.name}: {self.end - self.start} seconds')


def create_table_points(lx, ly, lz, dx=0, dy=0, dz=0, grid_size=0.01):
    '''
    **Input:**
    - lx:
    - ly:
    - lz:
    **Output:**
    - numpy array of the points with shape (-1, 3).
    '''
    xmap = np.linspace(0, lx, int(lx/grid_size))
    ymap = np.linspace(0, ly, int(ly/grid_size))
    zmap = np.linspace(0, lz, int(lz/grid_size))
    xmap, ymap, zmap = np.meshgrid(xmap, ymap, zmap, indexing='xy')
    xmap += dx
    ymap += dy
    zmap += dz
    points = np.stack([xmap, ymap, zmap], axis=-1)
    points = points.reshape([-1, 3])
    return points


def get_scene(scene_id, use_graspnet_collision=False):

    checker = GraspNetChecker() if use_graspnet_collision else PointCloudChecker()
    scene = Scene(collision_checker=checker)

    # collect object poses in world
    # ann_id = 0 as arbitrary ann_id as we transform object poses into world frame
    obj_list, pose_list, camera_pose, align_mat = GRASPNET.get_model_poses(scene_id, ann_id=0)
    cam_in_world = np.matmul(align_mat, camera_pose)

    object_poses = {
        obj_id: cam_in_world @ pose for obj_id, pose in zip(obj_list, pose_list)
    }

    # print(object_poses)

    global_object_ids = GRASPNET.getObjIds(scene_id)

    with Timer('load object models'):
        for object_id in global_object_ids:
            object_model = ObjectModel.from_dir(osp.join(GRASPNET_ROOT_PATH, 'models', f'{object_id:03d}'), model_name='textured')
            # TODO: add object pose in global frame
            object_model.pose = Pose.from_transformation_matrix(object_poses[object_id])

            scene.add_object(object_model)

    # create table as collision object
    table_points = create_table_points(1, 1, 0.05, -0.5, -0.5, -0.05, 0.008)
    table_object = CollisionObject(table_points)
    scene.add_static_collision_object(table_object)

    return scene


def main():
    args = _parse_args()

    params = EvalParameters(top_k=50, friction_coefficients=np.linspace(0.2, 1.2, 6).tolist())
    scene = get_scene(args.scene_id, use_graspnet_collision=args.use_graspnet_collision)

    scene_dir = osp.join(args.src_dir, f'scene_{args.scene_id:04d}', 'kinect')

    for step in tqdm.tqdm(range(256)):
        with Timer('eval step', level=0, quiet=not args.verbose):

            eval_res = None

            # get camera pose for this ann_id
            # we need this pose because graspnetAPI cropped their point cloud in camera frame
            _, _, camera_pose, align_mat = GRASPNET.get_model_poses(args.scene_id, ann_id=step)
            world_to_cam = np.linalg.inv(np.matmul(align_mat, camera_pose))

            # load grasps
            graspnet_grasps = GraspGroup().from_npy(osp.join(scene_dir, f'{step:04d}.npy'))
            graspnet_grasps = graspnet_grasps.sort_by_score()
            local_object_ids = graspnet_grasps.object_ids.astype(int)

            if -1 in local_object_ids:
                raise ValueError('Object id -1 found in graspnet grasps. This script assumes predefined object ids!')

            for local_object_id in np.unique(local_object_ids):
                with Timer('eval object', level=1, quiet=not args.verbose):
                    belongs_to_object = local_object_ids == local_object_id

                    if not np.any(belongs_to_object):
                        continue

                    object_grasps_graspnet = graspnet_grasps[belongs_to_object]
                    grasps_copy = deepcopy(object_grasps_graspnet)

                    grasp_list = CONVERTER_REGISTRY.convert(object_grasps_graspnet, ParallelGripperGrasp3DList)

                    grasp_list.transform(np.linalg.inv(world_to_cam))

                    # set height of grasps to 0.02 as is standard for graspnet
                    # otherwise it would be 1/3 * width
                    for grasp in grasp_list:
                        grasp.height = 0.02

                    suppressed = grasp_list.nms(0.03, 30)
                    grasps_copy = grasps_copy.nms(0.03, np.deg2rad(30))

                    if args.viz:
                        scene.show(grasp_list, simple_grasps=False)

                    with Timer('check collision', level=2, quiet=not args.verbose):
                        collision_res, collision_infos = scene.check_collision(grasp_list, object_id=local_object_id, world_to_camera=world_to_cam)

                    with Timer('grasp evaluation', level=2, quiet=not args.verbose):
                        res = eval_grasps_on_model(grasp_list, scene._objects[local_object_id], params)

                    res.add_info('collision', collision_res)
                    res.add_info('collision_base', 0)
                    res.add_info('collision_left', 0)
                    res.add_info('collision_right', 0)

                    res.add_info('object_id', local_object_id)
                    res.add_info('suppressed', suppressed)

                    for gi in range(len(grasp_list)):
                        res.update_info_of_grasp('collision_base', gi, collision_infos[gi]['base'])
                        res.update_info_of_grasp('collision_left', gi, collision_infos[gi]['left'])
                        res.update_info_of_grasp('collision_right', gi, collision_infos[gi]['right'])

                    if eval_res is None:
                        eval_res = res
                    else:
                        eval_res += res

            if args.output_path is not None and eval_res is not None:
                eval_res.save(osp.join(args.output_path, f'{step:04d}.csv'))

    # collect all saved results and compute ap for this scene
    step_aps = []
    for step in range(256):
        try:
            step_result = EvalResults.from_csv(osp.join(args.output_path, f'{step:04d}.csv'))
            step_aps.append(step_result.compute_ap(collision_filtered=True))
        except FileNotFoundError:
            continue
    print(np.mean(step_aps))


if __name__ == '__main__':
    main()
