import pytest

pytest.importorskip('graspnetAPI')

import numpy as np

from graspnetAPI.utils.eval_utils import collision_detection
from graspnetAPI.grasp import Grasp, GraspGroup

from nicr_grasping.collision.graspnet_checker import GraspNetChecker
from nicr_grasping.datatypes.grasp import ParallelGripperGrasp3D
from nicr_grasping.datatypes.intrinsics import PinholeCameraIntrinsic
from nicr_grasping.datatypes.grasp_conversion import CONVERTER_REGISTRY


def _create_pc():
    # generate pointcloud as grid with 0.05m spacing between -0.1 and 0.1
    x_ = np.linspace(-0.1, 0.1, 40)

    x, y, z = np.meshgrid(x_, x_, x_, indexing='ij')

    pc = np.stack([x, y, z], axis=-1).reshape(-1, 3)

    return pc


def test_against_graspnetapi():
    np.random.seed(1337)
    # pc = (np.random.random((10000, 3)) - 0.5) * 0.2
    pc = _create_pc()

    intrinsic = {
        'fx': 1,
        'fy': 1,
        'cx': 50,
        'cy': 50
    }

    grasp = ParallelGripperGrasp3D(0.1, 1, np.array([[0, 0, 0]]))
    grasp.height = 0.02

    graspnet_grasp = CONVERTER_REGISTRY.convert(grasp, Grasp)
    graspnet_graspgroup = GraspGroup(graspnet_grasp.grasp_array[np.newaxis])
    # list of grasps per model
    graspnet_grasp_list = [graspnet_graspgroup.grasp_group_array]

    checker = GraspNetChecker(PinholeCameraIntrinsic(**intrinsic))

    for point in pc:
        point = point[np.newaxis]
        checker.set_point_cloud(point)

        collision = checker.check_collision(grasp,
                                            frames={'camera': np.eye(4)},
                                            empty_thresh=0)
        collision_info = checker.collision_info

        collision_mask_list, empty_mask_list, seperated_collision_mask_list = collision_detection(
            graspnet_grasp_list,
            [point],
            [None], # they dont do anything with the dexnet models
            [np.eye(4)],
            point,
            empty_thresh=0,
            fill_seperated_masks=True
        )

        # because graspnet counts grasps with not enough points between the fingers as collisions
        # we need to recompute the collision mask from the additional info
        graspnet_collision = np.any((seperated_collision_mask_list[0][0] | seperated_collision_mask_list[0][1] | seperated_collision_mask_list[0][2]))

        # lm = np.load('/tmp/left_mask.npy')
        # rm = np.load('/tmp/right_mask.npy')
        # bm = np.load('/tmp/bottom_mask.npy')
        # im = np.load('/tmp/inner_mask.npy')

        assert graspnet_collision == collision

        # because we check each point individually, we can also check if the collision was deetected in
        # the correct part of the gripper
        assert seperated_collision_mask_list[0][0] == bool(collision_info['collision_left'])
        assert seperated_collision_mask_list[0][1] == bool(collision_info['collision_right'])
        assert seperated_collision_mask_list[0][2] == bool(collision_info['collision_base'])
