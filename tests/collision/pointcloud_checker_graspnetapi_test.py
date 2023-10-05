import pytest

pytest.importorskip('graspnetAPI')

import numpy as np
import cv2

from graspnetAPI.utils.eval_utils import collision_detection
from graspnetAPI.grasp import Grasp, GraspGroup

from nicr_grasping.collision import PointCloudChecker
from nicr_grasping.datatypes import ParallelGripperGrasp3D, PinholeCameraIntrinsic
from nicr_grasping.datatypes.grasp_conversion import CONVERTER_REGISTRY


def test_against_graspnetapi():
    np.random.seed(1337)
    pc = (np.random.random((10000, 3)) - 0.5) * 0.2

    intrinsic = {
        'fx': 1,
        'fy': 1,
        'cx': 50,
        'cy': 50
    }

    checker = PointCloudChecker(PinholeCameraIntrinsic(**intrinsic))
    checker.set_point_cloud(pc)

    grasp = ParallelGripperGrasp3D(0.1, 1, np.array([[0, 0, 0]]))
    grasp.height = 0.02

    graspnet_grasp = CONVERTER_REGISTRY.convert(grasp, Grasp)
    graspnet_graspgroup = GraspGroup(graspnet_grasp.grasp_array[np.newaxis])
    # list of grasps per model
    graspnet_grasp_list = [graspnet_graspgroup.grasp_group_array]

    collision = checker.check_collision(grasp)
    collision_info = checker.collision_info

    collision_mask_list, empty_mask_list, seperated_collision_mask_list = collision_detection(
        graspnet_grasp_list,
        [pc],
        [None], # they dont do anything with the dexnet models
        [np.eye(4)],
        pc,
        fill_seperated_masks=True
    )

    assert collision_mask_list[0] == collision
    assert seperated_collision_mask_list[0][0] == bool(collision_info['left'])
    assert seperated_collision_mask_list[0][1] == bool(collision_info['right'])
    assert seperated_collision_mask_list[0][2] == bool(collision_info['base'])
