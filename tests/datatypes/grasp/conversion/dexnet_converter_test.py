import pytest

pytest.importorskip('graspnetAPI')

import numpy as np

from scipy.spatial.transform import Rotation as R

from graspnetAPI import Grasp, GraspGroup
from graspnetAPI.utils.eval_utils import transform_points, matrix_to_dexnet_params

from nicr_grasping.datatypes.grasp.grasp_3d import ParallelGripperGrasp3D
from nicr_grasping.external.dexnet.grasping.grasp import ParallelJawPtGrasp3D

from nicr_grasping.datatypes.grasp_conversion import CONVERTER_REGISTRY


OBJECT_POSE = np.eye(4)


def graspnetapi_to_dexnet(grasp):
    grasps = GraspGroup(np.array([grasp.grasp_array])).grasp_group_array
    grasp_points = grasps[:, 13:16]
    grasp_poses = grasps[:, 4:13].reshape([-1,3,3])
    grasp_depths = grasps[:, 3]
    grasp_widths = grasps[:, 1]

    grasp_id = 0

    grasp_point = grasp_points[grasp_id]
    R = grasp_poses[grasp_id]
    width = grasp_widths[grasp_id]
    depth = grasp_depths[grasp_id]
    center = np.array([depth, 0, 0]).reshape([3, 1]) # gripper coordinate
    center = np.dot(grasp_poses[grasp_id], center).reshape([3])
    center = (center + grasp_point).reshape([1,3]) # camera coordinate
    center = transform_points(center, np.linalg.inv(OBJECT_POSE)).reshape([3]) # object coordinate
    R = np.dot(OBJECT_POSE[:3,:3].T, R)
    binormal, approach_angle = matrix_to_dexnet_params(R)
    dexnet_grasp = ParallelJawPtGrasp3D(ParallelJawPtGrasp3D.configuration_from_params(
                                center, binormal, width, approach_angle), depth)

    return dexnet_grasp


def test_dexnet_conversion():
    grasp = ParallelGripperGrasp3D()

    random_rotation = R.random().as_matrix()
    grasp.orientation = random_rotation

    # transform into object frame
    grasp.transform(OBJECT_POSE)

    graspnet_grasp = CONVERTER_REGISTRY.convert(grasp, Grasp)

    graspnet_dexnet = graspnetapi_to_dexnet(graspnet_grasp)

    dexnet_grasp = CONVERTER_REGISTRY.convert(grasp, ParallelJawPtGrasp3D)

    np.testing.assert_allclose(dexnet_grasp.center, graspnet_dexnet.center, atol=1e-16)
    np.testing.assert_allclose(dexnet_grasp.axis_, graspnet_dexnet.axis_, atol=1e-16)
    np.testing.assert_allclose(dexnet_grasp.max_grasp_width_, graspnet_dexnet.max_grasp_width_)
    np.testing.assert_allclose(dexnet_grasp.jaw_width_, graspnet_dexnet.jaw_width_)
    np.testing.assert_allclose(dexnet_grasp.min_grasp_width_, graspnet_dexnet.min_grasp_width_)
    np.testing.assert_allclose(dexnet_grasp.approach_angle_, graspnet_dexnet.approach_angle_)
    np.testing.assert_allclose(dexnet_grasp.max_grasp_depth, graspnet_dexnet.max_grasp_depth)
