import numpy as np
from scipy.spatial.transform import Rotation

from . import logger

from ..grasp.grasp_3d import ParallelGripperGrasp3D
from ..grasp.grasp_lists import ParallelGripperGrasp3DList
from ...external.dexnet.grasping.grasp import ParallelJawPtGrasp3D

from . import CONVERTER_REGISTRY

PRINTED_WARNING = False


def matrix_to_dexnet_params(matrix: np.ndarray) -> tuple[np.ndarray, float]:
    # code taken from graspnetAPI
    # we orthonormalize the matrix through scipy
    # otherwise we might encounter numerical issues
    # and invalid parameters for arccos
    matrix = Rotation.from_matrix(matrix).as_matrix()

    approach = matrix[:, 0]
    binormal = matrix[:, 1]
    axis_y = binormal
    axis_x = np.array([axis_y[1], -axis_y[0], 0])
    if np.linalg.norm(axis_x) == 0:
        axis_x = np.array([1, 0, 0])
    axis_x = axis_x / np.linalg.norm(axis_x)
    axis_y = axis_y / np.linalg.norm(axis_y)
    axis_z = np.cross(axis_x, axis_y)
    R = np.c_[axis_x, np.c_[axis_y, axis_z]]
    approach = R.T.dot(approach)
    cos_t, sin_t = approach[0], -approach[2]
    angle = np.arccos(cos_t)
    if sin_t < 0:
        angle = np.pi * 2 - angle
    return binormal, angle


def parallelgrippergrasp3d_to_dexnetgrasp(grasp: ParallelGripperGrasp3D) -> ParallelJawPtGrasp3D:

    global PRINTED_WARNING
    if not PRINTED_WARNING:
        logger.warning("Direct conversion to dexnet grasps is not optimized! Use with care")
        PRINTED_WARNING = True

    center = grasp.position

    rotation_fix = Rotation.from_euler('xy', [90, -90], degrees=True).as_matrix()
    grasp.orientation = grasp.orientation @ rotation_fix

    # center = np.array([0.02, 0, 0]).reshape([3, 1]) # gripper coordinate
    center_offset = np.eye(4)
    center_offset[0, 3] = 0.02

    grasp_pose = grasp.transformation_matrix @ center_offset
    center = grasp_pose[:3, 3]

    width = grasp.width
    depth = 0.02
    binormal, approach_angle = matrix_to_dexnet_params(grasp.orientation)

    res = ParallelJawPtGrasp3D(
        ParallelJawPtGrasp3D.configuration_from_params(
                                            center, binormal, width, approach_angle), depth
    )

    return res


CONVERTER_REGISTRY.register(ParallelGripperGrasp3D, ParallelJawPtGrasp3D, parallelgrippergrasp3d_to_dexnetgrasp)
