import pytest
import numpy as np
from nicr_grasping.datatypes.grasp import RectangleGrasp, RectangleGraspList, Grasp3D
from nicr_grasping.datatypes.grasp.grasp_3d import ParallelGripperGrasp3D
from nicr_grasping.datatypes.intrinsics import PinholeCameraIntrinsic

from nicr_grasping import GRASPNET_INSTALLED


@pytest.fixture
@pytest.mark.skipif(not GRASPNET_INSTALLED, reason="graspnetAPI was not installed")
def graspnet_rectgrasp():
    from graspnetAPI.grasp import RectGrasp

    # [center_x, center_y, open_x, open_y, height, score, object_id]
    grasp_params = [30, 60, 50, 55, 10, 0.5, 0]
    grasp_params = np.asarray(grasp_params)
    graspnet_grasp = RectGrasp(grasp_params)
    return graspnet_grasp

@pytest.fixture
def rectangle_grasp():
    return RectangleGrasp(0.1, np.array([[50, 20]]), 10, 1.2, 5)

@pytest.fixture
def other_rectangle_grasp():
    return RectangleGrasp(0.2, np.array([[30, 20]]), 10, -2, 5)

@pytest.fixture
def rectangle_grasp_list(rectangle_grasp, other_rectangle_grasp):
    grasps = RectangleGraspList([
        rectangle_grasp,
        other_rectangle_grasp
    ])

    return grasps

@pytest.fixture
def other_rectangle_grasp_list(rectangle_grasp, other_rectangle_grasp):
    grasps = RectangleGraspList([
        other_rectangle_grasp,
        rectangle_grasp
    ])

    return grasps

@pytest.fixture
def grasp_3d():
    grasp = Grasp3D(1,
                     np.array([[1, 0.5, 0.7]]),
                     np.eye(3))

    return grasp

@pytest.fixture
def parallel_gripper_grasp():
    return ParallelGripperGrasp3D(
        1,
        np.array([[0.5, 0.4, 0.8]]),
        points=np.array([[0, 0, 0.8],
                         [1, 0, 0.8],
                         [1, 0.8, 0.8],
                         [0, 0.8, 0.8]])
    )

@pytest.fixture
def camera_intrinsic():
    return PinholeCameraIntrinsic(631.55,
                                  631.21,
                                  638.43,
                                  366.50)
