import pytest
import numpy as np

import json

from nicr_grasping.datatypes.grasp import RectangleGrasp, RectangleGraspList, Grasp3D
from nicr_grasping.datatypes.grasp.grasp_3d import ParallelGripperGrasp3D
from nicr_grasping.datatypes.intrinsics import PinholeCameraIntrinsic

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

@pytest.fixture
def grasp_mira_json_file(tmp_path):
    file_content = """[
    {
        "First" : 4075523307879792642,
        "Second" : [
            {
                "@version[mira::roboticmanipulation::VirtualGraspPoseBase<mira::RigidTransformCov<float,3>>]" : 0,
                "GripperWidth" : 0.164,
                "Pose" : {
                    "Cov" : [
                        [ 0.000, 0.000, 0.000, 0.000, 0.000, 0.000 ],
                        [ 0.000, 0.000, 0.000, 0.000, 0.000, 0.000 ],
                        [ 0.000, 0.000, 0.000, 0.000, 0.000, 0.000 ],
                        [ 0.000, 0.000, 0.000, 0.000, 0.000, 0.000 ],
                        [ 0.000, 0.000, 0.000, 0.000, 0.000, 0.000 ],
                        [ 0.000, 0.000, 0.000, 0.000, 0.000, 0.000 ]
                    ],
                    "Pitch" : -23.147,
                    "Roll" : -39.338,
                    "X" : 0.002,
                    "Y" : -0.005,
                    "Yaw" : -59.346,
                    "Z" : 0.001
                },
                "Quality" : 0.750
            },
            {
                "@version[mira::roboticmanipulation::VirtualGraspPoseBase<mira::RigidTransformCov<float,3>>]" : 0,
                "GripperWidth" : 0.168,
                "Pose" : {
                    "Cov" : [
                        [ 0.000, 0.000, 0.000, 0.000, 0.000, 0.000 ],
                        [ 0.000, 0.000, 0.000, 0.000, 0.000, 0.000 ],
                        [ 0.000, 0.000, 0.000, 0.000, 0.000, 0.000 ],
                        [ 0.000, 0.000, 0.000, 0.000, 0.000, 0.000 ],
                        [ 0.000, 0.000, 0.000, 0.000, 0.000, 0.000 ],
                        [ 0.000, 0.000, 0.000, 0.000, 0.000, 0.000 ]
                    ],
                    "Pitch" : -11.684,
                    "Roll" : -18.422,
                    "X" : -0.008,
                    "Y" : 0.005,
                    "Yaw" : -48.787,
                    "Z" : 0.008
                },
                "Quality" : 0.737
            }
        ]
    }
    ]"""

    with (tmp_path / 'grasp.json').open('w') as f:
        f.write(file_content)

    return tmp_path / 'grasp.json'