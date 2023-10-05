import pytest
import numpy as np

from nicr_grasping.datatypes.transform import Pose


def test_pose_transform():
    trans = np.eye(4)

    pose = Pose()

    pose = pose.transform(trans)

    np.testing.assert_allclose(trans, pose.transformation_matrix)

    trans[:3, 3] = np.array([1, 2, 3])
    pose = pose.transform(trans)

    np.testing.assert_allclose(trans, pose.transformation_matrix)
