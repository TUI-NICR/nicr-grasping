import pytest
import numpy as np

from nicr_grasping.datatypes.transform import Pose, difference, DifferenceType, SymmetryType

def test_difference():
    gt_pose = Pose()

    pred_pose = Pose()
    transformation = np.eye(4)
    transformation[:3, :3] = np.diag([-1, -1, 1])

    pred_pose = pred_pose.transform(transformation)

    diff_euclidean = difference(gt_pose, pred_pose, difference_type=DifferenceType.EUCLIDEAN)
    assert diff_euclidean['error'] == 0.0

    diff_rot = difference(gt_pose, pred_pose, difference_type=DifferenceType.ROTATION)
    np.testing.assert_allclose(diff_rot['error'], 180.0)

    transformation[:3, :3] = np.diag([-1, 1, -1])
    pred_pose = Pose().transform(transformation)

    diff_rot = difference(gt_pose, pred_pose, difference_type=DifferenceType.ROTATION)
    np.testing.assert_allclose(diff_rot['error'], 180.0)

    diff_rot = difference(gt_pose, pred_pose, difference_type=DifferenceType.ROTATION,
                          symmetry=SymmetryType.Y)
    np.testing.assert_allclose(diff_rot['error'], 0.0)
    diff_rot = difference(gt_pose, pred_pose, difference_type=DifferenceType.ROTATION,
                          symmetry=SymmetryType.Y_FLIP)
    np.testing.assert_allclose(diff_rot['error'], 0.0)
