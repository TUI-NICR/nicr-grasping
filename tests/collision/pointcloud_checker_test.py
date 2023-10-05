import pytest

import numpy as np
import cv2

from nicr_grasping.collision import PointCloudChecker
from nicr_grasping.datatypes import ParallelGripperGrasp3D, PinholeCameraIntrinsic
from nicr_grasping.datatypes.grasp_conversion import CONVERTER_REGISTRY


def test_pointcloud_checker():
    grasp = ParallelGripperGrasp3D(0.1, 1, np.array([[0, 0, 0.52]]))

    depth_image = np.ones((100, 100, 1)) * 0.5
    intrinsic = {
        'fx': 1,
        'fy': 1,
        'cx': 50,
        'cy': 50
    }

    checker = PointCloudChecker(PinholeCameraIntrinsic(**intrinsic))

    checker.set_depth_image(depth_image)

    collision = checker.check_collision(grasp)

    assert collision == True

    pc = np.array([[0, 0, 0.515]])
    checker.set_point_cloud(pc)

    collision = checker.check_collision(grasp)

    assert collision == False

    pc = np.array([[0.055, 0, 0.515]])

    checker.set_point_cloud(pc)
    collision = checker.check_collision(grasp)

    assert collision == True


def test_scaled_and_cropped_depth_image():
    grasp = ParallelGripperGrasp3D(0.1, 1, np.array([[0, 0, 0.5]]))

    original_depth_image = np.ones((100, 100, 1)) * 0.5
    scaled_depth_image = cv2.resize(original_depth_image, (25, 50))
    scaled_depth_image = scaled_depth_image[:, :, np.newaxis]
    scaled_depth_image_scaling = np.array([[100/25, 100/50]])

    cropped_depth_image = original_depth_image[30:65, 25:75]
    cropped_depth_image_anchor = np.array([[25, 30]])

    cropped_and_scaled_depth_image = original_depth_image[:, 25:75]
    cropped_and_scaled_depth_image = cv2.resize(cropped_and_scaled_depth_image, (25, 25))
    cropped_and_scaled_depth_image = cropped_and_scaled_depth_image[:, :, np.newaxis]
    cropped_and_scaled_depth_image_anchor = np.array([[25, 0]])
    cropped_and_scaled_depth_image_scaling = np.array([[50/25, 50/25]])

    intrinsic = {
        'fx': 1,
        'fy': 1,
        'cx': 50,
        'cy': 50
    }

    checker = PointCloudChecker(PinholeCameraIntrinsic(**intrinsic))

    checker.set_depth_image(original_depth_image)
    original_pc = checker.point_cloud

    checker.set_depth_image(cropped_depth_image, cropped_depth_image_anchor)
    cropped_pc = checker.point_cloud

    for point in cropped_pc:
        assert any(np.equal(original_pc, point).all(1))

    checker.set_depth_image(scaled_depth_image, scaling=scaled_depth_image_scaling)
    scaled_pc = checker.point_cloud

    for point in scaled_pc:
        assert any(np.equal(original_pc, point).all(1))

    checker.set_depth_image(cropped_and_scaled_depth_image, cropped_and_scaled_depth_image_anchor, cropped_and_scaled_depth_image_scaling)
    cropped_and_scaled_pc = checker.point_cloud

    for point in cropped_and_scaled_pc:
        assert any(np.equal(original_pc, point).all(1))
