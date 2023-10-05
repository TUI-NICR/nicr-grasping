import pytest

import numpy as np

from nicr_grasping.datatypes.grasp import RectangleGrasp, RectangleGraspList

def test_rectanglegrasp_list_iou(rectangle_grasp: RectangleGrasp, rectangle_grasp_list: RectangleGraspList):
    iou = rectangle_grasp_list.iou(rectangle_grasp)

    np.testing.assert_array_equal(iou, np.array([1, 0]))

@pytest.mark.benchmark(group="iou")
def test_rectanglegrasp_list_iou_benchmark(benchmark, rectangle_grasp: RectangleGrasp, rectangle_grasp_list: RectangleGraspList):

    benchmark(rectangle_grasp_list.iou, rectangle_grasp)
