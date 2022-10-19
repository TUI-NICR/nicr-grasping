import pytest

import numpy as np

from nicr_grasping.datatypes.grasp import RectangleGrasp, RectangleGraspList

@pytest.mark.benchmark(group="iou")
def test_rectanglegrasp_list_iou_benchmark(benchmark, rectangle_grasp: RectangleGrasp, rectangle_grasp_list: RectangleGraspList):

    benchmark(rectangle_grasp_list.iou, rectangle_grasp)
