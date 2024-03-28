import pytest
import numpy as np
import torch
import matplotlib.pyplot as plt

from nicr_grasping.datatypes.grasp import RectangleGrasp, RectangleGraspList, RectangleGraspDrawingMode
from nicr_grasping.utils.postprocessing import convert_model_output_to_grasps, convert_ggcnn_output_to_grasps


@pytest.fixture
def rectangle_grasp_list():
    grasps = RectangleGraspList([
        RectangleGrasp(1, np.array([[50, 20]]), 30, 0),
        RectangleGrasp(1, np.array([[30, 20]]), 30, np.pi),
        RectangleGrasp(1, np.array([[60, 60]]), 30, 0)
    ])

    return grasps

# pytest.fixture cannnot be called in main
def debug_in_main_rectangle_grasp_list():
    grasps = RectangleGraspList([
        RectangleGrasp(1, np.array([[50, 20]]), 30, 0),
        RectangleGrasp(1, np.array([[30, 20]]), 30, np.pi),
        RectangleGrasp(1, np.array([[60, 60]]), 30, 0)
    ])

    return grasps

@pytest.mark.parametrize("mode", [RectangleGraspDrawingMode.GAUSS, RectangleGraspDrawingMode.CENTER_POINT, RectangleGraspDrawingMode.GAUSS_WITH_MARGIN])
def test_model_output_to_grasplist(rectangle_grasp_list: RectangleGraspList, mode):
    label_img = rectangle_grasp_list.create_sample_images((100, 100), mode=mode)
    label_img = [label_img[0], np.cos(label_img[1]), np.sin(label_img[1]), label_img[2]]
    label_img = [torch.from_numpy(l) for l in label_img]

    grasps = convert_model_output_to_grasps(label_img, num_grasps=len(rectangle_grasp_list), min_distance=5)

    grasps.sort_by_quality()
    rectangle_grasp_list.sort_by_quality()

    assert len(grasps) == len(rectangle_grasp_list), print(len(grasps), len(rectangle_grasp_list))
    for grasp in grasps:
        if mode==RectangleGraspDrawingMode.GAUSS:
            assert grasp in rectangle_grasp_list, print((
                    "Maybe there is a round-off error"
                    " in max quality coordinates while using RectangleGraspDrawingMode.GAUSS?"
                ))
        else:
            assert grasp in rectangle_grasp_list

if __name__=="__main__":
    test_model_output_to_grasplist(debug_in_main_rectangle_grasp_list())
