import pytest
import numpy as np
import cv2
import matplotlib.pyplot as plt
import copy
from nicr_grasping.datatypes.grasp import RectangleGrasp, RectangleGraspList, RectangleGraspDrawingMode
from nicr_grasping.utils.postprocessing import peak_local_max

import torch
import time

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

@pytest.mark.parametrize("mode", [RectangleGraspDrawingMode.GAUSS, RectangleGraspDrawingMode.CENTER_POINT])
def test_model_output_to_grasplist(rectangle_grasp_list: RectangleGraspList, mode):
    if torch.cuda.is_available():
        torch_version = "torch_gpu"
    else:
        torch_version = "torch_cpu"

    label_img = rectangle_grasp_list.create_sample_images((100, 100), mode=mode)
    quality_map = label_img[0]
    quality_map = np.random.rand(200, 200) # the test fails for random noise
    quality_map = torch.from_numpy(quality_map)
    min_distance = 10
    min_quality = 0.1
    num_grasps = 5

    local_max_cpu = peak_local_max(
        quality_map, min_distance=min_distance, threshold_abs=min_quality, num_peaks=num_grasps, loc_max_version="skimage_cpu")

    local_max_gpu = peak_local_max(
        quality_map, min_distance=min_distance, threshold_abs=min_quality, num_peaks=num_grasps, loc_max_version=torch_version)

    # visualization

    # quality_map = torch.from_numpy(quality_map[np.newaxis, np.newaxis, ...])
    # quality_map = torch.squeeze(quality_map.repeat(1,3,1,1)).permute(1,2,0).cpu().numpy()*255
    # quality_map = np.ascontiguousarray(quality_map, dtype=np.uint8)

    # img_cpu = quality_map.copy()
    # for loc_max in local_max_cpu:
    #     RED = (0, 0, 255)
    #     center = int(loc_max[1]), int(loc_max[0])
    #     axes = 3, 3
    #     angle = 0
    #     cv2.ellipse(img_cpu, center, axes, 0, angle, 360, RED, thickness=1)

    # img_gpu = quality_map.copy()
    # for loc_max in local_max_gpu:
    #     RED = (0, 0, 255)
    #     center = int(loc_max[1]), int(loc_max[0])
    #     axes = 3, 3
    #     angle = 0
    #     cv2.ellipse(img_gpu, center, axes, 0, angle, 360, RED, thickness=1)

    #cv2.imwrite("img_cpu.png", img_cpu)
    #cv2.imwrite("img_gpu.png", img_gpu)
    #assert np.all(local_max_cpu == local_max_gpu), print(local_max_cpu, local_max_gpu)
