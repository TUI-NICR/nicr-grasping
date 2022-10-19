import pytest

pytest.importorskip('graspnetAPI')

import numpy as np
import cv2

from graspnetAPI.grasp import RectGrasp

from nicr_grasping.datatypes.grasp_conversion import CONVERTER_REGISTRY
from nicr_grasping.datatypes.grasp import RectangleGrasp

@pytest.fixture
def graspnet_rectgrasp():
    # [center_x, center_y, open_x, open_y, height, score, object_id]
    grasp_params = [30, 60, 50, 55, 10, 0.5, 0]
    grasp_params = np.asarray(grasp_params)
    graspnet_grasp = RectGrasp(grasp_params)
    return graspnet_grasp

def test_graspnetrect_to_rect(graspnet_rectgrasp):
    grasp2d = CONVERTER_REGISTRY.convert(graspnet_rectgrasp, RectangleGrasp)

    # compute rect points
    # taken from to_opencv_image function
    center_x, center_y, open_x, open_y, height, _, _ = graspnet_rectgrasp.rect_grasp_array
    center = np.array([center_x, center_y])
    left = np.array([open_x, open_y])
    axis = left - center
    normal = np.array([-axis[1], axis[0]])
    normal = normal / np.linalg.norm(normal) * height / 2
    p1 = center + normal + axis
    p2 = center + normal - axis
    p3 = center - normal - axis
    p4 = center - normal + axis

    assert grasp2d.quality == graspnet_rectgrasp.score
    assert np.isclose(grasp2d.length, graspnet_rectgrasp.height)
    np.testing.assert_allclose(grasp2d.points, np.array([p2, p3, p4, p1]), atol=0.0001)

def test_plot(graspnet_rectgrasp):
    img = np.zeros((100, 100, 3))
    graspnet_img = graspnet_rectgrasp.to_opencv_image(img.copy()).astype(np.uint8)

    grasp = CONVERTER_REGISTRY.convert(graspnet_rectgrasp, RectangleGrasp)

    grasp2d_img = grasp.plot(img.copy()).astype(np.uint8)

    # graspnet_grasp2 = CONVERTER_REGISTRY.convert(grasp, RectGrasp)
    graspnet_grasp2 = CONVERTER_REGISTRY.convert(grasp, RectGrasp)
    graspnet2_img = graspnet_grasp2.to_opencv_image(img.copy()).astype(np.uint8)

    graspnet_img = cv2.cvtColor(graspnet_img, cv2.COLOR_BGR2RGB)
    graspnet2_img = cv2.cvtColor(graspnet2_img, cv2.COLOR_BGR2RGB)
    grasp2d_img = cv2.cvtColor(grasp2d_img, cv2.COLOR_BGR2RGB)

    cv2.imwrite('grasp2d.png', grasp2d_img)
    cv2.imwrite('graspnet1.png', graspnet_img)
    cv2.imwrite('graspnet2.png', graspnet2_img)

    # compare images by thresholding number of differing elements (NOT PIXELS)
    assert (grasp2d_img != graspnet_img).sum() <= 30
    assert (grasp2d_img != graspnet2_img).sum() <= 30
