import pytest
import numpy as np
from nicr_grasping.datatypes.grasp import RectangleGrasp, RectangleGraspList

def test_rectangle_grasp_setters(rectangle_grasp):
    orig_grasp = rectangle_grasp.copy()
    orig_width = rectangle_grasp.width
    orig_length = rectangle_grasp.length

    rectangle_grasp.width = orig_width + 10

    assert np.isclose(rectangle_grasp.width, orig_width + 10)

    rectangle_grasp.length = orig_length + 10
    assert np.isclose(rectangle_grasp.length, orig_length + 10)
    np.testing.assert_almost_equal(rectangle_grasp.center, orig_grasp.center)


def test_rectangle_grasp_from_points(rectangle_grasp):
    g2 = RectangleGrasp.from_points(rectangle_grasp.points)

    assert pytest.approx(rectangle_grasp.angle) == g2.angle
    assert pytest.approx(rectangle_grasp.length) == g2.length
    assert pytest.approx(rectangle_grasp.center[0, 0]) == g2.center[0, 0]
    assert pytest.approx(rectangle_grasp.center[0, 1]) == g2.center[0, 1]
    assert pytest.approx(rectangle_grasp.width) == g2.width


def test_rect_label_generation(rectangle_grasp: RectangleGrasp):
    shape = (300, 200)
    single_label = [np.zeros(shape), np.zeros(shape), np.zeros(shape)]
    single_label = rectangle_grasp.draw_label(single_label)

    rect_grasp_list = RectangleGraspList([rectangle_grasp])
    list_label = rect_grasp_list.create_sample_images(shape)

    np.testing.assert_array_equal(single_label, list_label)


def test_compare(rectangle_grasp, other_rectangle_grasp, rectangle_grasp_list, other_rectangle_grasp_list):
    assert rectangle_grasp == rectangle_grasp
    assert not rectangle_grasp == other_rectangle_grasp

    assert rectangle_grasp_list == rectangle_grasp_list
    assert not rectangle_grasp_list ==other_rectangle_grasp_list
