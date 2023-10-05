import pytest
import numpy as np


def test_grasp_rotation(rectangle_grasp):
    orig_grasp = rectangle_grasp.copy()
    rectangle_grasp.rotate(np.pi)
    rectangle_grasp.rotate(-np.pi)

    assert rectangle_grasp == orig_grasp

    rectangle_grasp.rotate(0.3 * np.pi)
    rectangle_grasp.rotate(-0.3 * np.pi)

    assert rectangle_grasp == orig_grasp


def test_grasp_scaling(rectangle_grasp):
    orig_grasp = rectangle_grasp.copy()

    rectangle_grasp.scale(2)
    rectangle_grasp.scale(1/2)

    assert rectangle_grasp == orig_grasp
