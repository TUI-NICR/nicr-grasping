import pytest
import numpy as np

from nicr_grasping.datatypes.grasp import RectangleGraspList, RectangleGrasp


def test_rectangle_grasp_save_load(rectangle_grasp, tmp_path):
    grasp_file = tmp_path / 'graspfile.npy'
    rectangle_grasp.save(grasp_file)

    loaded_grasp = RectangleGrasp.load_from_file(grasp_file)

    np.testing.assert_array_equal(rectangle_grasp.points, loaded_grasp.points)


def test_rectangle_grasp_list_save_load(rectangle_grasp_list, tmp_path):
    grasp_file = tmp_path / 'graspfile.pkl'
    rectangle_grasp_list.save(grasp_file)

    loaded_grasp_list = RectangleGraspList.load_from_file(grasp_file)

    assert len(rectangle_grasp_list.grasps) == len(loaded_grasp_list.grasps)

    for g1, g2 in zip(rectangle_grasp_list.grasps, loaded_grasp_list.grasps):
        np.testing.assert_array_equal(g1.points, g2.points)

    rectangle_grasp_list.save(grasp_file, compressed=True)

    loaded_grasp_list = RectangleGraspList.load_from_file(grasp_file)

    assert len(rectangle_grasp_list.grasps) == len(loaded_grasp_list.grasps)

    for g1, g2 in zip(rectangle_grasp_list.grasps, loaded_grasp_list.grasps):
        np.testing.assert_array_equal(g1.points, g2.points)
