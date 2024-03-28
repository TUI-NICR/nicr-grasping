import pytest
import numpy as np

import json

from nicr_grasping.datatypes.grasp import RectangleGraspList, RectangleGrasp, ParallelGripperGrasp3DList


def test_rectangle_grasp_save_load(rectangle_grasp, tmp_path):
    grasp_file = tmp_path / 'graspfile.npy'
    rectangle_grasp.save(grasp_file)

    loaded_grasp = RectangleGrasp.load(grasp_file)

    np.testing.assert_array_equal(rectangle_grasp.points, loaded_grasp.points)


def test_rectangle_grasp_list_save_load(rectangle_grasp_list, tmp_path):
    grasp_file = tmp_path / 'graspfile.pkl'
    rectangle_grasp_list.save(grasp_file)

    loaded_grasp_list = RectangleGraspList.load(grasp_file)

    assert len(rectangle_grasp_list) == len(loaded_grasp_list)

    for g1, g2 in zip(rectangle_grasp_list, loaded_grasp_list):
        np.testing.assert_array_equal(g1.points, g2.points)

    rectangle_grasp_list.save(grasp_file, compressed=True)

    loaded_grasp_list = RectangleGraspList.load(grasp_file)

    assert len(rectangle_grasp_list) == len(loaded_grasp_list)

    for g1, g2 in zip(rectangle_grasp_list, loaded_grasp_list):
        np.testing.assert_array_equal(g1.points, g2.points)


def test_rectanlge_grasp_list_from_mira_xml(grasp_mira_json_file):

    with open(str(grasp_mira_json_file), 'r') as f:
        json_object = json.load(f)

    for grasps in json_object:
        grasp_list = ParallelGripperGrasp3DList.from_mira_json(grasps['Second'])

        assert grasp_list[0].quality == 0.750
        assert grasp_list[1].quality == 0.737
