import numpy as np

from nicr_grasping.datatypes.grasp import RectangleGraspList

def test_rectangle_list_creation(rectangle_grasp_list):
    centers = np.asarray([g.center for g in rectangle_grasp_list])
    widths = np.asarray([g.width for g in rectangle_grasp_list])
    lengths = np.asarray([g.length for g in rectangle_grasp_list])
    angles = np.asarray([g.angle for g in rectangle_grasp_list])
    qualities = np.asarray([g.quality for g in rectangle_grasp_list])

    base_grasps = np.zeros((len(rectangle_grasp_list), 2, 4))

    base_grasps[:, 1, 1:3] = lengths / 2
    base_grasps[:, 1, 0] = -lengths / 2
    base_grasps[:, 1, 3] = -lengths / 2

    base_grasps[:, 0, 0:2] = widths / 2
    base_grasps[:, 0, 2:4] = -widths / 2

    rotmats = np.array(
        [
            [np.cos(angles), -np.sin(angles)],
            [np.sin(angles), np.cos(angles)],
        ]
    )

    # rotate by angle
    base_grasps = np.matmul(rotmats.T, base_grasps)

    base_grasps += np.swapaxes(centers, 1, 2)

    for i in range(len(rectangle_grasp_list)):
        np.testing.assert_array_equal(base_grasps[i].T, rectangle_grasp_list[i].points)

    grasp_list = RectangleGraspList.from_points(base_grasps, qualities)

    for g1, g2 in zip(grasp_list, rectangle_grasp_list):
        assert g1 == g2
