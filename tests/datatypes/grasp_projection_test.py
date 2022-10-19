import numpy as np

def test_3d_to_2d_projection_grasp(grasp_3d, camera_intrinsic):
    grasp_2d = grasp_3d.to_2d(camera_intrinsic)
    depth_image = np.ones((2000, 1000)) * grasp_3d.position[0, 2]

    projected_grasp = grasp_2d.to_3d(depth_image, camera_intrinsic)

    # ignore rotation when comparing because this information is lost
    np.testing.assert_allclose(grasp_3d.position, projected_grasp.position)
    assert grasp_3d.quality == projected_grasp.quality


def test_3d_to_2d_projection_rect_grasp(parallel_gripper_grasp, camera_intrinsic):
    grasp_2d = parallel_gripper_grasp.to_2d(camera_intrinsic)
    depth_image = np.ones((2000, 1000)) * parallel_gripper_grasp.position[0, 2]

    projected_grasp = grasp_2d.to_3d(depth_image, camera_intrinsic)

    # ignore rotation when comparing because this information is lost
    np.testing.assert_allclose(parallel_gripper_grasp.position, projected_grasp.position)
    np.testing.assert_allclose(parallel_gripper_grasp.points, projected_grasp.points)
    assert parallel_gripper_grasp.quality == projected_grasp.quality
