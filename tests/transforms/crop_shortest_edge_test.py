import numpy as np
import cv2

from nicr_grasping.datatypes.grasp import RectangleGrasp, RectangleGraspList

from nicr_grasping.transforms.crop_shortest_edge import CropShortestEdge

def test_crop_shortest_edge_image():
    cse = CropShortestEdge()
    img = np.zeros((100, 50, 1))

    cse.initialize(img)

    r = cse(img)

    assert r.shape == (50, 50, 1)

    ir = cse(r, invert=True)

    assert ir.shape == (100, 50, 1)


def test_crop_shortest_edge_grasp():
    cse = CropShortestEdge()
    img = np.ones((50, 100, 1)) * 125

    cse.initialize(img)

    grasp = RectangleGrasp(0.1, np.array([[50, 20]]), 10, 0, 5)
    orig_grasp_img = grasp.plot(img.copy())
    cv2.imwrite("orig_grasp_img.png", orig_grasp_img)

    tg = cse(grasp)
    ti = cse(img)

    transformed_grasp_img = tg.plot(ti)
    cv2.imwrite("transformed_grasp_img.png", transformed_grasp_img)

    assert np.allclose(tg.points[:, 1], grasp.points[:, 1])

    tii = cse(ti, invert=True)
    tgi = cse(tg, invert=True)

    assert np.allclose(tgi.points, grasp.points)

    recovered_grasp_img = tgi.plot(tii)
    cv2.imwrite("recovered_grasp_img.png", recovered_grasp_img)
