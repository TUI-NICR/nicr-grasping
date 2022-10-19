import pytest
import numpy as np

from nicr_grasping.datatypes.grasp import RectangleGraspDrawingMode

@pytest.mark.parametrize("mode", [RectangleGraspDrawingMode.INNER_RECTANGLE, RectangleGraspDrawingMode.GAUSS, RectangleGraspDrawingMode.CENTER_POINT])
@pytest.mark.benchmark(group='labeldrawing')
def test_label_drawing_benchmark(benchmark, rectangle_grasp, mode):
    img = np.zeros((100, 100))
    labels = [img, img.copy(), img.copy()]
    labels = benchmark(rectangle_grasp.draw_label, labels, mode=mode)
