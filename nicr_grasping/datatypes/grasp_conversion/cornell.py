import numpy as np
import math

from . import CONVERTER_REGISTRY
from ..grasp import RectangleGrasp, Grasp2DList

from grasp_detection.utils.dataset_processing.grasp import GraspRectangle, GraspRectangles

from . import GRASPNET_INSTALLED

# Conversion GraspNet -> GGCNN
if GRASPNET_INSTALLED:
    from graspnetAPI.grasp import RectGrasp
    def rectgrasp_to_grasprect(grasp : RectGrasp) -> GraspRectangle:
        grasp = CONVERTER_REGISTRY.convert(grasp, RectangleGrasp)
        res = GraspRectangle(grasp.points)

        return res

    CONVERTER_REGISTRY.register(RectGrasp, GraspRectangle, rectgrasp_to_grasprect)
