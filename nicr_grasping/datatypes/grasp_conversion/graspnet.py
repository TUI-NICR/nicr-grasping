import numpy as np
import math

from . import CONVERTER_REGISTRY
from ..grasp import RectangleGrasp, RectangleGraspList
from . import logger as baselogger

from graspnetAPI.grasp import Grasp, RectGrasp, RectGraspGroup

def graspnetrect_to_grasp2d(graspnet_grasp : RectGrasp) -> RectangleGrasp:

    # invert x and y position
    # needed because different coordinate systems (matrix vs image)
    center = graspnet_grasp.center_point
    open_point = graspnet_grasp.open_point

    center = np.asarray(center)
    open_point = np.asarray(open_point)

    diff = center - open_point
    width = np.linalg.norm(diff) * 2

    # order of elements because we live in image coordinate system
    angle = np.arctan2(-diff[1], diff[0])

    res = RectangleGrasp(quality=graspnet_grasp.score,
                         center=center.reshape(-1, 2),
                         length=graspnet_grasp.height,
                         width=width,
                         angle=angle)

    return res

def grasp2d_to_graspnetrect(grasp2d : RectangleGrasp) -> RectGrasp:
    # parameters needed: [center_x, center_y, open_x, open_y, height, score, object_id]
    center_x, center_y = grasp2d.center[0]
    open_x, open_y = np.mean(grasp2d.points[:2], axis=0)
    height = grasp2d.length

    params = np.array([center_x, center_y, open_x, open_y, height, grasp2d.quality, -1])

    return RectGrasp(params)

def graspnetrectlist_to_grasp2dlist(graspnetrectlist : RectGraspGroup) -> RectangleGraspList:
    res = [CONVERTER_REGISTRY.convert(grasp, RectangleGrasp) for grasp in graspnetrectlist]
    return RectangleGraspList(res)

def grasp2dlist_to_graspnetrectlist(grasp2dlist : RectangleGraspList) -> RectGraspGroup:
    grasp_converted = [CONVERTER_REGISTRY.convert(grasp, RectGrasp) for grasp in grasp2dlist]
    res = RectGraspGroup()

    for grasp in grasp_converted:
        res.add(grasp)

    return res

CONVERTER_REGISTRY.register(RectGrasp, RectangleGrasp, graspnetrect_to_grasp2d)
CONVERTER_REGISTRY.register(RectangleGrasp, RectGrasp, grasp2d_to_graspnetrect)

CONVERTER_REGISTRY.register(RectGraspGroup, RectangleGraspList, graspnetrectlist_to_grasp2dlist)
CONVERTER_REGISTRY.register(RectangleGraspList, RectGraspGroup, grasp2dlist_to_graspnetrectlist)
