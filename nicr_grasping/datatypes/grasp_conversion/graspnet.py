import math

import numpy as np
from graspnetAPI.grasp import Grasp, RectGrasp, RectGraspGroup, GraspGroup
from scipy.spatial.transform import Rotation as R

from ..grasp import ParallelGripperGrasp3D, RectangleGrasp, RectangleGraspList, ParallelGripperGrasp3DList
from . import CONVERTER_REGISTRY
from . import logger as baselogger


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

def grasp3d_to_graspnetgrasp(grasp3d : ParallelGripperGrasp3D) -> Grasp:
    # params needed: [score, width, height, depth, rotation_matrix(9), translation(3), object_id]

    # definition of grasp is different in MIRA so we need to rotate the grasp
    rotation_fix = R.from_euler('xy', [90, -90], degrees=True).as_matrix()

    grasp_params = np.zeros(17)
    grasp_params[0] = grasp3d.quality
    grasp_params[1] = grasp3d.width
    grasp_params[2] = grasp3d.height

    # depth of the grasp meaning the length of the yaws
    # this is fixed to 0.02m in the graspnet dataset
    grasp_params[3] = 0.02

    grasp_params[4:13] = (grasp3d.orientation @ rotation_fix).reshape(-1)
    grasp_params[13:16] = grasp3d.position.reshape(-1)
    grasp_params[16] = -1

    return Grasp(grasp_params)

def parallelgrasp3dlist_to_graspgroup(grasplist: ParallelGripperGrasp3DList) -> GraspGroup:
    res = GraspGroup()
    for grasp in grasplist:
        res.add(CONVERTER_REGISTRY.convert(grasp, Grasp))

    return res

def graspnetgrasp_to_grasp3d(graspnet_grasp : Grasp) -> ParallelGripperGrasp3D:
    grasp = ParallelGripperGrasp3D()

    rotation_fix = R.from_euler('xy', [90, -90], degrees=True).as_matrix()
    rotation_fix = np.linalg.inv(rotation_fix)

    grasp.quality = graspnet_grasp.score
    grasp.width = graspnet_grasp.width
    grasp.height = graspnet_grasp.height

    grasp.orientation = graspnet_grasp.rotation_matrix @ rotation_fix
    grasp.position = graspnet_grasp.translation

    return grasp

def graspgroup_to_parallelgrasp3dlist(graspgroup : GraspGroup) -> ParallelGripperGrasp3DList:
    grasps = []
    for grasp in graspgroup:
        grasps.append(CONVERTER_REGISTRY.convert(grasp, ParallelGripperGrasp3D))

    return ParallelGripperGrasp3DList(grasps)

CONVERTER_REGISTRY.register(RectGrasp, RectangleGrasp, graspnetrect_to_grasp2d)
CONVERTER_REGISTRY.register(RectangleGrasp, RectGrasp, grasp2d_to_graspnetrect)

CONVERTER_REGISTRY.register(RectGraspGroup, RectangleGraspList, graspnetrectlist_to_grasp2dlist)
CONVERTER_REGISTRY.register(RectangleGraspList, RectGraspGroup, grasp2dlist_to_graspnetrectlist)

CONVERTER_REGISTRY.register(ParallelGripperGrasp3D, Grasp, grasp3d_to_graspnetgrasp)
CONVERTER_REGISTRY.register(ParallelGripperGrasp3DList, GraspGroup, parallelgrasp3dlist_to_graspgroup)

CONVERTER_REGISTRY.register(Grasp, ParallelGripperGrasp3D, graspnetgrasp_to_grasp3d)
CONVERTER_REGISTRY.register(GraspGroup, ParallelGripperGrasp3DList, graspgroup_to_parallelgrasp3dlist)
