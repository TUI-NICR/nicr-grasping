import pytest

pytest.importorskip('graspnetAPI')

import numpy as np

from nicr_grasping.evaluation import EvalParameters
from nicr_grasping.evaluation.evaluation import eval_grasps_on_model
from nicr_grasping.collision import PointCloudChecker
from nicr_grasping.datatypes.grasp import ParallelGripperGrasp3DList
from nicr_grasping.datatypes.objects import ObjectModel, Scene
from nicr_grasping.datatypes.grasp_conversion import CONVERTER_REGISTRY
from nicr_grasping import graspnet_dataset_path

from nicr_grasping.external.meshpy import ObjFile, SdfFile

from graspnetAPI import GraspNet, GraspGroup
from graspnetAPI.utils.rotation import batch_viewpoint_params_to_matrix
from graspnetAPI.utils.utils import generate_views

OBJECT_ID = 8
GRASPNET_ROOT = graspnet_dataset_path()

MODEL_PATH = f'{GRASPNET_ROOT}/models/{OBJECT_ID:03d}/textured.obj'
SDF_PATH = f'{GRASPNET_ROOT}/models/{OBJECT_ID:03d}/textured.sdf'


def get_graspnet_grasp_labels():
    graspnet = GraspNet(GRASPNET_ROOT, split='test_seen')

    num_views, num_angles, num_depths = 300, 12, 4
    template_views = generate_views(num_views)
    template_views = template_views[np.newaxis, :, np.newaxis, np.newaxis, :]
    template_views = np.tile(template_views, [1, 1, num_angles, num_depths, 1])

    grasp_labels = graspnet.loadGraspLabels(OBJECT_ID)

    grasp_group = GraspGroup()

    sampled_points, offsets, fric_coefs = grasp_labels[OBJECT_ID]
    # collision = collision_dump[i]
    point_inds = np.arange(sampled_points.shape[0])

    num_points = len(point_inds)
    target_points = sampled_points[:, np.newaxis, np.newaxis, np.newaxis, :]
    target_points = np.tile(target_points, [1, num_views, num_angles, num_depths, 1])
    views = np.tile(template_views, [num_points, 1, 1, 1, 1])
    angles = offsets[:, :, :, :, 0]
    depths = offsets[:, :, :, :, 1]
    widths = offsets[:, :, :, :, 2]

    mask1 = (fric_coefs > -1) & (fric_coefs < 1.2)
    mask2 = depths.round(2) == 0.02
    mask = mask1 & mask2
    target_points = target_points[mask]
    # target_points = transform_points(target_points, trans)
    # target_points = transform_points(target_points, np.linalg.inv(camera_pose))
    views = views[mask]
    angles = angles[mask]
    depths = depths[mask]
    widths = widths[mask]
    fric_coefs = fric_coefs[mask]

    Rs = batch_viewpoint_params_to_matrix(-views, angles)

    num_grasp = widths.shape[0]
    scores = (1.1 - fric_coefs).reshape(-1,1)
    widths = widths.reshape(-1,1)
    heights = 0.02 * np.ones((num_grasp,1))
    depths = depths.reshape(-1,1)
    rotations = Rs.reshape((-1,9))
    object_ids = OBJECT_ID * np.ones((num_grasp,1), dtype=np.int32)

    obj_grasp_array = np.hstack([scores, widths, heights, depths, rotations, target_points, object_ids]).astype(np.float32)

    grasp_group.grasp_group_array = np.concatenate((grasp_group.grasp_group_array, obj_grasp_array))

    return grasp_group


@pytest.mark.benchmark(group="eval")
def test_model_eval():

    params = EvalParameters(top_k=50, friction_coefficients=np.linspace(0.1, 1.1, 11)[::-1])

    grasps = get_graspnet_grasp_labels()[:500]
    # grasps.sort_by_score()
    grasp_list = CONVERTER_REGISTRY.convert(grasps, ParallelGripperGrasp3DList)

    mesh = ObjFile(MODEL_PATH)
    sdf = SdfFile(SDF_PATH)

    model = ObjectModel(
        mesh.read(),
        sdf.read()
    )

    res = eval_grasps_on_model(
        grasp_list,
        model,
        params
    )

    scores = 1.1 - res.get_info('min_friction')

    np.testing.assert_allclose(scores.round(4), grasps.scores.round(4))


def test_scene_eval():
    params = EvalParameters(top_k=50, friction_coefficients=np.linspace(0.1, 1.1, 11))

    grasps = get_graspnet_grasp_labels()[:500]
    # grasps.sort_by_score()
    grasp_list = CONVERTER_REGISTRY.convert(grasps, ParallelGripperGrasp3DList)

    mesh = ObjFile(MODEL_PATH)
    sdf = SdfFile(SDF_PATH)

    model = ObjectModel(
        mesh.read(),
        sdf.read()
    )

    scene = Scene(PointCloudChecker())
    scene.add_object(model)

    res = eval_grasps_on_model(
        grasp_list,
        model,
        params
    )

    scores = 1.1 - res.get_info('min_friction')

    np.testing.assert_allclose(scores.round(4), grasps.scores.round(4))


if __name__ == '__main__':
    test_model_eval()
    test_scene_eval()
