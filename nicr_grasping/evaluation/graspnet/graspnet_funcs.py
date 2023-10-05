import os
import tqdm
import cv2
import matplotlib.pyplot as plt

from typing import Optional
import numpy as np

from ...datatypes.grasp_conversion import CONVERTER_REGISTRY
from ...utils.postprocessing import convert_model_output_to_grasps

from graspnetAPI.grasp import RectGraspGroup
from graspnetAPI.graspnet_eval import GraspNetEval, GraspNet
from graspnetAPI.utils.utils import batch_center_area_depth

def eval_on_graspnet(graspnet_root: str,
                     prediction_folder: str,
                     camera: str='kinect',
                     split: str='test',
                     num_jobs: int=24):

    ge_k = GraspNetEval(root = graspnet_root, camera = camera, split = split)

    eval_res = {}

    if split == 'test_seen':
        res, ap = ge_k.eval_seen(prediction_folder, proc = num_jobs)
        eval_res['AP-seen'] = ap
    elif split == 'test_similar':
        res, ap = ge_k.eval_similar(prediction_folder, proc = num_jobs)
        eval_res['AP-similar'] = ap
    elif split == 'test_novel':
        res, ap = ge_k.eval_novel(prediction_folder, proc = num_jobs)
        eval_res['AP-novel'] = ap
    else:
        res, ap = ge_k.eval_all(prediction_folder, proc = num_jobs)
        eval_res['AP-seen'] = ap[1]
        eval_res['AP-similar'] = ap[2]
        eval_res['AP-novel'] = ap[3]

    return eval_res

def gt_predictor(graspnet_eval: GraspNetEval,
                 scene_id: int, anno_id: int,
                 camera:str = 'kinect',
                 grasp_labels = None,
                 collision_labels = None):
    """Fuction acting as predictor using the ground truth labels.
    Usefull for sanity checking the labels.
    """
    depth = graspnet_eval.loadDepth(scene_id, camera, anno_id)

    grasps_6d = graspnet_eval.loadGrasp(scene_id, anno_id, format='6d', camera=camera, grasp_labels=grasp_labels, collision_labels=collision_labels,
                                        fric_coef_thresh = 1.0)

    # do this first because otherwise to_rect_grasp_group will filter through this check
    # and we lose corresponding grasps otherwise
    grasps_6d.remove(np.argwhere(grasps_6d.rotation_matrices[:, 2, 0] <= 0.99))
    grasp_rect = grasps_6d.to_rect_grasp_group(camera)

    # remove grasps at the image borders
    # comparison with shape - 1 because 719.9 gets rounded to 720
    idxs = np.argwhere(grasp_rect.center_points[:, 0] >= 1279)
    grasps_6d.remove(idxs)
    grasp_rect.remove(idxs)

    idxs = np.argwhere(grasp_rect.center_points[:, 1] >= 719)
    grasps_6d.remove(idxs)
    grasp_rect.remove(idxs)

    grasp_backprojection = grasp_rect.to_grasp_group(camera, depth, depth_method=batch_center_area_depth)

    translation_difference = np.linalg.norm(grasps_6d.translations - grasp_backprojection.translations, axis=1)
    grasp_backprojection.remove(np.argwhere(translation_difference < 0.001))

    return grasp_backprojection

    ids = graspnet_eval.getDataIds([scene_id])
    grasps = RectGraspGroup().from_npy(graspnet_eval.rectLabelPath[ids[anno_id]])


    return grasps.to_grasp_group(camera, depth, depth_method=batch_center_area_depth)

def gt_predictor_base(graspnet_eval: GraspNetEval,
                 scene_id: int, anno_id: int,
                 camera:str = 'kinect',
                 gt_folder: Optional[str]=None):
    """Fuction acting as predictor using the ground truth labels.
    Usefull for sanity checking the labels.

    Parameters
    ----------
    graspnet_eval : GraspNetEval
        GraspNetEval object used for loading grasps and depth images.
    scene_id : int
        Id of scene to predict.
    anno_id : int
        Id of sample of scene to predict.
    camera : str, optional
        Name of the camera ('kinect' or 'realsense'), by default 'kinect'
    gt_folder : Optional[str], optional
        If this folder is specified it will be used for loading grasps.
        Otherwise the original rect grasp labels of GraspNet will be used , by default None

    Returns
    -------
    GraspGroup
        3D grasps computed from labels.
    """
    if gt_folder is None:
        grasps = graspnet_eval.loadGrasp(scene_id, anno_id, format='rect', camera=camera)
    else:
        grasps = RectGraspGroup().from_npy(os.path.join(gt_folder, f'scene_{scene_id:04}', camera, 'rect', f'{anno_id:04d}.npy'))

    # remove grasps at the image borders
    # comparison with shape - 1 because 719.9 gets rounded to 720
    idxs = np.argwhere(grasps.center_points[:, 0] >= 1279)
    grasps.remove(idxs)

    idxs = np.argwhere(grasps.center_points[:, 0] < 0)
    grasps.remove(idxs)

    idxs = np.argwhere(grasps.center_points[:, 1] >= 719)
    grasps.remove(idxs)

    idxs = np.argwhere(grasps.center_points[:, 1] < 0)
    grasps.remove(idxs)

    depth = graspnet_eval.loadDepth(scene_id, camera, anno_id)
    return grasps.to_grasp_group(camera, depth)


def run_predictor(predictor: "torch.engine.DefaultPredictor",
                  graspnet: GraspNet,
                  savedir: str,
                  scene_ids = None,
                  debug: bool=False):

    import torch

    if scene_ids is None:
        scene_ids = graspnet.sceneIds

    with torch.no_grad():
        for scene_idx, scene_id in enumerate(tqdm.tqdm(scene_ids, total=len(scene_ids), desc='Iterating scenes')):
            scene_eval = []
            for ann_id in tqdm.tqdm(range(256), total=256, desc='Iterating samples', leave=False):
                depth_img = graspnet.loadDepth(sceneId = scene_id, camera = graspnet.camera, annId = ann_id)
                rgb_img = graspnet.loadRGB(sceneId = scene_id, camera = graspnet.camera, annId = ann_id)

                masks = graspnet.loadMask(scene_id, graspnet.camera, ann_id)

                workspace = graspnet.loadWorkSpace(sceneId=scene_id, camera=graspnet.camera, annId=ann_id)
                workspace_mask = np.zeros_like(depth_img)
                workspace_mask[workspace[0]:workspace[2], workspace[1]:workspace[3]] = 1

                workspace_mask = masks != 0
                workspace_mask = workspace_mask.astype(np.uint8)

                inp_depth = torch.from_numpy(np.expand_dims(depth_img.copy(), 0).astype(np.float32)).cuda()
                inp_rgb =  torch.from_numpy(rgb_img.copy().astype(np.float32)).cuda()
                inp_rgb = inp_rgb.permute(2,0,1)

                # inp_depth *= 0.001
                inp_depth -= inp_depth.min()
                inp_depth /= (inp_depth.max())

                pred = predictor.predict([{'image': {'depth': inp_depth, 'color': inp_rgb}}])
                maps = [pred[0]['pos'], pred[0]['ang'], pred[0]['width']]

                quality, angle, width = maps
                quality *= workspace_mask

                maps = (quality, angle, width)

                grasps = convert_model_output_to_grasps(maps, min_quality=0.01, num_grasps=200, min_distance=1)

                graspnet_grasps = CONVERTER_REGISTRY.convert(grasps, RectGraspGroup)

                rgb_img = grasps.plot(rgb_img)
                rgb_img = np.flip(rgb_img, (0, 1)).astype(np.uint8)

                qual = maps[0]
                qual = np.stack([qual, qual, qual], axis=-1)
                qual = grasps.plot(qual)

                os.makedirs(os.path.join(savedir, f'scene_{scene_id:04d}', graspnet.camera), exist_ok=True)
                current_save_dir = os.path.join(savedir, f'scene_{scene_id:04d}', graspnet.camera, f'{ann_id:04d}')

                graspnet_grasps = graspnet_grasps.to_grasp_group(graspnet.camera, depth_img)
                graspnet_grasps.save_npy(current_save_dir)

                if debug:
                    break
            if debug:
                break
