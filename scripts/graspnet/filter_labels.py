import tqdm
import os
import pickle
import argparse
import json

import numpy as np

from graspnetAPI.graspnet_eval import GraspNetEval

GRASPNET_ROOT = "/datasets_nas/grasping/graspnet"


def get_args():
    ap = argparse.ArgumentParser()

    ap.add_argument('target_dir',
                    help='Directory where filtered labels are to be saved. Structure will be in GraspNet-format.',
                    type=str)
    ap.add_argument('--src-dir', dest='src_dir', type=str,
                    help='Directory where grasps are to be loaded from.',
                    default='/results_nas/best3125/grasping/eval/GT')
    ap.add_argument('--max-friction-coeff', dest='max_friction_coeff',
                    help='Maximum friction coefficient for filtering',
                    default=1.2,
                    type=float)
    ap.add_argument('--camera', help='Which camera to use. Either "kinect" or "realsense". Default: "kinect"',
                    default='kinect', type=str)

    return ap.parse_args()


def main():

    args = get_args()

    graspnet_eval = GraspNetEval(GRASPNET_ROOT, args.camera, split='all')
    scene_ids = graspnet_eval.sceneIds

    os.makedirs(args.target_dir, exist_ok=True)

    for scene_idx, scene_id in enumerate(tqdm.tqdm(scene_ids, total=len(scene_ids), desc='Iterating scenes')):

        with open(os.path.join(args.src_dir, f'scene_{scene_id:04d}_log.json'), 'r') as f:
            log_data = json.load(f)
            scores_list = log_data['unfiltered']['score_lists']
            collisions_list = log_data['unfiltered']['collision_lists']
            sortidx_list = log_data['unfiltered']['sortidx_lists']

        for ann_id in tqdm.tqdm(range(256), total=256, desc='Iterating samples', leave=False):
            gt_grasps = graspnet_eval.loadGrasp(scene_id, ann_id, 'rect', camera=graspnet_eval.camera)

            is_out_of_bounds = gt_grasps.center_points[:, 0] >= 1279
            is_out_of_bounds = np.logical_or(is_out_of_bounds, gt_grasps.center_points[:, 1] >= 719)

            gt_grasps = gt_grasps[~is_out_of_bounds]

            scores = np.array(scores_list[ann_id])
            collisions = np.array(collisions_list[ann_id])
            sortidx = np.array(sortidx_list[ann_id])

            # with open(os.path.join(args.src_dir, f'scene_{scene_id:04d}', "kinect", f'{ann_id:04d}_scores.pkl'), 'rb') as f:
            #     scores = pickle.load(f)
            # with open(os.path.join(args.src_dir, f'scene_{scene_id:04d}', "kinect", f'{ann_id:04d}_collisions.pkl'), 'rb') as f:
            #     collisions = pickle.load(f)
            # with open(os.path.join(args.src_dir, f'scene_{scene_id:04d}', "kinect", f'{ann_id:04d}_sortidx.pkl'), 'rb') as f:
            #     sortidx = pickle.load(f)

            gt_grasps = gt_grasps[sortidx]
            remove_idx = np.logical_or(collisions, scores <= 0)
            remove_idx = np.logical_or(remove_idx, scores > args.max_friction_coeff)
            gt_grasps.remove(np.argwhere(remove_idx == True))
            scores = scores[~remove_idx]

            gt_grasps.scores = 1.2 - scores

            base_dir = os.path.join(args.target_dir, 'scenes', f'scene_{scene_id:04d}', args.camera, 'rect')
            os.makedirs(base_dir, exist_ok=True)
            current_save_dir = os.path.join(base_dir, f'{ann_id:04d}')
            gt_grasps.save_npy(current_save_dir)


if __name__ == '__main__':
    main()
