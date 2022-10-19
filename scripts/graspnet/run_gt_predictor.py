import tqdm
import os

import argparse
import logging

from nicr_grasping.evaluation.graspnet import gt_predictor, gt_predictor_base
from graspnetAPI.graspnet_eval import GraspNetEval


GRASPNET_ROOT = "/datasets_nas/grasping/graspnet"
# TARGET_DIR = "/results_nas/best3125/grasping/eval/GT2"
# ROOT = "/results_nas/best3125/grasping/eval/GT"
CAMERA = 'kinect'


def get_args():
    ap = argparse.ArgumentParser()

    ap.add_argument('target_dir', help="Directory where computed grasps will be saved to.",
                    type=str)
    ap.add_argument('--source-dir', dest='source_dir',
                    help="Directory where grasps labels will be loaded from. Default is original GraspNet labels",
                    type=str, default='')

    return ap.parse_args()


def main():

    args = get_args()

    if args.source_dir != '':
        assert os.path.exists(args.source_dir)
        logging.info('Using ' + args.source_dir + ' for grasp labels.')

    graspnet_eval = GraspNetEval(GRASPNET_ROOT, CAMERA, split='all')
    scene_ids = graspnet_eval.sceneIds

    os.makedirs(args.target_dir, exist_ok=True)

    for scene_idx, scene_id in enumerate(tqdm.tqdm(scene_ids, total=len(scene_ids), desc='Iterating scenes')):

        for ann_id in tqdm.tqdm(range(256), total=256, desc='Iterating samples', leave=False):
            if args.source_dir != '':
                grasps = gt_predictor_base(graspnet_eval, scene_id, ann_id, CAMERA,
                                           gt_folder=args.source_dir)
            else:
                grasps = gt_predictor_base(graspnet_eval, scene_id, ann_id, CAMERA)

            os.makedirs(os.path.join(args.target_dir, f'scene_{scene_id:04d}', CAMERA), exist_ok=True)
            current_save_dir = os.path.join(args.target_dir, f'scene_{scene_id:04d}', CAMERA, f'{ann_id:04d}')

            grasps.save_npy(current_save_dir)


if __name__ == '__main__':
    main()
