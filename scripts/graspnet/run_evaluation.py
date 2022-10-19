# GraspNetAPI example for evaluate grasps for a scene.
# change the graspnet_root path
import numpy as np
import pickle
import time
import os
import json
from graspnetAPI import GraspNetEval
import argparse

from nicr_grasping import graspnet_dataset_path

# graspnet_root = '/datasets_nas/grasping/graspnet'  # '/datasets_nas/grasping/graspnet' # ROOT PATH FOR GRASPNET

graspnet_root = graspnet_dataset_path()

def get_args():
    ap = argparse.ArgumentParser()

    ap.add_argument('src_folder', type=str, help='Folder from which the grasps will be loaded.')

    ap.add_argument('--eval-all', action='store_const', const=True, default=False,
                    dest='eval_all')
    ap.add_argument('--top-k', dest='top_k', type=int, default=50, help='Number of top grasps to use for evaluation. Ignored if --eval-all is used.')
    ap.add_argument('--scene-id', type=int, help='If supplied onyl this scene will be evaluated.', default=-1,
                    dest='scene_id')
    ap.add_argument('--camera', type=str, choices=['kinect', 'realsense'],
                    dest='camera', default='kinect')
    ap.add_argument('--split', type=str, choices=['train', 'test', 'test_seen', 'test_novel', 'test_similar'], default='train')

    ap.add_argument('--log-eval', action='store_const', const=True, help='If supplied logs will be saved here as json.', default=False,
                    dest='log_eval')

    ap.add_argument('--num-worker', dest='num_worker', type=int, default=10)

    return ap.parse_args()


def main():

    args = get_args()

    assert os.path.exists(args.src_folder)

    top_k = None if args.eval_all else args.top_k

    if args.scene_id != -1:
        split = 'all'
    else:
        split = args.split

    ge_k = GraspNetEval(root = graspnet_root, camera = args.camera, split = split)

    eval_start = time.time()
    if args.scene_id != -1:
        print('Evaluating scene:{}, camera:{}'.format(args.scene_id, args.camera))
        log_dict = {}
        acc = ge_k.eval_scene(scene_id = args.scene_id, dump_folder = args.src_folder, vis=False, TOP_K=top_k,
                              log_dict = log_dict)

        with open(os.path.join(args.src_folder, f'scene_{args.scene_id:04d}_log.json'), 'w') as f:
            json.dump(log_dict, f)
    else:
        if split == "train":
            acc, ap = ge_k.eval_train(dump_folder=args.src_folder, proc=args.num_worker, log=args.log_eval, TOP_K=top_k)
        elif split == "test":
            acc, ap = ge_k.eval_all(dump_folder=args.src_folder, proc=args.num_worker, log=args.log_eval, TOP_K=top_k)
        elif split == "test_seen":
            acc, ap = ge_k.eval_seen(dump_folder=args.src_folder, proc=args.num_worker, log=args.log_eval, TOP_K=top_k)
        elif split == "test_novel":
            acc, ap = ge_k.eval_novel(dump_folder=args.src_folder, proc=args.num_worker, log=args.log_eval, TOP_K=top_k)
        else:
            pass

    # np_acc = np.array(acc)
    print(acc.keys())
    eval_time = time.time()-eval_start
    print('\n')
    print('RESULTS HERE: \n')
    for key in acc:
        print('mean accuracy {}:\t{}'.format(key, np.mean(acc[key])))

    aps = {}

    for key in acc.keys():
        if split == "train":
            aps[f'AP-train-{key}'] = ap
            print('\nEvaluation Result:, AP={}'.format(ap))
        elif split == "test":
            print(key, '\nEvaluation Result:, AP={}, AP Seen={}, AP Similar={}, AP Novel={}'.format(ap[key][0], ap[key][1], ap[key][2], ap[key][3]))
            aps[f'AP-seen-{key}'] = ap[key][1]
            aps[f'AP-seen-08-{key}'] = np.mean(acc[key][:30, :, :, 3])
            aps[f'AP-seen-04-{key}'] = np.mean(acc[key][:30, :, :, 1])

            aps[f'AP-similar-{key}'] = ap[key][2]
            aps[f'AP-similar-08-{key}'] = np.mean(acc[key][30:60, :, :, 3])
            aps[f'AP-similar-04-{key}'] = np.mean(acc[key][30:60, :, :, 1])

            aps[f'AP-novel-{key}'] = ap[key][3]
            aps[f'AP-novel-08-{key}'] = np.mean(acc[key][60:90, :, :, 3])
            aps[f'AP-novel-04-{key}'] = np.mean(acc[key][60:90, :, :, 1])

        elif split == "test_seen":
            print(key, '\tEvaluation Result:, AP Seen={}'.format(ap[key]))
            aps[f'AP-seen-{key}'] = ap[key]
            aps[f'AP-seen-08-{key}'] = np.mean(acc[key][:, :, :, 3])
            aps[f'AP-seen-04-{key}'] = np.mean(acc[key][:, :, :, 1])
        elif split == "test_novel":
            print(key, '\nEvaluation Result:, AP Novel={}'.format(ap))
            aps[f'AP-novel-08-{key}'] = np.mean(acc[key][:, :, :, 3])
            aps[f'AP-novel-04-{key}'] = np.mean(acc[key][:, :, :, 1])
        else:
            pass
    print('eval time:\t{}'.format(eval_time))

    with open(os.path.join(args.src_folder, 'eval.json'), 'w') as f:
        json.dump(aps, f)

    with open(os.path.join(args.src_folder, 'acc.pkl'), 'wb') as f:
        pickle.dump(acc, f)


if __name__ == '__main__':
    main()
