import argparse

from graspnetAPI.utils.visualize_logs import visualize_scene_logs, parallel_visualize_scene_logs

SPLIT_TO_SCENES = {
    'test': range(100, 190),
    'test_seen': range(100, 130),
    'test_similar': range(130, 160),
    'test_novel': range(160, 190),
}

def get_args():
    ap = argparse.ArgumentParser()

    ap.add_argument('src_dir', dest='scr_dir', type=str)
    ap.add_argument('--scene-id', dest='scene_id', type=int, default=-1)
    ap.add_argument('--split', type=str, choices=['test', 'test_seen', 'test_similar', 'test_novel'], default='test_seen')

    return ap.parse_args()

def main():
    args = get_args()

    if args.scene_id != -1:
        visualize_scene_logs(args.scene_id, args.src_dir, True)
    else:
        parallel_visualize_scene_logs(SPLIT_TO_SCENES[args.split], d, True)

if __name__ == '__main__':
    main()
