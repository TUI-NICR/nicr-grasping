import argparse
import os

from nicr_grasping.dataset.interfaces import get_interface, get_interfaces
from nicr_grasping.datatypes.grasp import RectangleGraspDrawingMode


def get_args():
    ap = argparse.ArgumentParser(description='Script for converting datasets into the grasp-benchmark format. Interface will be selected by name.')

    ap.add_argument('datasetname', type=str, choices=get_interfaces())
    ap.add_argument('source_path', type=str)
    ap.add_argument('destination_path', type=str)
    ap.add_argument('--num-worker', type=int, default=1, dest='num_worker', help="Number of parallel jobs for generation")
    ap.add_argument('--num-samples', type=int, default=-1, dest='num_samples', help="Number of samples to be generated. -1 for all samples.")

    ap.add_argument('--draw-mode', dest='draw_mode', type=str, choices=[mode.name for mode in RectangleGraspDrawingMode], help='Mode for drawing labels.')

    # GraspNet
    ap.add_argument('--rect-label-path', type=str, default=None, dest='rect_label_path',
                    help='Used only for GraspNet interface. If specified the grasp labels will be loaded from this directory instead.')

    return ap.parse_args()

def main():
    args = get_args()

    mode_mapping = {mode.name: mode for mode in RectangleGraspDrawingMode}

    assert os.path.exists(args.source_path)

    os.makedirs(args.destination_path)

    interface = get_interface(args.datasetname)(args.source_path, args.destination_path,
                                                rect_label_path=args.rect_label_path)
    interface.convert(num_samples=args.num_samples, num_worker=args.num_worker, draw_mode=mode_mapping[args.draw_mode])

if __name__ == '__main__':
    main()
