import numpy as np
import argparse

from torchmetrics import MetricCollection

from nicr_grasping.evaluation.graspnet_evaluator import GraspNetEvaluator
from nicr_grasping.evaluation import EvalParameters
from nicr_grasping.evaluation.metrics.ap import APPerObject, APPerFrame
from nicr_grasping.evaluation.metrics.statistics import NumGraspsPerObject

from nicr_grasping.evaluation.graspnet import GRASPNET_STATS_PER_SPLIT

from nicr_grasping import logger
# logger.setLevel('INFO')

def _parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument('root_dir', type=str)
    ap.add_argument('--save-dir', type=str, default=None)
    ap.add_argument('--top-k', type=int, default=5)
    ap.add_argument('--num-workers', type=int, default=1)
    ap.add_argument('--use-graspnet-collision', action='store_true')
    ap.add_argument('--vis', action='store_true')
    ap.add_argument('--log-level', type=str, default='WARNING', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])
    ap.add_argument('--num-tasks-per-child', type=int, default=50)
    ap.add_argument('--chunksize', type=int, default=50)
    ap.add_argument('--split', type=str)

    return ap.parse_args()


def main():

    args = _parse_args()
    logger.setLevel(args.log_level)

    if args.vis:
        assert args.num_workers == 1, 'Visualization only supported with num_workers=1'

    def sample_filter(sample):
        return True
        obj_name, scene_id, ann_id = sample
        if scene_id == 100:
            return True
        return False

    eval_params = EvalParameters(
        top_k=args.top_k,
        friction_coefficients=np.linspace(0.2, 1.2, 6),
        nms_translation_threshold=0.03,
        nms_rotation_threshold=30
    )

    num_unique_objects = GRASPNET_STATS_PER_SPLIT[args.split].num_unique_objects

    # substract 1 as we discard first sample of each scene as it is not part of the camera trajectory
    num_samples_per_scene = GRASPNET_STATS_PER_SPLIT[args.split].num_samples_per_scene - 1

    metrics = MetricCollection({
        'ap': APPerObject(top_k=args.top_k,
                          filter_collisions=False,
                          num_unique_objects=num_unique_objects,
                          num_samples_per_object=num_samples_per_scene),
        'ap_filtered': APPerObject(top_k=args.top_k,
                                   filter_collisions=True,
                                   num_unique_objects=num_unique_objects,
                                   num_samples_per_object=num_samples_per_scene),
        'num_grasps_per_obj': NumGraspsPerObject(filter_collisions=False, num_unique_objects=num_unique_objects),
        'num_grasps_per_obj_filtered': NumGraspsPerObject(filter_collisions=True, num_unique_objects=num_unique_objects),
        'ap_per_frame': APPerFrame(top_k=50, filter_collisions=False),
        'ap_per_frame_filtered': APPerFrame(top_k=50, filter_collisions=True)
        }, compute_groups=False)

    evaluator = GraspNetEvaluator(
        args.split,
        args.root_dir,
        eval_params,
        args.save_dir,
        use_graspnet_collision=args.use_graspnet_collision,
        sample_filter_func=sample_filter,
        num_tasks_per_child=args.num_tasks_per_child,
        chunksize=args.chunksize,
    )

    evaluator.evaluate(num_workers=args.num_workers,
                       vis=args.vis,
                       metrics=metrics)


if __name__ == '__main__':
    main()
