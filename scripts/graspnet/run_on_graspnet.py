import glob
import os
import numpy as np
import torch
# from PIL import Image
import cv2

import torch
import matplotlib.pyplot as plt

from pathlib import Path

import detectron2.data.transforms as T
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
)
from detectron2.data import (
    DatasetCatalog
)
from detectron2.engine import (
    DefaultPredictor,
    default_argument_parser,
)

from detectron2.modeling import build_model
from detectron2.config import get_cfg

import nicr_detectron2.modeling
from nicr_detectron2.data.graspnet_mapper import GraspDatasetMapper as DatasetMapper
from nicr_detectron2.config.default_config import add_grasp_config

from graspnetAPI.graspnet import GraspNet

from torchvision.transforms.functional import (
    resize, rotate, center_crop, crop, hflip, vflip, pad
)

from nicr_grasping.evaluation.graspnet import run_predictor
from nicr_grasping import graspnet_dataset_path

CAMERA = 'kinect'

GRASPNET_DIR = graspnet_dataset_path()

np.random.seed(2233)

class Predictor(DefaultPredictor):
    def __init__(self, cfg):
        self.cfg = cfg.clone()  # cfg can be modified by model
        self.model = build_model(self.cfg)
        self.model.eval()
        if len(cfg.DATASETS.TEST):
            self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

        checkpointer = DetectionCheckpointer(self.model)
        print('Loading checkpoint')
        checkpointer.load(cfg.MODEL.WEIGHTS)

        # self.aug = T.ResizeShortestEdge(
        #     [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        # )

        self.input_format = cfg.INPUT.FORMAT

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(
            DatasetCatalog.get(dataset_name),
            mapper=DatasetMapper(cfg, is_train=False))

    def predict(self, d):
        shape = d[0]['image']['color'].shape
        shortest_edge = min(shape[1:])
        padding = [(s - shortest_edge) // 2 for s in shape[1:]]

        scaling_factor = 320 / shortest_edge
        for key in d[0]['image']:
            new_shape = [int(s * scaling_factor) for s in d[0]['image'][key].shape[1:]]
            d[0]['image'][key] = resize(d[0]['image'][key], new_shape)
        #     d[0]['image'][key] = d[0]['image'][key][:, padding[0]:d[0]['image'][key].shape[1]-padding[0], padding[1]:d[0]['image'][key].shape[2]-padding[1]]
        #     d[0]['image'][key] = resize(d[0]['image'][key], (320, 320))

        def _resize_output(output, shortest_edge, padding):
            # prediction is 320x320 resize back to original crop
            output = cv2.resize(output, shape[1:][::-1])

            # pad image back to original resolution
            # needed for grasp extraction as grasp centers are used
            # for depth lookup and need to correspond to original image pixels
            # pred_pil = pad(pred_pil, padding[::-1])
            # output = cv2.copyMakeBorder(output, padding[0], padding[0], padding[1], padding[1], cv2.BORDER_CONSTANT)

            return output

        with torch.no_grad():
            predictions = self.model(d)
            for pred_i in range(len(predictions)):
                for key in predictions[pred_i]:
                    p = predictions[pred_i][key]
                    if isinstance(p, torch.Tensor):
                        p = p.cpu().numpy()
                        p = np.moveaxis(p, 0, -1)

                        p = _resize_output(p, shortest_edge, padding)

                        # convert to numpy
                        predictions[pred_i][key] = np.array(p).squeeze()
                    elif isinstance(p, dict):
                        for extra_key, p_extra in p.items():
                            p_extra = p_extra.cpu().numpy()

                            p_extra = np.moveaxis(p_extra, 0, -1)

                            p_extra = _resize_output(p_extra, shortest_edge, padding)

                            # convert to numpy
                            predictions[pred_i][key][extra_key] = np.array(p_extra).squeeze()

            return predictions


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg = add_grasp_config(cfg)

    cfg.merge_from_file(args.config_file)
    # cfg = CfgNode(cfg)

    cfg.MODEL.WEIGHTS = ""

    cfg.merge_from_list(args.opts)
    if cfg.MODEL.WEIGHTS == "":
        weights = Path(args.config_file).parent / 'model_final.pth'
        if not weights.exists():
            print('model_final.pth not found. Looking for most recent checkpoint')
            checkpoints = list(Path(args.config_file).parent.glob('*.pth'))
            checkpoints = sorted(checkpoints)
            weights = checkpoints[-1]
        cfg.MODEL.WEIGHTS = str(weights)

    print('Loading weights from {}'.format(cfg.MODEL.WEIGHTS))

    cfg.freeze()

    # default_setup(cfg, args)
    return cfg

def main():
    ap = default_argument_parser()
    ap.add_argument('--debug', action='store_const', const=True, default=False, dest='debug')
    ap.add_argument('--scene_ids', type=int, nargs='*', default=None)

    args = ap.parse_args()
    cfg = setup(args)

    predictor = Predictor(cfg)

    # g = GraspNet(GRASPNET_DIR, camera=CAMERA, split='test_seen')
    g = GraspNet(GRASPNET_DIR, camera=CAMERA, split='test_seen')
    # ge_k = GraspNetEval(root = GRASPNET_DIR, camera = CAMERA, split = 'test_seen')

    # savedir = '/results_nas/lahi6385/grasping/bench_ggcnn2_gauss_slow/output_cornell/24_01_2022-08_37_39-277409'
    # savedir = os.path.join(SAVE_DIR_TEMPLATE.format(model=cfg.MODEL.META_ARCHITECTURE,
    #                                                 timestamp=Path(cfg.OUTPUT_DIR).stem))
    # savedir = str(Path(args.config_file).parent / 'graspnet_predictions/')
    savedir = str(Path('/tmp', 'graspnet_predictions'))
    os.makedirs(savedir, exist_ok=True)

    accs = []

    run_predictor(predictor, g, savedir=savedir, debug=args.debug, scene_ids=args.scene_ids)

    print('done')


if __name__ == '__main__':
    main()
