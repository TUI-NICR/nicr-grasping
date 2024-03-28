# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
.. codeauthor:: Benedict Stephan <benedict.stephan@tu-ilmenau.de>
"""
from glob import glob
import os
import re
import sys
import warnings

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import json
from copy import deepcopy
from tqdm import tqdm

from nicr_mt_scene_analysis.data import move_batch_to_device
from nicr_mt_scene_analysis.data import mt_collate

from emsanet import args as emsa_args
emsa_args.KNOWN_DATASETS = emsa_args.KNOWN_DATASETS + ('GraspNet',)

from emsanet.args import ArgParserEMSANet
from emsanet.data import DataHelper, parse_datasets, ConcatDataset
from emsanet.model import EMSANet
from emsanet.preprocessing import get_preprocessor
from emsanet.visualization import visualize_predictions
from emsanet.weights import load_weights

from nicr_mt_scene_analysis.data.preprocessing.resize import get_fullres

from graspnet_dataset import GraspNet


_SCORE_MAX = 0.999


def get_dataset(args, split):
    # prepare names, paths, and splits
    # ':' indicates joined datasets
    dataset_split = split.lower()
    n_datasets = len(parse_datasets(args.dataset))
    if 'train' == dataset_split and n_datasets > 1:
        # backward compatibility: use 'train' split for all datasets
        dataset_split = ':'.join(['train'] * n_datasets)

    # parse full dataset information
    datasets = parse_datasets(
        datasets_str=args.dataset,
        datasets_path_str=args.dataset_path,
        datasets_split_str=dataset_split
    )

    # define default kwargs dict for all datasets
    dataset_depth_mode = 'raw' if args.raw_depth else 'refined'
    default_dataset_kwargs = {
        'graspnet': {
            'depth_mode': dataset_depth_mode,
        }
    }

    # determine sample keys
    sample_keys = list(args.input_modalities) + list(args.tasks)
    # add identifier for easier debugging and plotting
    sample_keys.append('identifier')

    # instance task requires semantic for determining foreground
    if 'instance' in args.tasks and 'semantic' not in args.tasks:
        sample_keys.append('semantic')
    # rgbd (single encoder) modality still require rgb and depth
    if 'rgbd' in sample_keys:
        if 'rgb' not in sample_keys:
            sample_keys.append('rgb')
        if 'depth' not in sample_keys:
            sample_keys.append('depth')
        # remove rgbd key
        sample_keys.remove('rgbd')

    sample_keys = tuple(sample_keys)

    # get dataset instances
    dataset_instances = []
    for dataset in datasets:
        if 'none' == dataset['split']:
            # indicates that this dataset should not be loaded (e.g., for
            # training on ScanNet and SunRGB-D but validation only on SunRGB-D)
            continue

        Dataset = GraspNet

        # get default kwargs for dataset
        dataset_kwargs = deepcopy(default_dataset_kwargs[dataset['name']])

        # check if all sample keys are available
        sample_keys_avail = Dataset.get_available_sample_keys(dataset['split'])
        sample_keys_missing = set(sample_keys) - set(sample_keys_avail)

        if sample_keys_missing:
            # this indicates a common problem, however, it also happens for
            # inference ScanNet on test split
            warnings.warn(
                f"Sample keys '{sample_keys_missing}' are not available for "
                f"dataset '{dataset['name']}' and split '{dataset['split']}'. "
                "Removing them from sample keys."
            )
            sample_keys = tuple(set(sample_keys) - sample_keys_missing)

        # instantiate dataset object
        dataset_instance = Dataset(
            dataset_path=dataset['path'],
            split=dataset['split'],
            sample_keys=sample_keys,
            use_cache=args.cache_dataset,
            cache_disable_deepcopy=False,    # False as we modify samples inplace
            cameras=dataset['cameras'],
            **dataset_kwargs
        )

        dataset_instances.append(dataset_instance)

    if 1 == len(dataset_instances):
        # single dataset
        return dataset_instances[0]

    # concatenated datasets
    return ConcatDataset(dataset_instances[0], *dataset_instances[1:])


def get_datahelper(args) -> DataHelper:
    # get datasets
    dataset_train = get_dataset(args, args.split)
    dataset_valid = get_dataset(args, args.validation_split)

    # create list of datasets for validation (each with only one camera ->
    # same resolution)
    dataset_valid_list = []
    for camera in dataset_valid.cameras:
        dataset_camera = deepcopy(dataset_valid).filter_camera(camera)
        dataset_valid_list.append(dataset_camera)

    # combine everything in a data helper
    return DataHelper(
        dataset_train=dataset_train,
        subset_train=args.subset_train,
        subset_deterministic=args.subset_deterministic,
        batch_size_train=args.batch_size,
        datasets_valid=dataset_valid_list,
        batch_size_valid=args.validation_batch_size,
        n_workers=args.n_workers,
        persistent_worker=args.cache_dataset,     # only if caching is enabled
    )


def _semantic_and_instance_to_panoptic_bgr(semantic, instance):
    assert semantic.max() <= np.iinfo('uint8').max
    semantic_uint8 = semantic.astype('uint8')

    assert instance.shape == semantic.shape
    assert instance.max() <= np.iinfo('uint16').max
    instance_uint16 = instance.astype('uint16')

    r = semantic_uint8                              # semantic class
    g = (instance_uint16 >> 8).astype('uint8')      # upper 8bit of instance id
    b = (instance_uint16 & 0xFF).astype('uint8')    # lower 8bit of instance id

    # BGR for opencv
    panoptic_img = np.stack([b, g, r], axis=2)

    return panoptic_img


def write_graspnet_segmentation_output(
    batch,
    prediction,
    output_path,
    instance_use_panoptic_score=True,
    semantic_class_mapper=lambda x: x,
    compressed=True
):
    # we only write predictions (see MIRA dataset readers in
    # nicr_scene_analysis_datasets for loading)

    def _write_as_npz(dirname, tensor_to_write):
        path = os.path.join(output_path, dirname)
        for i, tensor in enumerate(tensor_to_write):
            path_i = os.path.join(path, *batch['identifier'][i][:-1])
            filename_i = batch['identifier'][i][-1] + '.npz'
            os.makedirs(path_i, exist_ok=True)
            if compressed:
                np.savez_compressed(os.path.join(path_i, filename_i), tensor)
            else:
                np.savez(os.path.join(path_i, filename_i), tensor)

    # panoptic semantic prediction (float32: class + score)
    # note panoptic merging is done on CPU
    pan_sem_scores = get_fullres(
        prediction,
        'panoptic_segmentation_deeplab_semantic_score'
    )
    pan_sem_scores = torch.clamp(pan_sem_scores, min=0, max=_SCORE_MAX)
    pan_sem_classes = get_fullres(prediction, 'panoptic_segmentation_deeplab_semantic_idx')
    pan_sem_classes = pan_sem_classes.to(torch.uint8)    # < 255 classes
    pan_sem_scores = pan_sem_scores.cpu().numpy()
    pan_sem_classes = pan_sem_classes.cpu().numpy()
    pan_sem_classes = semantic_class_mapper(pan_sem_classes)    # map classes
    pan_sem_output = pan_sem_classes.astype('float32') + pan_sem_scores
    assert (pan_sem_output.astype('uint8') == pan_sem_classes).all()

    # convert to topk format (topk, h, w) with topk=1
    pan_sem_output = pan_sem_output[:, None, ...]

    _write_as_npz('pred_panoptic_semantic', pan_sem_output)

    # panoptic instance prediction
    if instance_use_panoptic_score:
        # use panoptic score instead of instance score
        # score: score_instance_center * (mean_semantic_score_of_instance)
        pan_ins_scores = get_fullres(
            prediction,
            'panoptic_segmentation_deeplab_panoptic_score'
        )
    else:
        # use raw instance score
        # score: score_instance_center
        pan_ins_scores = get_fullres(
            prediction,
            'panoptic_segmentation_deeplab_instance_score'
        )
    pan_ins_scores = torch.clamp(pan_ins_scores, min=0, max=_SCORE_MAX)
    pan_ins_ids = get_fullres(prediction, 'panoptic_segmentation_deeplab_instance_idx')
    pan_ins_scores = pan_ins_scores.cpu().numpy()
    pan_ins_ids = pan_ins_ids.cpu().numpy()
    pan_ins_output = pan_ins_ids.astype('float32') + pan_ins_scores
    _write_as_npz('pred_panoptic_instance', pan_ins_output)

    # panoptic instance meta
    pan_ins_meta = prediction['panoptic_segmentation_deeplab_instance_meta']
    path = os.path.join(output_path, 'pred_panoptic_instance_meta')
    for i, meta in enumerate(pan_ins_meta):
        # apply semantic class mapping
        meta_i = deepcopy(meta)    # copy to be avoid to modify inplace
        for k in meta_i:
            if 'semantic_idx' in meta_i[k]:  # filter instances without pixels
                meta_i[k]['semantic_idx'] = int(semantic_class_mapper(
                    meta_i[k]['semantic_idx'])
                )
        path_i = os.path.join(path, *batch['identifier'][i][:-1])
        filename_i = batch['identifier'][i][-1] + '.json'
        os.makedirs(path_i, exist_ok=True)
        with open(os.path.join(path_i, filename_i), 'w') as f:
            json.dump(meta_i, f, sort_keys=True, indent=4)

    # predicted panoptic as png
    path = os.path.join(output_path, 'pred_panoptic_png')
    panoptic_segmentation_semantic = get_fullres(prediction, 'panoptic_segmentation_deeplab_semantic_idx').cpu().numpy()
    panoptic_segmentation_semantic = semantic_class_mapper(panoptic_segmentation_semantic)    # map classes
    panoptic_segmentation_instance = get_fullres(prediction, 'panoptic_segmentation_deeplab_instance_idx').cpu().numpy()

    for b_idx in range(panoptic_segmentation_semantic.shape[0]):
        path_i = os.path.join(path, *batch['identifier'][b_idx][:-1])
        filename_i = batch['identifier'][b_idx][-1] + '.png'
        os.makedirs(path_i, exist_ok=True)
        cv2.imwrite(
            os.path.join(path_i, filename_i),
            _semantic_and_instance_to_panoptic_bgr(
                panoptic_segmentation_semantic[b_idx],
                panoptic_segmentation_instance[b_idx]
            )
        )


def _get_args():
    parser = ArgParserEMSANet()

    # add additional arguments
    group = parser.add_argument_group('Inference')
    group.add_argument(
        '--inference-data-basepath',
        required=True,
        type=str,
        help="Path to data root directory, should contain a 'color' or 'rgb' "
             "and a 'depth' folder."
    )
    group.add_argument(    # useful for appm context module
        '--inference-input-height',
        type=int,
        default=480,
        dest='validation_input_height',    # used in test phase
        help="Network input height for predicting on inference data."
    )
    group.add_argument(    # useful for appm context module
        '--inference-input-width',
        type=int,
        default=640,
        dest='validation_input_width',    # used in test phase
        help="Network input width for predicting on inference data."
    )
    group.add_argument(
        '--depth-max',
        type=float,
        default=None,
        help="Additional max depth values. Values above are set to zero as "
             "they are most likely not valid. Note, this clipping is applied "
             "before scaling the depth values."
    )
    group.add_argument(
        '--depth-scale',
        type=float,
        default=1.0,
        help="Additional depth scaling factor to apply."
    )
    group.add_argument(
        '--shuffle-data',
        action='store_true',
        help="Shuffle data before inference.",
        default=False
    )
    group.add_argument(
        '--output-basepath',
        type=str,
        help="Where to store results. If not given, plots will be showed.",
        default=None
    )

    return parser.parse_args()


def _load_img(fp):
    img = cv2.imread(fp, cv2.IMREAD_UNCHANGED)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def main():
    args = _get_args()
    assert all(x in args.input_modalities for x in ('rgb', 'depth')), \
        "Only RGBD inference supported so far"

    device = torch.device('cuda')

    # output path
    if args.output_basepath is not None:
        print(f"writing results to '{args.output_basepath}'")
        # create output directory, but do not overwrite
        os.makedirs(args.output_basepath, exist_ok=False)
        # store argv for reproducibility
        with open(os.path.join(args.output_basepath, 'argv.txt'), 'w') as f:
            f.write(' '.join(sys.argv[1:]))

    # data and model
    data = get_datahelper(args)
    dataset_config = data.dataset_config
    model = EMSANet(args, dataset_config=dataset_config)

    # load weights
    print(f"Loading checkpoint: '{args.weights_filepath}'")
    checkpoint = torch.load(args.weights_filepath)
    state_dict = checkpoint['state_dict']
    if 'epoch' in checkpoint:
        print(f"-> Epoch: {checkpoint['epoch']}")
    load_weights(args, model, state_dict, verbose=True)

    torch.set_grad_enabled(False)
    model.eval()
    model.to(device)

    # build preprocessor
    preprocessor = get_preprocessor(
        args,
        dataset=data.datasets_valid[0],
        phase='test',
        multiscale_downscales=None
    )

    # get samples
    rgb_path = os.path.join(args.inference_data_basepath, 'color')
    if not os.path.exists(rgb_path):
        rgb_path = os.path.join(args.inference_data_basepath, 'rgb')
    depth_path = os.path.join(args.inference_data_basepath, 'depth')

    # match samples based on timestamp in filename
    # matching is done from rgb to depth
    rgb_filepaths = sorted(glob(os.path.join(rgb_path, '*.png')))
    depth_filepaths = sorted(glob(os.path.join(depth_path, '*.png')))

    rgb_timestamps = np.array(
        [int(re.findall(r'[0-9]+', os.path.basename(fp))[0])
         for fp in rgb_filepaths]
    )
    depth_timestamps = np.array(
        [int(re.findall(r'[0-9]+', os.path.basename(fp))[0])
         for fp in depth_filepaths]
    )

    # this allows for data captured with timestamps and
    # also works with the GraspNet-1Billion dataset
    filepaths = []
    for fp_rgb, t_rgb in zip(rgb_filepaths, rgb_timestamps):
        t_diff = np.abs(depth_timestamps - t_rgb)
        idx = np.argmin(t_diff)
        depth_fp = depth_filepaths[idx]
        filepaths.append((fp_rgb, depth_fp))

    # shuffle samples
    if args.shuffle_data:
        np.random.shuffle(filepaths)

    for fp_rgb, fp_depth in tqdm(filepaths,
                                 disable=args.output_basepath is None):
        # load rgb and depth image
        img_rgb = _load_img(fp_rgb)

        img_depth = _load_img(fp_depth).astype('float32')
        if args.depth_max is not None:
            img_depth[img_depth > args.depth_max] = 0
        img_depth *= args.depth_scale

        # preprocess sample
        sample = preprocessor({
            'rgb': img_rgb,
            'depth': img_depth,
            'identifier': [os.path.basename(os.path.splitext(fp_rgb)[0])]
        })

        # add batch axis as there is no dataloader
        batch = mt_collate([sample])
        batch = move_batch_to_device(batch, device=device)

        # apply model
        predictions = model(batch, do_postprocessing=True)

        # visualize predictions
        preds_viz = visualize_predictions(
            predictions=predictions,
            batch=batch,
            dataset_config=dataset_config
        )

        write_graspnet_segmentation_output(
            batch,
            predictions,
            args.output_basepath
        )

        # show results
        _, axs = plt.subplots(2, 4, figsize=(20, 12), dpi=150)
        [ax.set_axis_off() for ax in axs.ravel()]

        axs[0, 0].set_title('RGB')
        axs[0, 0].imshow(
            img_rgb
        )
        axs[0, 1].set_title('Depth')
        axs[0, 1].imshow(
            img_depth,
            interpolation='nearest'
        )
        axs[0, 2].set_title('Semantic')
        axs[0, 2].imshow(
            preds_viz['semantic_segmentation_idx_fullres'][0],
            interpolation='nearest'
        )
        axs[0, 3].set_title('Semantic (panoptic)')
        axs[0, 3].imshow(
            preds_viz['panoptic_segmentation_deeplab_semantic_idx_fullres'][0],
            interpolation='nearest'
        )
        axs[1, 0].set_title('Instance (panoptic)')
        axs[1, 0].imshow(
            preds_viz['panoptic_segmentation_deeplab_instance_idx_fullres'][0],
            interpolation='nearest'
        )
        axs[1, 1].set_title('Instance centers')
        axs[1, 1].imshow(
            preds_viz['instance_centers'][0]
        )
        axs[1, 2].set_title('Instance offsets')
        axs[1, 2].imshow(
            preds_viz['instance_offsets'][0]
        )
        axs[1, 3].set_title('Panoptic (with orientations)')
        axs[1, 3].imshow(
            preds_viz['panoptic_segmentation_deeplab'][0],
            interpolation='nearest'
        )

        plt.suptitle(
            f"Image: ({os.path.basename(fp_rgb)}, "
            f"{os.path.basename(fp_depth)}), "
            f"Model: {args.weights_filepath}, "
            #f"Scene: {preds_viz['scene'][0]}"
        )
        plt.tight_layout()

        if args.output_basepath is not None:
            fp = os.path.join(
                args.output_basepath,
                f"{os.path.basename(fp_rgb)}.png"
            )
            plt.savefig(fp, bbox_inches='tight', pad_inches=0.05, dpi=150)
            plt.close()
        else:
            plt.show()


if __name__ == '__main__':
    main()
