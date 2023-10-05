# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
.. codeauthor:: Benedict Stephan <benedict.stephan@tu-ilmenau.de>
"""
from typing import Any, Optional, Tuple

import os

import cv2
import numpy as np

from nicr_scene_analysis_datasets.dataset_base import build_dataset_config
from nicr_scene_analysis_datasets.dataset_base import DatasetConfig
from nicr_scene_analysis_datasets.dataset_base import DepthStats
from nicr_scene_analysis_datasets.dataset_base import RGBDDataset
from nicr_scene_analysis_datasets.pytorch import _PytorchDatasetWrapper
from nicr_scene_analysis_datasets.dataset_base import SampleIdentifier
from nicr_scene_analysis_datasets.dataset_base import SceneLabelList
from nicr_scene_analysis_datasets.dataset_base import SemanticLabelList
from nicr_scene_analysis_datasets.dataset_base import SemanticLabel


AVAILABLE_SAMPLE_KEYS = ('identifier', 'rgb', 'depth', 'semantic', 'instance')
CAMERAS = ('kinect',)
SEMANTIC_N_CLASSES = 2
SEMANTIC_LABEL_LIST = SemanticLabelList([
    # class_name, is_thing, use orientations, color
    SemanticLabel('void', False, False, (0, 0, 0)),  # actually does not exist
    SemanticLabel('bg', False, False, (255, 223, 94)),
    SemanticLabel('fg', True,  False, (109, 212, 106)),
])

TRAIN_SPLIT_DEPTH_STATS = DepthStats(
    min=85.0,
    max=11089.0,
    mean=487.9365179519856,
    std=263.3877170299714
)


class GraspNet(RGBDDataset, _PytorchDatasetWrapper):
    def __init__(
        self,
        *,
        dataset_path: Optional[str] = None,
        split: str = 'train',
        sample_keys: Tuple[str] = ('rgb', 'depth', 'semantic'),
        use_cache: bool = False,
        cameras: Optional[Tuple[str]] = None,
        depth_mode: str = 'raw',
        **kwargs: Any
    ) -> None:
        super().__init__(
            dataset_path=dataset_path,
            depth_mode=depth_mode,
            sample_keys=sample_keys,
            use_cache=use_cache,
            **kwargs
        )

        assert depth_mode in ('raw',)
        assert all(sk in AVAILABLE_SAMPLE_KEYS for sk in sample_keys)
        self._semantic_n_classes = SEMANTIC_N_CLASSES
        self._split = split
        self._depth_mode = depth_mode

        # cameras
        if cameras is None:
            # use all available cameras (=default dummy camera)
            self._cameras = CAMERAS
        else:
            # use subset of cameras (does not really apply to this dataset)
            assert all(c in CAMERAS for c in cameras)
            self._cameras = cameras

        # load file list
        if dataset_path is not None:
            raise NotImplemented('This code is not for training. Only for inference!')

        elif not self._disable_prints:
            print(f"Loaded GraspNet dataset without files")

        # build config object
        self._config = build_dataset_config(
            semantic_label_list=SEMANTIC_LABEL_LIST,
            scene_label_list=SceneLabelList(),
            depth_stats=TRAIN_SPLIT_DEPTH_STATS
        )

        # register loader functions
        self.auto_register_sample_key_loaders()

    @property
    def cameras(self) -> Tuple[str]:
        return self._cameras

    @property
    def config(self) -> DatasetConfig:
        return self._config

    @property
    def split(self) -> str:
        return self._split

    @property
    def depth_mode(self) -> str:
        return self._depth_mode

    def __len__(self) -> int:
        return len(self._files['rgb'])

    @staticmethod
    def get_available_sample_keys(split: str) -> Tuple[str]:
        return AVAILABLE_SAMPLE_KEYS

    def _load(self, filepath: str) -> np.ndarray:
        fp = os.path.join(self._dataset_path, filepath)

        img = cv2.imread(fp, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise IOError(f"Unable to load image: '{fp}'")
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img

    def _load_rgb(self, idx) -> np.ndarray:
        return self._load(self._files['rgb'][idx])

    def _load_depth(self, idx) -> np.ndarray:
        return self._load(self._files['depth'][idx])

    def _load_identifier(self, idx: int) -> Tuple[str]:
        # e.g. scene_0160/kinect/rgb/0018.png
        # the folder structure is not nice, so we have to strip some parts
        fp = self._files['rgb'][idx]
        scene, camera, _, filename = fp.split('/')
        return SampleIdentifier((scene, camera, os.path.splitext(filename)[0]))

    def _load_semantic(self, idx: int) -> np.ndarray:
        # use label to derive foreground mask
        # 0: void - actually does not exists
        # 1: background
        # 2: foreground
        semantic = (self._load(self._files['label'][idx]) != 0).astype('uint8')
        semantic += 1  # shift by 1 to add void

        return semantic

    def _load_instance(self, idx: int) -> np.ndarray:
        return self._load(self._files['label'][idx]).astype('int32')
