import os
import json
import numpy as np
import scipy.sparse as sp

import matplotlib.pyplot as plt

from pathlib import Path

from typing import List, Dict, Tuple, Union
from nicr_grasping.datatypes.grasp.grasp_lists import GraspList

from nicr_grasping.datatypes.modalities import get_modality_from_name
from nicr_grasping.datatypes.grasp import RectangleGraspList
from nicr_grasping.datatypes.modalities.modality_base import ModalityBase


class DatasetInfo:
    def __init__(self,
                 root_path: Union[Path, str]) -> None:
        self._root_path = Path(root_path)

        with (self._root_path / 'metadata.json').open('r') as f:
            self._metadata = json.load(f)

    def get_example_samples(self,
                            sample_ids: Union[List[int], None] = None,
                            num_samples: int = 1) -> Dict[int, Tuple[List[ModalityBase], Dict[str, np.ndarray], GraspList]]:
        _sample_ids = sample_ids or np.random.choice(list(range(self._metadata['num_total_samples'])), num_samples).tolist()

        sample_data = {}
        for sample_id in _sample_ids:
            sample_data[sample_id] = self._load_sample(sample_id)

        return sample_data

    @property
    def sample_size(self) -> int:
        return self._metadata['num_total_samples']

    @property
    def modalities(self) -> List[str]:
        return self._metadata['modalities']

    def _load_sample(self, sample_id: int) -> Tuple[List[ModalityBase], Dict[str, np.ndarray], GraspList]:
        sample_i_str = str(sample_id).rjust(len(str(self._metadata['num_total_samples'])), '0')
        grasps = RectangleGraspList.load(os.path.join(self._root_path / 'grasp_lists' / (sample_i_str + '.pkl')))

        inputs = []

        for i, modality_str in enumerate(self._metadata['modalities']):
            modality = get_modality_from_name(modality_str)()

            modality.load(os.path.join(self._root_path / modality.MODALITY_NAME / (sample_i_str + modality.FILE_ENDING)))

            inputs.append(modality)

        labels = {}
        for label_i, label_type in enumerate(['quality', 'angle', 'width']):
            label = sp.load_npz(os.path.join(self._root_path / 'grasp_labels' / (sample_i_str + f'_{label_type}.npz')))
            label = label.toarray()

            labels[label_type] = label

        return inputs, labels, grasps

    def plot_example_images(self,
                            num_samples: int = 1,
                            top_k: Union[int, None] = None) -> Dict[int, Tuple[plt.Figure, List[plt.Axes]]]:
        sample_idx = np.random.choice(list(range(self._metadata['num_total_samples'])), num_samples)

        res = {}

        for sample_i in sample_idx:
            sample_i_str = str(sample_i).rjust(len(str(self._metadata['num_total_samples'])), '0')
            grasps = RectangleGraspList.load(os.path.join(self._root_path / 'grasp_lists' / (sample_i_str + '.pkl')))
            grasps.sort_by_quality()
            if top_k is not None:
                grasps = grasps[:top_k]

            # f, axes = plt.subplots(1, len(self._metadata['modalities']) + 1)
            f = plt.figure(figsize=(13, 5))
            spec = f.add_gridspec(ncols=3, nrows=3)
            axes = []
            for i, modality_str in enumerate(self._metadata['modalities']):
                modality = get_modality_from_name(modality_str)()

                modality.load(os.path.join(self._root_path / modality.MODALITY_NAME / (sample_i_str + modality.FILE_ENDING)))

                ax = f.add_subplot(spec[0, i])
                ax.set_title(modality.MODALITY_NAME)
                ax.imshow(modality.data)
                axes.append(ax)

                ax = f.add_subplot(spec[1:, i])
                data = modality.data
                data = grasps.plot(data)

                ax.set_title(modality.MODALITY_NAME + ' w/ labels')
                ax.imshow(data)
                axes.append(ax)

            for label_i, label_type in enumerate(['quality', 'angle', 'width']):
                label = sp.load_npz(os.path.join(self._root_path / 'grasp_labels' / (sample_i_str + f'_{label_type}.npz')))
                label = label.toarray()

                ax = f.add_subplot(spec[label_i, -1])
                ax.set_title(label_type)
                ax.imshow(label)

                axes.append(ax)

            for ax in axes:
                ax.axis('off')
            f.tight_layout()

            res[sample_i] = (f, axes)

        return res

    def get_summary(self) -> None:
        print('Dataset summary:')
        print('\tName:\t\t', self._root_path.stem)
        print('\tNum samples:\t', self._metadata['num_total_samples'])
        print('\tModalities:\t', ', '.join(self._metadata['modalities']))
