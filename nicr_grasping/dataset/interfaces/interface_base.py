import abc
import enum
import os
from typing import Dict, List, Tuple
import multiprocessing as mp
import tqdm
import json
from datetime import datetime

import scipy.sparse as sp

from ...datatypes.modalities.image import ColorImage, DepthImage
from ...datatypes.grasp.grasp_2d import RectangleGraspDrawingMode

from .. import SPLIT_DEFINITIONS
from ..split_generation import SplitGenerator

#TODO: Write comments for abstact methods on how to fill them. Used in documentation.

class ConversionMode(enum.Enum):
    MULTIGRASP_2D = 0
    SINGLEGRASP_2D = 1

class DatasetInterface(abc.ABC):
    DATASET_NAME = ''
    def __init__(self,
                 root_dir : str,
                 dest_dir : str) -> None:
        self._root_dir = root_dir
        self._dest_dir = dest_dir

        self._dest_root = self.get_dest_dir()

    def get_dest_dir(self) -> str:
        return os.path.join(self._dest_dir, self.DATASET_NAME)

    def sample_id_to_string(self, sample_id: int):
        return str(sample_id).rjust(len(str(len(self))), '0')

    def save_2d_grasp_labels(self,
                             grasp_labels: "numpy.ndarray",
                             dest_dir: str,
                             sample_idx_str: str):
        for label, name in zip(grasp_labels, ['quality', 'angle', 'width']):
            # label = grasp_labels[:, :, label_idx]
            sparse_matrix = sp.coo_matrix(label.squeeze())
            sp.save_npz(os.path.join(dest_dir, 'grasp_labels', sample_idx_str + f'_{name}'), sparse_matrix)

    @abc.abstractmethod
    def _prepare_destination_directories(self):
        """Create directories where samples are being saved to.
        One directory for every modality and for grasp labels.
        """
        pass

    def convert(self,
                mode: ConversionMode = ConversionMode.MULTIGRASP_2D,
                draw_mode: RectangleGraspDrawingMode = RectangleGraspDrawingMode.INNER_RECTANGLE,
                num_samples: int = -1,
                num_worker: int = 1) -> None:

        if num_samples == -1:
            num_samples = len(self)

        assert num_samples > 0

        self._prepare_destination_directories()


        dest_dir = self.get_dest_dir()
        if mode == ConversionMode.MULTIGRASP_2D:
            args = [(sample_idx, sample, dest_dir, draw_mode) for sample_idx, sample in enumerate(self)]
            args = args[:num_samples]

            with mp.Pool(num_worker) as pool:
                r = list(tqdm.tqdm(pool.imap(self._convert, args), total=len(args)))
        elif mode == ConversionMode.SINGLEGRASP_2D:
            raise NotImplementedError()

        self.write_metadata(conversion_mode = mode, draw_mode = draw_mode)
        self.generate_splits()

        return

    def write_metadata(self, **kwargs) -> None:
        metadata = {
            'num_total_samples': len(self),
            'conversion_date': str(datetime.now()),
            'modalities': self.get_modalities()
        }

        for key, value in kwargs.items():
            metadata[key] = str(value)

        with open(os.path.join(self.get_dest_dir(), 'metadata.json'), 'w') as f:
            json.dump(metadata, f)

    def generate_splits(self):
        split_params = SPLIT_DEFINITIONS[self.DATASET_NAME]
        split_generator = SplitGenerator(
            self._dest_root, **split_params
        )

        split_generator.generate_splits()

    @abc.abstractmethod
    def get_modalities(self) -> List[str]:
        pass

    @abc.abstractmethod
    def _convert(self, args) -> None:
        pass

    @abc.abstractmethod
    def _collect_samples(self):
        pass

    @abc.abstractmethod
    def __len__(self):
        raise NotImplementedError

    @abc.abstractmethod
    def __getitem__(self, idx) -> Dict[str, str]:
        pass

    # @abc.abstractmethod
    # def get_grasps(self, idx):
    #     pass

    # @abc.abstractmethod
    # def get_color_image(self, idx):
    #     pass

    # @abc.abstractmethod
    # def get_depth_image(self, idx):
    #     pass
