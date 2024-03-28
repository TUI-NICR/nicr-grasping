import cv2
import os
import glob
import numpy as np
from typing import Dict, List, Any

import tqdm

from .interface_base import DatasetInterface, ConversionMode
from ...datatypes.modalities.image import DepthImage, ColorImage
from ...datatypes.grasp import RectangleGraspList, RectangleGrasp


def _gr_text_to_no(line: str, offset: tuple[int, int] = (0, 0)) -> List[int]:
    """Transform a single point from a Cornell file line to a pair of ints.

    Parameters
    ----------
    line : str
        Line from Cornell grasp file
    offset : tuple, optional
        Offset to apply to point positions, by default (0, 0)

    Returns
    -------
    list
        Point as (y, x)
    """
    x, y = line.split()
    return [int(round(float(y))) - offset[0], int(round(float(x))) - offset[1]]


class CornellInterface(DatasetInterface):
    DATASET_NAME = 'cornell'

    def __init__(self,
                 root_dir: str,
                 dest_dir: str,
                 **kwargs: Any) -> None:
        super().__init__(root_dir, dest_dir)
        self._grasp_files: List[str] = []

        self._collect_samples()

    def _collect_samples(self) -> None:
        graspf = glob.glob(os.path.join(self._root_dir, 'pcd*cpos.txt'))
        graspf.sort()
        num_files = len(graspf)
        if num_files == 0:
            raise FileNotFoundError('No dataset files found. Check path: {}'.format(self._root_dir))

        self._grasp_files = graspf

    def __len__(self) -> int:
        return len(self._grasp_files)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        res = {}
        res['grasp-file'] = self._grasp_files[idx]
        res['depth-file'] = res['grasp-file'].replace('cpos.txt', 'd.tiff')
        res['color-file'] = res['grasp-file'].replace('cpos.txt', 'r.png')
        return res

    def load_grasp_file(self, file_path: str) -> RectangleGraspList:
        grs = RectangleGraspList()
        with open(file_path) as f:
            while True:
                # Load 4 lines at a time, corners of bounding box.
                p0 = f.readline()
                if not p0:
                    break  # EOF
                p1, p2, p3 = f.readline(), f.readline(), f.readline()
                try:
                    gr = np.array([
                        _gr_text_to_no(p3),
                        _gr_text_to_no(p0),
                        _gr_text_to_no(p1),
                        _gr_text_to_no(p2)
                    ], dtype=float)

                    # switch x and y coords to be consistent with grasp representation in nicr_grasping
                    gr = gr[:, ::-1]

                    rgr = RectangleGrasp.from_points(gr)
                    rgr.quality = 1

                    grs.append(rgr)

                except ValueError:
                    # Some files contain weird values.
                    continue
        return grs

    def _prepare_destination_directories(self) -> None:
        """Create directories where samples are being saved to.
        One directory for every modality and for grasp labels.
        """
        dest_dir = self.get_dest_dir()
        os.makedirs(os.path.join(dest_dir, ColorImage.MODALITY_NAME))
        os.makedirs(os.path.join(dest_dir, DepthImage.MODALITY_NAME))
        os.makedirs(os.path.join(dest_dir, 'grasp_labels'))
        os.makedirs(os.path.join(dest_dir, 'grasp_lists'))

    def get_modalities(self) -> List[str]:
        return [ColorImage.MODALITY_NAME, DepthImage.MODALITY_NAME]

    def _convert(self, args: tuple) -> None:
        sample_idx, sample, dest_dir, draw_mode = args

        color = ColorImage.from_file(sample['color-file'])
        depth = DepthImage.from_file(sample['depth-file'])
        grasps = self.load_grasp_file(sample['grasp-file'])

        sample_idx_str = self.sample_id_to_string(sample_idx)

        # sort grasps lowest to highest
        # in case of different qualities this assures
        # high quality grasps overwrite lower ones
        grasps.sort_by_quality(reverse=True)

        grasps.save(os.path.join(dest_dir, 'grasp_lists', sample_idx_str + '.pkl'))

        grasp_labels = grasps.create_sample_images((480, 640, 1), mode=draw_mode)

        self.save_2d_grasp_labels(grasp_labels, dest_dir, sample_idx_str)

        color.save(os.path.join(dest_dir, color.MODALITY_NAME, sample_idx_str + color.FILE_ENDING))
        depth.save(os.path.join(dest_dir, depth.MODALITY_NAME, sample_idx_str + depth.FILE_ENDING))
