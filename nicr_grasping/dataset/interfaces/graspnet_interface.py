from typing import Dict, Any
from graspnetAPI.graspnet import GraspNet
import os

from .interface_base import DatasetInterface
from ...datatypes.modalities.image import DepthImage, ColorImage
from ...datatypes.grasp import RectangleGraspList
from ...datatypes.grasp_conversion import CONVERTER_REGISTRY

from graspnetAPI.grasp import RectGraspGroup

from typing import List


def load_grasp_file(file_path: str) -> RectangleGraspList:
    graspgroup = RectGraspGroup().from_npy(file_path)
    return CONVERTER_REGISTRY.convert(graspgroup, RectangleGraspList)


class GraspNetInterface(DatasetInterface):
    DATASET_NAME = 'graspnet'

    def __init__(self,
                 root_dir: str,
                 dest_dir: str,
                 **kwargs: Any) -> None:
        super().__init__(root_dir, dest_dir)

        self._camera = kwargs.get('camera', 'kinect')
        self._split = kwargs.get('split', 'all')

        self._graspnet_manager = GraspNet(self._root_dir,
                                          camera=self._camera,
                                          split=self._split,
                                          rect_label_root=kwargs.get('rect_label_path', None))

    def get_modalities(self) -> List[str]:
        return [ColorImage.MODALITY_NAME, DepthImage.MODALITY_NAME]

    def _collect_samples(self) -> None:
        return

    def __len__(self) -> int:
        return len(self._graspnet_manager.rgbPath)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        res = {}
        res['grasp-file'] = self._graspnet_manager.rectLabelPath[idx]
        res['depth-file'] = self._graspnet_manager.depthPath[idx]
        res['color-file'] = self._graspnet_manager.rgbPath[idx]
        res['segmentation-file'] = self._graspnet_manager.segLabelPath[idx]
        return res

    def _prepare_destination_directories(self) -> None:
        """Create directories where samples are being saved to.
        One directory for every modality and for grasp labels.
        """
        dest_dir = self.get_dest_dir()
        os.makedirs(os.path.join(dest_dir, ColorImage.MODALITY_NAME))
        os.makedirs(os.path.join(dest_dir, DepthImage.MODALITY_NAME))
        os.makedirs(os.path.join(dest_dir, 'grasp_labels'))
        os.makedirs(os.path.join(dest_dir, 'grasp_lists'))

    def _convert(self, args: tuple) -> None:
        sample_idx, sample, dest_dir, draw_mode = args
        color = ColorImage.from_file(sample['color-file'])
        depth = DepthImage.from_file(sample['depth-file'])
        grasps = load_grasp_file(sample['grasp-file'])

        sample_idx_str = self.sample_id_to_string(sample_idx)

        # sort grasps lowest to highest
        # in case of different qualities this assures
        # high quality grasps overwrite lower ones
        grasps.sort_by_quality(reverse=True)

        grasps.save(os.path.join(dest_dir, 'grasp_lists', sample_idx_str + '.pkl'))

        grasp_labels = grasps.create_sample_images((720, 1280, 1), mode=draw_mode)
        # grasp_labels = np.concatenate(grasp_labels, axis=-1)

        # np.save(os.path.join(dest_dir, 'grasp_labels', f'{sample_idx:04d}.npy'), grasp_labels)
        self.save_2d_grasp_labels(grasp_labels, dest_dir, sample_idx_str)

        color.save(os.path.join(dest_dir, color.MODALITY_NAME, sample_idx_str + color.FILE_ENDING))
        depth.save(os.path.join(dest_dir, depth.MODALITY_NAME, sample_idx_str + depth.FILE_ENDING))
