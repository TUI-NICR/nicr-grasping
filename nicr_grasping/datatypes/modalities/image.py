import cv2
import numpy as np
from typing import Union
from imageio import imread, imwrite

from .modality_base import ModalityBase
from ..rotation import Rotation2D
from . import register_modality


class ImageBase(ModalityBase):
    IMREAD_FLAG = cv2.IMREAD_UNCHANGED
    INTERPOLATION_FLAG = cv2.INTER_LINEAR

    MODALITY_NAME = 'image'
    FILE_ENDING = '.png'

    def __init__(self) -> None:
        """Basic type for images.
        The class properties IMREAD_FLAG and IMWRITE_FLAG
        should be set by all derivatives of this class so
        loading and saving works.
        """
        super(ImageBase, self).__init__()

    def load(self, path: str) -> None:
        self.data = cv2.imread(path, self.IMREAD_FLAG)

    def save(self, path: str) -> None:
        cv2.imwrite(path, self.data)

    def rotate(self,
               rotation: Rotation2D,
               center: Union[np.ndarray, None] = None) -> np.ndarray:
        assert isinstance(rotation, Rotation2D)

        if center is None:
            center = np.array(self.data.shape[1::-1]) / 2

        rot_mat = cv2.getRotationMatrix2D(center, rotation.angle, 1.0)
        result = cv2.warpAffine(self.data, rot_mat,
                                self.data.shape[1::-1],
                                flags=self.INTERPOLATION_FLAG)
        return result


class DepthImage(ImageBase):
    MODALITY_NAME = 'depth_image'
    FILE_ENDING = '.tiff'
    INTERPOLATION_FLAG = cv2.INTER_AREA
    IMREAD_FLAG = cv2.IMREAD_ANYDEPTH

    def __init__(self) -> None:
        super(DepthImage, self).__init__()

    def load(self, path: str) -> None:
        self.data = imread(path)

    def save(self, path: str) -> None:
        imwrite(path, self.data)


class ColorImage(ImageBase):
    IMREAD_FLAG = cv2.IMREAD_ANYCOLOR
    INTERPOLATION_FLAG = cv2.INTER_LINEAR

    MODALITY_NAME = 'color_image'
    FILE_ENDING = '.png'

    def __init__(self) -> None:
        super(ColorImage, self).__init__()


register_modality(ColorImage)
register_modality(DepthImage)
