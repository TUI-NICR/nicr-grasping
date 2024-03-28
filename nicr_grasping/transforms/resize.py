from typing import Tuple

import cv2
import numpy as np

from .base import Transform
from ..datatypes.grasp.grasp_2d import Grasp2D


class Resize(Transform):
    def __init__(self, new_shape: Tuple[int, ...]) -> None:
        super().__init__()

        self._new_shape = new_shape
        self._factor_h = 0.0
        self._factor_w = 0.0

    def initialize(self, image: np.ndarray) -> None:
        self._orig_shape = image.shape

        self._factor_h = self._new_shape[0] / self._orig_shape[0]
        self._factor_w = self._new_shape[1] / self._orig_shape[1]

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        return cv2.resize(image, self._new_shape)

    def apply_inverse_image(self, image: np.ndarray) -> np.ndarray:
        return cv2.resize(image, self._orig_shape)

    def apply_grasp(self, grasp: Grasp2D) -> Grasp2D:
        grasp._points *= np.array([[self._factor_w, self._factor_h]])

        return grasp

    def apply_inverse_grasp(self, grasp: Grasp2D) -> Grasp2D:
        grasp._points *= np.array([[1 / self._factor_w, 1 / self._factor_h]])

        return grasp
