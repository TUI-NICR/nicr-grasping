from typing import Optional, Tuple

import numpy as np

from .base import Transform
from ..datatypes.grasp.grasp_2d import Grasp2D


class CropShortestEdge(Transform):

    def __init__(self) -> None:
        super().__init__()

        self._crop_size = 0
        self._crop_height = 0
        self._crop_width = 0
        self._crop_y = 0
        self._crop_x = 0

        self._orig_shape: Tuple[int, ...] = (0, 0)

        self._is_initialized = False

    def initialize(self, input_image: np.ndarray) -> None:
        self._orig_shape = input_image.shape

        # do center crop to shortest edge
        height, width, _ = input_image.shape
        if height < width:
            self._crop_size = height
            self._crop_height = self._crop_size
            self._crop_width = self._crop_size
            self._crop_y = 0
            self._crop_x = (width - self._crop_size) // 2

        else:
            self._crop_size = width
            self._crop_height = self._crop_size
            self._crop_width = self._crop_size
            self._crop_y = (height - self._crop_size) // 2
            self._crop_x = 0

        self._is_initialized = True

    def apply_image(self, input_image: np.ndarray) -> np.ndarray:
        assert self._is_initialized, "CropShortestEdge must be initialized before use"

        output_image = input_image[self._crop_y:self._crop_y + self._crop_height, self._crop_x:self._crop_x + self._crop_width]
        return output_image

    def apply_inverse_image(self, input_image: np.ndarray) -> np.ndarray:
        output_image = np.zeros(self._orig_shape, dtype=input_image.dtype)
        output_image[self._crop_y:self._crop_y + self._crop_height, self._crop_x:self._crop_x + self._crop_width] = input_image

        return output_image

    def apply_grasp(self, input_grasp: Grasp2D) -> Grasp2D:
        assert self._is_initialized, "CropShortestEdge must be initialized before use"

        output_grasp = input_grasp.copy()
        output_grasp.translate(-np.array([[self._crop_x, self._crop_y]]))

        return output_grasp

    def apply_inverse_grasp(self, input_grasp: Grasp2D) -> Grasp2D:
        assert self._is_initialized, "CropShortestEdge must be initialized before use"

        output_grasp = input_grasp.copy()
        output_grasp.translate(np.array([[self._crop_x, self._crop_y]]))

        return output_grasp
