import abc

import numpy as np

from pathlib import Path
from copy import deepcopy
from typing import Union


class Grasp(abc.ABC):
    """Base class for grasps. Defines basic functions.
    """
    def __init__(self, quality:float = 0):
        self.quality = quality

    def __eq__(self, __o: object) -> bool:
        if isinstance(__o, Grasp):
            # as quality is float we only care if it is close enough
            return np.isclose(self.quality, __o.quality)

        return False

    def plot(self, image : np.ndarray, **kwags):
        """Function for plotting grasp in image.

        Parameters
        ----------
        image : np.ndarray
            Base image to plot grasp into.

        Raises
        ------
        NotImplementedError
            [description]
        """
        raise NotImplementedError()

    def save(self, file_path: Union[Path, str]):
        raise NotImplementedError()

    @classmethod
    def load_from_file(cls, file_path: Union[Path, str]):
        raise NotImplementedError()

    # def to_type(self, to_type : Any) -> Any:
    #     return CONVERTER_REGISTRY.convert(self, to_type)

    def copy(self):
        return deepcopy(self)