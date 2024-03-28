import abc

import numpy as np

from pathlib import Path
from copy import deepcopy
from typing import Union, Any, TypeVar, Type


T = TypeVar('T', bound='Grasp')


class Grasp(abc.ABC):
    """Base class for grasps. Defines basic functions.
    """

    def __init__(self,
                 quality: float = 0,
                 object_id: Union[int, None] = None) -> None:
        self.quality = quality
        self.object_id = object_id

    def __eq__(self, __o: object) -> bool:
        if isinstance(__o, Grasp):
            # as quality is float we only care if it is close enough
            return bool(np.isclose(self.quality, __o.quality))

        return False

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.quality}, {self.object_id})"

    def plot(self, image: np.ndarray, **kwags: Any) -> np.ndarray:
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

    def save(self, file_path: Union[Path, str]) -> None:
        raise NotImplementedError()

    @classmethod
    def load(cls: Type[T], file_path: Union[Path, str]) -> T:
        raise NotImplementedError()

    # def to_type(self, to_type : Any) -> Any:
    #     return CONVERTER_REGISTRY.convert(self, to_type)

    def copy(self: T) -> T:
        return deepcopy(self)
