from typing import Any

import numpy as np

from ..datatypes.grasp import Grasp2D


class Transform:
    def __init__(self) -> None:
        pass

    def __call__(self,
                 input: Any,
                 invert: bool = False) -> Any:

        if isinstance(input, np.ndarray):
            input_type = "image"
        elif isinstance(input, Grasp2D):
            input_type = "grasp"
        else:
            raise ValueError(f"Unknown input type {type(input)}")

        if invert:
            apply_func = getattr(self, f'apply_inverse_{input_type}')
        else:
            apply_func = getattr(self, f'apply_{input_type}')
        return apply_func(input)
