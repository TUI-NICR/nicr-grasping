import importlib
from collections import defaultdict
from typing import Dict, Callable, Type, TypeAlias, Any
import logging
from .. import logger as baselogger

from ... import GRASPNET_INSTALLED, GRASP_DETECTION_INSTALLED

logger = baselogger.getChild('conversion')

__all__ = ['CONVERTER_REGISTRY']

ConverterFunctionType: TypeAlias = Callable[[Any], Any]


class ConverterRegistry:
    CONVERSIONS: Dict[Type, Dict[Type, ConverterFunctionType]] = defaultdict(dict)

    def __init__(self, logger: logging.Logger) -> None:
        self._logger = logger

    def register(self,
                 from_type: Type, to_type: Type, converter_func: ConverterFunctionType) -> None:
        self._logger.info(f'Registered conversion from {from_type} to {to_type}')
        self.CONVERSIONS[from_type][to_type] = converter_func

    def convert(self,
                obj: Any,
                goal_type: Type) -> Any:
        from_type = type(obj)
        converters = self.CONVERSIONS[from_type]
        if goal_type not in converters:
            raise RuntimeError(f'No conversion from {from_type} to {goal_type} registered!')

        return converters[goal_type](obj)


CONVERTER_REGISTRY: ConverterRegistry = ConverterRegistry(logger)

if GRASPNET_INSTALLED:
    from .graspnet import *

if GRASP_DETECTION_INSTALLED:
    from .cornell import *

from .dexnet import *
