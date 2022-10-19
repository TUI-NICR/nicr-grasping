import importlib
from collections import defaultdict
import logging
from .. import logger as baselogger

from ... import GRASPNET_INSTALLED, GRASP_DETECTION_INSTALLED

logger = baselogger.getChild('conversion')

__all__ = ['CONVERTER_REGISTRY']


class ConverterRegistry:
    CONVERSIONS = defaultdict(dict)
    def __init__(self, logger) -> None:
        self._logger = logger

    def register(self, from_type, to_type, converter_func):
        self._logger.info(f'Registered conversion from {from_type} to {to_type}')
        self.CONVERSIONS[from_type][to_type] = converter_func

    def convert(self, obj, goal_type):
        from_type = type(obj)
        converters = self.CONVERSIONS[from_type]
        if goal_type not in converters:
            raise RuntimeError(f'No conversion from {from_type} to {goal_type} registered!')

        return converters[goal_type](obj)

CONVERTER_REGISTRY = ConverterRegistry(logger)

if GRASPNET_INSTALLED:
    from .graspnet import *

if GRASP_DETECTION_INSTALLED:
    from .cornell import *
