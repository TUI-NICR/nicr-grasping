from typing import Type, List
from .interface_base import DatasetInterface


def get_interface(dataset_name: str) -> Type[DatasetInterface]:
    if dataset_name == 'cornell':
        from .cornell_interface import CornellInterface
        return CornellInterface
    elif dataset_name == 'graspnet':
        from .graspnet_interface import GraspNetInterface
        return GraspNetInterface
    else:
        raise ValueError(f'No interface for dataset "{dataset_name}"!')


def get_interfaces() -> List[str]:
    return ['cornell', 'graspnet']
