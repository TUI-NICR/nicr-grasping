from typing import Type, Dict

from .modality_base import ModalityBase

MODALITY_NAME_MAPPING: Dict[str, Type[ModalityBase]] = {}


def get_modality_from_name(modality_name: str) -> Type[ModalityBase]:
    return MODALITY_NAME_MAPPING[modality_name]


def register_modality(modality: Type[ModalityBase]) -> None:
    global MODALITY_NAME_MAPPING
    MODALITY_NAME_MAPPING[modality.MODALITY_NAME] = modality


from .image import *
