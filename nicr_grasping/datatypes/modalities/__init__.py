from .modality_base import ModalityBase

MODALITY_NAME_MAPPING = {}

def get_modality_from_name(modality_name: str) -> ModalityBase:
    return MODALITY_NAME_MAPPING[modality_name]

def register_modality(modality: ModalityBase) -> None:
    global MODALITY_NAME_MAPPING
    MODALITY_NAME_MAPPING[modality.MODALITY_NAME] = modality

from .image import *
