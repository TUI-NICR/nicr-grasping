import abc
from typing import Any

class ModalityBase(abc.ABC):

    MODALITY_NAME = ''
    FILE_ENDING = ''

    def __init__(self) -> None:
        self.data : Any = None

    @abc.abstractmethod
    def load(self, path : str = None) -> None:
        pass

    @abc.abstractmethod
    def save(self, path : str) -> None:
        pass

    @classmethod
    def from_file(cls, file_path: str):
        obj = cls()
        obj.load(file_path)
        return obj
