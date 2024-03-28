import abc
from typing import Any


class ModalityBase(abc.ABC):

    MODALITY_NAME = ''
    FILE_ENDING = ''

    def __init__(self) -> None:
        self.data: Any = None

    @abc.abstractmethod
    def load(self, path: str) -> None:
        pass

    @abc.abstractmethod
    def save(self, path: str) -> None:
        pass

    @classmethod
    def from_file(cls, file_path: str) -> 'ModalityBase':
        obj = cls()
        obj.load(file_path)
        return obj
