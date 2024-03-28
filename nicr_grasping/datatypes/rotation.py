import enum
import abc
import numpy as np

from typing import Optional

# from scipy.spatial.transform import Rotation as R


def rad2deg(angle_rad: float) -> float:
    return angle_rad / np.pi * 180


def deg2rad(angle_deg: float) -> float:
    return angle_deg / 180 * np.pi


class RotationType(enum.Enum):
    ROTATION_2D = 0
    ROTATION_3D = 1


class RotationBase(abc.ABC):
    def __init__(self) -> None:
        pass

    @abc.abstractmethod
    def to_mat(self) -> np.ndarray:
        pass


class Rotation2D(RotationBase):
    def __init__(self,
                 angle_deg: Optional[float] = None,
                 angle_rad: Optional[float] = None) -> None:
        super().__init__()
        assert angle_deg is not None or angle_rad is not None
        if angle_deg is not None and angle_rad is not None:
            raise ValueError("Only one of angle_deg or angle_rad can be set")

        if angle_rad is not None:
            self.angle = angle_rad
        if angle_deg is not None:
            self.angle = deg2rad(angle_deg)

    def to_mat(self) -> np.ndarray:
        rotmat = np.array(
            [
                [np.cos(self.angle), -np.sin(self.angle)],
                [np.sin(self.angle), np.cos(self.angle)],
            ]
        )
        return rotmat
