from dataclasses import dataclass


@dataclass(frozen=True)
class ParalellGripperParameters:
    # width of the yaws
    finger_width: float = 0.01
    # length of the yaws
    finger_depth: float = 0.04

    # offset of the base origin relative to grasp pose
    base_offset: float = 0.02
    # thickness of the base
    base_depth: float = 0.01
