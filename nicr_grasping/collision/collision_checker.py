from typing import Dict, Any

import numpy as np

from ..datatypes.grasp import Grasp3D

# TODO: implement a check_collision method which computes collisions for
#      multiple grasps at once. This would change the format of collision_info
#      to be a list of dicts where each dict contains the collision info for one grasp


class CollisionChecker:
    # keys which the checker will fill in the collision info dict
    INFO_KEYS = {
        'collision_base': 0,
        'collision_left': 0,
        'collision_right': 0,
    }

    def __init__(self) -> None:
        pass

    def check_collision(self,
                        grasp: Grasp3D,
                        frames: Dict[str, np.ndarray] = {},
                        **kwargs: Any) -> bool:
        raise NotImplementedError

    @property
    def collision_info(self) -> Dict[str, Any]:
        raise NotImplementedError
