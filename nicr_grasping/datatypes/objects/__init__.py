from .. import logger as baselogger

logger = baselogger.getChild('objects')

from .scene_object import SceneObject
from .collision_objects import CollisionObject
from .graspable_objects import ObjectModel
from .scene import Scene
