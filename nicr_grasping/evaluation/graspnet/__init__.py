from dataclasses import dataclass

from importlib import resources
import json


ID_TO_NAME_MAPPING_FILE = str(resources.path('nicr_grasping.evaluation.graspnet', 'id_to_name_mapping.json'))
with open(ID_TO_NAME_MAPPING_FILE, 'r') as f:
    ID_TO_NAME_MAPPING = json.load(f)

NAME_TO_ID_MAPPING = {v: int(k) for k, v in ID_TO_NAME_MAPPING.items()}


def id_to_name(object_id: int) -> str:
    return ID_TO_NAME_MAPPING[str(object_id)]


def name_to_id(object_name: str) -> int:
    return NAME_TO_ID_MAPPING[object_name]


@dataclass(frozen=True)
class GraspnetStats:
    num_scenes: int
    num_unique_objects: int
    num_samples_per_scene: int


GRASPNET_STATS_PER_SPLIT = {
    'train': GraspnetStats(
        num_scenes=100,
        num_unique_objects=240,
        num_samples_per_scene=256),
    'test': GraspnetStats(
        num_scenes=30,
        num_unique_objects=750,
        num_samples_per_scene=256),
    'test_seen': GraspnetStats(
        num_scenes=30,
        num_unique_objects=280,
        num_samples_per_scene=256),
    'test_similar': GraspnetStats(
        num_scenes=30,
        num_unique_objects=258,
        num_samples_per_scene=256),
    'test_novel': GraspnetStats(
        num_scenes=30,
        num_unique_objects=242,
        num_samples_per_scene=256),
}
