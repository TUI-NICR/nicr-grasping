import os.path as osp
from typing import Dict, List

import functools

import numpy as np

from ...datatypes.objects import ObjectModel

from ...utils.paths import graspnet_dataset_path

from .. import logger as baselogger
logger = baselogger.getChild('graspnet_eval')


def create_table_points(lx: float, ly: float, lz: float,
                        dx: float = 0, dy: float = 0, dz: float = 0,
                        grid_size: float = 0.01) -> np.ndarray:
    '''
    **Input:**
    - lx:
    - ly:
    - lz:
    **Output:**
    - numpy array of the points with shape (-1, 3).
    '''
    xmap = np.linspace(0, lx, int(lx/grid_size))
    ymap = np.linspace(0, ly, int(ly/grid_size))
    zmap = np.linspace(0, lz, int(lz/grid_size))
    xmap, ymap, zmap = np.meshgrid(xmap, ymap, zmap, indexing='xy')
    xmap += dx
    ymap += dy
    zmap += dz
    points = np.stack([xmap, ymap, zmap], axis=-1)
    points = points.reshape([-1, 3])
    return points


# maxsize of 1 as samples will be sorted by scene so we only really need to cache the current one
@functools.lru_cache(maxsize=1)
def _get_graspnet_objects(scene_id: int) -> Dict[int, ObjectModel]:

    global_object_ids = _get_object_ids(scene_id)

    logger.info(f'Loading {len(global_object_ids)} objects. This might take a while.')
    objects = {}

    for object_id in global_object_ids:
        object_model = ObjectModel.from_dir(
            osp.join(
                graspnet_dataset_path(),
                'models',
                f'{object_id:03d}'),
            model_name='textured')

        objects[object_id] = object_model

    return objects


# maxsize of 1 as samples will be sorted by scene so we only really need to cache the current one
@functools.lru_cache(maxsize=1)
def _get_object_ids(scene_id: int) -> List[int]:
    # we implement this function to guarantee that the order is equal to the contents
    # of the txt file of the dataset
    # the GraspNet class uses set() which might change the order
    return np.loadtxt(osp.join(graspnet_dataset_path(), 'scenes', f'scene_{int(scene_id):04d}', 'object_id_list.txt'), dtype=np.int32).tolist()
